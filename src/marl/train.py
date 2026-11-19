# src/marl/train.py
"""
Trainer of the communication-free MARL baseline (shared-parameter PPO). Design and evaluation protocol:
.ignore/revision/marl_baseline_design.md.

Each round: save the current policy; run a batch of missions in parallel, each an ordinary single-mission run of the batch runner (in-process
drones; the strategy samples its choices and writes one record per decision); turn the records into advantages; update the policy with PPO.
Every few rounds the greedy policy is run on validation seeds and the best checkpoint is kept.

Rewards: the team makespan objective. A drone's decisions form a sequence; between two consecutive decisions the reward is minus the steps that
passed (divided by STEP_REF); after the last decision it is minus the steps to the end of the mission (the last delivery, or the step limit if
parcels remain), minus UNDELIVERED_PENALTY times the share of parcels left undelivered, minus LOSS_PENALTY per drone lost. The sum from any decision
on equals minus the time left to the end of the mission, and every drone's sequence shares the mission outcome. Advantages: GAE (gamma 1).

Critic (--critic):
  central  MAPPO: a centralised value function that also sees the global state of the mission (every drone's position and whether it carries, the
           open parcels), reconstructed from the mission logs. Training only; the deployed policy never sees it and does not contain it.
  local    IPPO: the value head of the policy, from the drone's own features only.

Seeds: training uses replicate indices TRAIN_REPLICATE_BASE and above, validation 50 to 57; evaluation (outside this file) uses 0 to 15.
Resumable: the full training state is saved after every round (state.pt) and picked up if the run is started again with the same --out.
Usage:
  python -m src.marl.train --out experiments/marl/seed0 --seed 0 --max-rounds 100 --jobs 20 --critic central
"""
import argparse
import glob
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import torch

from src.marl import policy_np
from src.marl.features import STEP_REF
from src.marl.global_state import MissionLog
from src.marl.policy_torch import CentralCritic, PolicyNet, collate, collate_global

# Version 2 (2026-10-16): the policy sees the drones' declared speeds and the trip time, and the team return counts time and energy equally, each against its own ideal for the mission (see
# marl_baseline_design.md section 10); the earlier policies (experiments/marl, version 1) were trained on the world before the engine, perception, speed and bid changes and are not used.
UNDELIVERED_PENALTY = 1.0
LOSS_PENALTY = 0.5
TRAIN_REPLICATE_BASE = 100
VALIDATION_REPLICATES = list(range(50, 58))
# Version 2.1 (2026-10-16): a wide training mixture, so that what is learned carries over to other configurations. The configurations held out for the transfer test are never drawn.
HELD_OUT = {"A2_P8", "A3_P12", "A16_P16", "A10_P24", "A20_P24"}
_INFEASIBLE_CORNER = {"A2_P20", "A2_P24", "A2_P26", "A2_P30", "A2_P36", "A3_P36", "A4_P36"}        # the battery-bound configurations no strategy completes
def _grid():
    for path in ("scripts/paper_runs/grid64.txt", ".ignore/sandbox/grid64.txt"):          # same 64-configuration list in both places
        try:
            return open(path).read().split()
        except OSError:
            pass
    return ["A4_P12", "A4_P24", "A6_P18", "A8_P12", "A8_P24", "A12_P24"]
TRAIN_CONFIGS = sorted((set(_grid()) | {"A4_P12", "A6_P18"}) - HELD_OUT - _INFEASIBLE_CORNER)
# arm tag -> the drones' speeds (alternating over the fleet); `limited_hs3` is only for the transfer test and is never trained on
SPEED_PROFILES = {"limited": [1.0], "limited_hs": [1.0, 0.5], "limited_hs75": [1.0, 0.75], "limited_hs3": [1.0, 0.75, 0.5]}
VALIDATION_CONFIGS = ["A4_P12", "A8_P24", "A12_P24"]


def run_episode(cfg: str, sensing: str, replicate: int, weights: str, work: str, sample: bool, central: bool,
                repo: str) -> Dict:
    """One mission with the given policy; returns the outcome and, when sampling, the decision records (with the global state if central)."""
    out = os.path.join(work, f"{cfg}_{sensing}_{replicate}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=".:src", SDL_VIDEODRIVER="dummy", PYTHONHASHSEED="0", MARL_WEIGHTS=weights)
    if sample:
        env["MARL_TRACE_DIR"] = os.path.join(out, "trace")
    cmd = [sys.executable, "-m", "src.run_batch", "--configs", cfg, "--arms", f"marl_ippo_{sensing}",
           "--replicates", str(replicate + 1), "--only-replicates", str(replicate), "--out", out, "--jobs", "1", "--mem-gb", "6"]
    subprocess.run(cmd, env=env, cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
    files = glob.glob(os.path.join(out, "*", "experiment_*", "experiment.json"))
    if not files:
        return {"ok": False, "cfg": cfg}
    d = json.load(open(files[0]))
    r = d["results"][0]
    m, t = r["metrics"], r.get("termination") or {}
    n_agents = d["scale"]["n_agents"]
    prof = SPEED_PROFILES.get(sensing, [1.0])
    speeds = [prof[i % len(prof)] for i in range(n_agents)]
    t_ideal, e_ideal = ideals(d, speeds)
    result = {"ok": True, "cfg": cfg, "sensing": sensing, "status": r.get("status"), "makespan": m["mission_makespan_steps"],
              "energy": m["total_energy_consumed"], "t_ideal": t_ideal, "e_ideal": e_ideal,
              "delivered": m["tasks_completed"], "parcels": d["scale"]["n_parcels"],
              "lost": t.get("dead_total", 0) + t.get("lost_in_field", 0), "records": []}
    if sample:
        for path in glob.glob(os.path.join(out, "trace", "*.pkl")):
            with open(path, "rb") as fh:
                while True:
                    try:
                        result["records"].append(pickle.load(fh))
                    except EOFError:
                        break
        if central and result["records"]:
            agents_path = glob.glob(os.path.join(os.path.dirname(files[0]), "marl_ippo_*", "agents.json"))[0]
            log = MissionLog(agents_path, files[0])
            for rec in result["records"]:
                rec["g_drones"], rec["g_parcels"], rec["g_scalars"] = log.at(rec["step"], rec["agent"])
    shutil.rmtree(out, ignore_errors=True)
    return result


def ideals(d: Dict, speeds: List[float]):
    """The two ideals of the mission, as in SPPDEC's equal-importance bid: the ideal finish time (all flight work, station to pickup and pickup to station, over the fleet's total speed) and the ideal
    energy (every parcel flown by the cheapest consumption, here the nominal one, from the station)."""
    from src.artifacts import Drone
    from src import sim_clock
    st = d["station"]; centre = (st["col"] + st["w"] // 2, st["row"] + st["h"] // 2)
    work = e_ideal = 0.0
    for p in d["parcels"]:
        dist = ((p["col"] - centre[0]) ** 2 + (p["row"] - centre[1]) ** 2) ** 0.5        # straight-line distance, as the simulator measures it
        w = float(p.get("weight", 1.0))
        work += 2.0 * dist * sim_clock.STEPS_PER_CELL
        e_ideal += Drone.BASE_COST_PER_CELL * dist + Drone.BASE_COST_PER_CELL * dist * (1.0 + w * Drone.WEIGHT_FACTOR) + 2.0 * Drone.PICK_DROP_COST * (1.0 + w * Drone.WEIGHT_FACTOR)
    return max(1.0, work / sum(speeds)), max(1e-6, e_ideal)


def terminal_penalty(res: Dict) -> float:
    undelivered = 1.0 - res["delivered"] / max(1, res["parcels"])
    return UNDELIVERED_PENALTY * undelivered + LOSS_PENALTY * res["lost"]


def validation_score(res: Dict) -> float:
    """Minus the mission time (scaled) and the terminal penalties: higher is better."""
    return -res["makespan"] / res["t_ideal"] - res["energy"] / res["e_ideal"] - terminal_penalty(res)


def rewards_for(res: Dict, shape: float = 0.0) -> Dict[str, List]:
    """Per drone, the records sorted by step with the reward that follows each decision.
    shape > 0 adds to each decision the cost of the trip it chose, in the two units of the mission objective: minus the trip's flight time over the mission's ideal time
    and minus the trip's energy over the mission's ideal energy, times `shape` (the drone's own cost of its choice; a team-return term that does not depend on the other
    drones' choices, so the signal of one decision is not drowned in the outcome of the whole mission)."""
    by_agent = defaultdict(list)
    for rec in res["records"]:
        by_agent[rec["agent"]].append(rec)
    out = {}
    for agent, recs in by_agent.items():
        recs.sort(key=lambda r: r["step"])
        rewards = []
        for k, rec in enumerate(recs):
            own = 0.0
            if shape > 0.0 and len(rec["tasks"]) and 0 <= rec["action"] < len(rec["tasks"]):
                row = rec["tasks"][rec["action"]]
                own = -shape * (float(row[10]) + float(row[4]) * res.get("capacity", 100.0) / res["e_ideal"])
            if k + 1 < len(recs):
                rewards.append(-(recs[k + 1]["step"] - rec["step"]) / res["t_ideal"] + own)
            else:
                rewards.append(-max(0.0, res["makespan"] - rec["step"]) / res["t_ideal"] - res["energy"] / res["e_ideal"] - terminal_penalty(res) + own)
        out[agent] = (recs, rewards)
    return out


def gae(rewards: List[float], values: List[float], lam: float) -> List[float]:
    adv, nxt_adv = [0.0] * len(rewards), 0.0
    for k in reversed(range(len(rewards))):
        v_next = values[k + 1] if k + 1 < len(values) else 0.0
        delta = rewards[k] + v_next - values[k]
        nxt_adv = delta + lam * nxt_adv
        adv[k] = nxt_adv
    return adv


def central_values(critic: CentralCritic, records: List[Dict], device: str) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(records), 4096):
            out.append(critic(*collate_global(records[s:s + 4096], device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def save_state(path: str, obj: Dict) -> None:
    """
    Save the training state so that a power cut cannot lose it: write to a temporary file and flush it to the disk, keep the previous good state
    as state.prev.pt, then put the new one in place and flush the directory entry. A reader falls back to state.prev.pt if state.pt is missing or unreadable.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.exists(path):
        os.replace(path, os.path.join(os.path.dirname(path), "state.prev.pt"))
    os.replace(tmp, path)
    dfd = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def load_state(path: str, device: str):
    """The latest readable training state (state.pt, else state.prev.pt), or None if there is none."""
    for candidate in (path, os.path.join(os.path.dirname(path), "state.prev.pt")):
        if os.path.exists(candidate):
            try:
                st = torch.load(candidate, map_location=device, weights_only=False)
                if candidate != path:
                    print(f"[MARL] {os.path.basename(path)} was missing or unreadable; using {os.path.basename(candidate)}", flush=True)
                return st
            except Exception as exc:                                  # truncated or corrupt file
                print(f"[MARL] could not read {candidate}: {exc}", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=1.5, help="wall-clock training budget")
    ap.add_argument("--max-rounds", type=int, default=10 ** 6)
    ap.add_argument("--episodes", type=int, default=200, help="missions per round")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--critic", choices=["central", "local"], default="central")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--shape", type=float, default=0.0, help="weight of the cost of the chosen trip added to each decision's reward (0: the team return only)")
    ap.add_argument("--minibatch", type=int, default=2048, help="decisions per gradient step")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--sensing", nargs="+", default=["limited", "limited_hs", "limited_hs75"])
    ap.add_argument("--configs", nargs="+", default=TRAIN_CONFIGS)
    ap.add_argument("--validate-every", type=int, default=5)
    ap.add_argument("--repo", default=os.getcwd(), help="directory the missions run in (a frozen copy of the source keeps a long run independent of edits)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    args.out = os.path.abspath(args.out)
    work = os.path.join(args.out, "work")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    central = args.critic == "central"
    print(f"[MARL] training on {device}, critic {args.critic}", flush=True)
    model = PolicyNet(policy_np.init_params(args.seed)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    critic = CentralCritic().to(device) if central else None
    copt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr) if central else None
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    log = open(os.path.join(args.out, "training_log.jsonl"), "a")
    counter = TRAIN_REPLICATE_BASE + args.seed * 100000
    best_val = -float("inf")
    episodes_done = 0
    start_round, elapsed_before = 0, 0.0
    # Resumption: the whole training state is saved after every round (state.pt); a run that is interrupted (power cut, crash) continues from
    # the last saved round, with the same optimiser state, random generators, episode counter and best validation score. At most the round in
    # progress (about a minute of missions) is lost.
    state_path = os.path.join(args.out, "state.pt")
    st = load_state(state_path, device)
    if st is not None:
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if central:
            critic.load_state_dict(st["critic"])
            copt.load_state_dict(st["copt"])
        start_round, counter, best_val = st["round"] + 1, st["counter"], st["best_val"]
        episodes_done, elapsed_before = st["episodes_done"], st["elapsed"]
        random.setstate(st["py_rng"])
        np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"].cpu())
        print(f"[MARL] resumed from round {st['round']} ({episodes_done} missions done)", flush=True)
    started = time.time() - elapsed_before
    pool = ThreadPoolExecutor(max_workers=args.jobs)
    for rnd in range(start_round, args.max_rounds):
        if time.time() - started > args.hours * 3600:
            break
        weights = os.path.abspath(os.path.join(args.out, "current.npz"))
        policy_np.save(weights, model.numpy_params())
        jobs = []
        for _ in range(args.episodes):
            jobs.append((random.choices(args.configs, weights=[1.0 / (int(c[1:].split("_")[0]) + 1) for c in args.configs])[0], random.choice(args.sensing), counter))     # larger fleets cost more per mission: drawn less often
            counter += 1
        futures = [pool.submit(run_episode, c, s, r, weights, work, True, central, args.repo) for c, s, r in jobs]
        results = [f.result() for f in futures]
        results = [r for r in results if r["ok"] and r["records"]]
        episodes_done += len(results)
        if not results:
            print("[MARL] a round produced no usable episodes; stopping", flush=True)
            break
        records, advantages, returns = [], [], []
        for res in results:
            seqs = rewards_for(res, args.shape)
            ep_records = [rec for recs, _ in seqs.values() for rec in recs]
            v_all = central_values(critic, ep_records, device) if central else None
            offset = 0
            for recs, rewards in seqs.values():
                values = list(v_all[offset:offset + len(recs)]) if central else [rec["value"] for rec in recs]
                offset += len(recs)
                adv = gae(rewards, values, args.lam)
                records += recs
                advantages += adv
                returns += [a + v for a, v in zip(adv, values)]
        n = len(records)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
        old_logp = torch.tensor([r["logp"] for r in records], dtype=torch.float32, device=device)
        actions = torch.tensor([r["action"] for r in records], dtype=torch.long, device=device)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
        for _ in range(args.epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, args.minibatch):
                idx = perm[s:s + args.minibatch]
                batch = [records[i] for i in idx]
                x_self, tasks, task_mask, peers, peer_mask = collate(batch, device=device)
                logits, value = model(x_self, tasks, task_mask, peers, peer_mask)
                logp_all = torch.log_softmax(logits, dim=-1)
                logp = logp_all.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(logp - old_logp[idx])
                surr = torch.min(ratio * adv_t[idx], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t[idx])
                policy_loss = -surr.mean()
                prob = torch.exp(logp_all)
                entropy = -torch.where(task_mask, prob * logp_all, torch.zeros_like(prob)).sum(-1).mean()
                loss = policy_loss - args.entropy * entropy
                if central:
                    cv = critic(*collate_global(batch, device))
                    value_loss = ((cv - ret_t[idx]) ** 2).mean()
                    copt.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    copt.step()
                else:
                    value_loss = ((value - ret_t[idx]) ** 2).mean()
                    loss = loss + 0.5 * value_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy)
                stats["updates"] += 1
        u = max(1, stats.pop("updates"))
        stats = {k: v / u for k, v in stats.items()}
        entry = {"round": rnd, "episodes": len(results), "episodes_total": episodes_done, "decisions": n,
                 "mean_makespan": float(np.mean([r["makespan"] for r in results])),
                 "mean_delivered_share": float(np.mean([r["delivered"] / max(1, r["parcels"]) for r in results])),
                 "mean_return": float(np.mean(returns)), "hours": (time.time() - started) / 3600.0, **stats}
        if (rnd + 1) % args.validate_every == 0:
            policy_np.save(weights, model.numpy_params())
            vfut = [pool.submit(run_episode, c, s, r, weights, work, False, False, args.repo)
                    for c in VALIDATION_CONFIGS for s in args.sensing for r in VALIDATION_REPLICATES]
            vres = [f.result() for f in vfut]
            vres = [r for r in vres if r["ok"]]
            if vres:
                val = float(np.mean([validation_score(r) for r in vres]))
                entry["validation_return"] = val
                entry["validation_makespan"] = float(np.mean([r["makespan"] for r in vres]))
                if val > best_val:
                    best_val = val
                    policy_np.save(os.path.join(args.out, "best.npz"), model.numpy_params())
                    entry["new_best"] = True
        save_state(state_path, {"model": model.state_dict(), "opt": opt.state_dict(),
                                "critic": critic.state_dict() if central else None, "copt": copt.state_dict() if central else None,
                                "round": rnd, "counter": counter, "best_val": best_val, "episodes_done": episodes_done,
                                "elapsed": time.time() - started, "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                                "torch_rng": torch.get_rng_state()})
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print("[MARL]", json.dumps(entry), flush=True)
    policy_np.save(os.path.join(args.out, "final.npz"), model.numpy_params())
    if not os.path.exists(os.path.join(args.out, "best.npz")):
        policy_np.save(os.path.join(args.out, "best.npz"), model.numpy_params())
    print(f"[MARL] done: {episodes_done} episodes in {(time.time() - started) / 3600.0:.2f} h", flush=True)


if __name__ == "__main__":
    main()
