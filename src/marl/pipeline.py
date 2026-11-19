# src/marl/pipeline.py
"""
The whole MARL training protocol, unattended and resumable (design: .ignore/revision/marl_baseline_design.md):
  1. a frozen copy of the source, so that edits made while this runs cannot reach the missions;
  2. a fixed hyperparameter search on validation seeds (TUNING_GRID, TUNING_HOURS each, MAPPO), the best by the mean of the last three validation scores;
  3. FINAL_SEEDS independent training runs with the chosen settings (MAPPO, the primary variant): experiments/marl/seed<k>/best.npz;
  4. ABLATION_SEEDS training runs with the local critic (IPPO): experiments/marl/local_seed<k>/best.npz.
Every final run has the same budget: FINAL_ROUNDS rounds of 200 missions (wall-clock hours are only a safety cap).
Resumable at two levels: a finished run (final.npz present) is skipped, and an unfinished run continues from its last saved round (state.pt), so the script can simply be started again after
a power cut or a crash (append to the log: `nohup python -m src.marl.pipeline >> experiments/marl/pipeline.log 2>&1 &`).
Usage: nohup python -m src.marl.pipeline > experiments/marl/pipeline.log 2>&1 &
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = "experiments/marl"
TUNING_GRID = [(3e-4, 0.01), (1e-3, 0.01), (1e-4, 0.01), (3e-4, 0.003), (3e-4, 0.03), (1e-3, 0.003)]   # (learning rate, entropy weight)
TUNING_ROUNDS = 30          # 200 missions per round: 6,000 missions per tuning run
FINAL_ROUNDS = 100          # 20,000 missions (about 900,000 decisions) per final run; the same for every seed and both critics
HOURS_CAP = 8.0             # safety cap only; the budget is the number of rounds, so that every run gets the same number of missions
FINAL_SEEDS = 5
ABLATION_SEEDS = 3
JOBS = 20


def train(out: str, seed: int, rounds: int, critic: str, lr: float, entropy: float, repo: str, episodes: int, jobs: int) -> None:
    if os.path.exists(os.path.join(out, "final.npz")):
        print(f"[PIPELINE] skip {out} (finished)", flush=True)
        return
    if not os.path.exists(os.path.join(out, "state.pt")):
        shutil.rmtree(out, ignore_errors=True)         # no saved state: start clean; with one, train.py resumes from it
    cmd = [sys.executable, "-m", "src.marl.train", "--out", out, "--seed", str(seed), "--max-rounds", str(rounds), "--hours", str(HOURS_CAP), "--jobs", str(jobs), "--episodes", str(episodes),
           "--critic", critic, "--lr", str(lr), "--entropy", str(entropy), "--repo", repo]
    print("[PIPELINE]", " ".join(cmd[2:]), flush=True)
    env = dict(os.environ, PYTHONPATH=".:src")
    subprocess.run(cmd, env=env, cwd=repo, check=False)


def tuning_score(out: str) -> float:
    rows = [json.loads(l) for l in open(os.path.join(out, "training_log.jsonl"))]
    val = [r["validation_return"] for r in rows if "validation_return" in r]
    return sum(val[-3:]) / max(1, len(val[-3:])) if val else -float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--episodes", type=int, default=200, help="missions per round")
    ap.add_argument("--tuning-rounds", type=int, default=TUNING_ROUNDS)
    ap.add_argument("--final-rounds", type=int, default=FINAL_ROUNDS)
    ap.add_argument("--final-seeds", type=int, default=FINAL_SEEDS)
    ap.add_argument("--ablation-seeds", type=int, default=ABLATION_SEEDS)
    ap.add_argument("--parallel", type=int, default=1, help="training runs at the same time (the CPU is not saturated by one run)")
    ap.add_argument("--jobs", type=int, default=JOBS, help="missions in parallel inside one run")
    ap.add_argument("--grid-size", type=int, default=len(TUNING_GRID), help="use only the first settings of the tuning grid (smoke tests)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    grid = TUNING_GRID[:args.grid_size]
    os.makedirs(root, exist_ok=True)
    repo = os.path.join(root, "_repo")
    if not os.path.exists(os.path.join(repo, "src")):
        os.makedirs(repo, exist_ok=True)
        shutil.copytree("src", os.path.join(repo, "src"), ignore=shutil.ignore_patterns("__pycache__"))
        for f in (".env",):
            if os.path.exists(f):
                shutil.copy(f, repo)
    print("[PIPELINE] source copy at", repo, flush=True)
    def phase(tasks):
        """Run training runs, `args.parallel` at a time; each task is (out, seed, rounds, critic, lr, entropy)."""
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            futures = [pool.submit(train, out, seed, rounds, critic, lr_, ent_, repo, args.episodes, args.jobs)
                       for out, seed, rounds, critic, lr_, ent_ in tasks]
            for f in futures:
                f.result()

    phase([(os.path.join(root, "tune", f"t{i}"), 1000 + i, args.tuning_rounds, "central", lr, ent) for i, (lr, ent) in enumerate(grid)])
    results = {}
    for i, (lr, ent) in enumerate(grid):
        results[i] = tuning_score(os.path.join(root, "tune", f"t{i}"))
        print(f"[PIPELINE] tuning {i}: lr {lr} entropy {ent} -> validation score {results[i]:.4f}", flush=True)
    best = max(results, key=results.get)
    if results[best] == -float("inf"):
        raise SystemExit("[PIPELINE] no tuning run produced a validation score (too few rounds for --validate-every); not choosing a setting by default")
    lr, ent = grid[best]
    json.dump({"scores": results, "chosen": {"lr": lr, "entropy": ent}}, open(os.path.join(root, "tuning_result.json"), "w"), indent=1)
    print(f"[PIPELINE] chosen: lr {lr}, entropy {ent}", flush=True)
    phase([(os.path.join(root, f"seed{k}"), k, args.final_rounds, "central", lr, ent) for k in range(args.final_seeds)]
          + [(os.path.join(root, f"local_seed{k}"), k, args.final_rounds, "local", lr, ent) for k in range(args.ablation_seeds)])
    print("[PIPELINE] all training finished", flush=True)


if __name__ == "__main__":
    main()
