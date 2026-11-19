# analysis/revision/peer_choice_fit.py
"""
What explains which parcel a peer takes next? A conditional-logit (softmax over the open parcels) fitted to the logged picks of the fleet, with the truth (peers' true
positions at the decision) so that the question is about the choice rule and not about stale positions. Feature sets, each nested in the next:
  M0 pickup distance                                  (the cost SPPDEC uses today)
  M1 + delivery leg (distance from the parcel to the station centre)   (the whole trip)
  M2 + competition (how much nearer the closest other peer is to the parcel than this peer is)
  M3 + distance from the peer's start cell            (a territory effect)
  M4 + the parcel's rank by pickup distance among the open ones
Fitted on odd-numbered missions, scored on even-numbered ones: mean log-likelihood of the true choice, top-1 and top-3 accuracy (chance is 1 / open parcels).
Usage: python -m analysis.revision.peer_choice_fit <batch dir> <arm>
"""
import glob
import json
import sys

import numpy as np


def man(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def collect(root, arm):
    data = []          # list of (mission index, features [n_candidates x k], index of true, peer index)
    mission = 0
    for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
        for d in sorted(glob.glob(ep.replace("experiment.json", f"{arm}_instance_*"))):
            mission += 1
            fleet = json.load(open(f"{d}/fleet.json"))
            ppos = {p["parcel_id"]: tuple(p["spawn_cell"]) for p in fleet["parcels"]}
            exp = json.load(open(ep))
            st = exp["station"]
            centre = (st["col"] + st["w"] // 2, st["row"] + st["h"] // 2)
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            cells = {a["agent_id"]: [tuple(t["cell"]) for t in a["trajectory"]] for a in ags}
            picks = {a["agent_id"]: sorted((x["time"]["step"], x["parcel_id"]) for x in a["actions"] if x["type"] == "pick") for a in ags}
            drops = {a["agent_id"]: sorted(x["time"]["step"] for x in a["actions"] if x["type"] == "drop") for a in ags}
            picked_at = {pid: t for pl in picks.values() for t, pid in pl}
            starts = {aid: c[0] for aid, c in cells.items()}
            times = sorted({dec["time"]["step"] for g in ags for dec in g.get("slm_decisions", [])})
            for t in times:
                open_ids = [i for i in ppos if picked_at.get(i, 10 ** 9) > t]
                if len(open_ids) < 2:
                    continue
                for pid, traj in cells.items():
                    carrying = any(pt <= t < next((dt for dt in drops[pid] if dt >= pt), 10 ** 9) for pt, _ in picks[pid])
                    if carrying:
                        continue
                    nxt = next(((pt, pp) for pt, pp in picks[pid] if pt > t), None)
                    if nxt is None or nxt[1] not in open_ids:
                        continue
                    pos = traj[min(t, len(traj)) - 1]
                    others = [cells[o][min(t, len(cells[o])) - 1] for o in cells if o != pid]
                    dp = np.array([man(pos, ppos[i]) for i in open_ids], float)
                    dd = np.array([man(ppos[i], centre) for i in open_ids], float)
                    comp = np.array([max(0.0, d0 - min(man(o, ppos[i]) for o in others)) for d0, i in zip(dp, open_ids)], float)
                    dst = np.array([man(starts[pid], ppos[i]) for i in open_ids], float)
                    rk = np.argsort(np.argsort(dp)).astype(float)
                    data.append((mission, np.stack([dp, dd, comp, dst, rk], axis=1), open_ids.index(nxt[1]), int(pid.split('_')[-1])))
    return data


def fit(X, y, l2=1e-3, iters=400, lr=0.05):
    k = X[0].shape[1]
    w = np.zeros(k)
    mu = np.mean(np.concatenate(X), axis=0)
    sd = np.std(np.concatenate(X), axis=0) + 1e-9
    Z = [(x - mu) / sd for x in X]
    for _ in range(iters):
        g = -l2 * w
        for z, t in zip(Z, y):
            s = z @ w
            p = np.exp(s - s.max())
            p /= p.sum()
            g += z[t] - p @ z
        w += lr * g / len(Z)
    return w, mu, sd


def score(w, mu, sd, X, y):
    ll = t1 = t3 = 0.0
    for x, t in zip(X, y):
        s = ((x - mu) / sd) @ w
        p = np.exp(s - s.max())
        p /= p.sum()
        ll += np.log(max(p[t], 1e-12))
        order = np.argsort(-s)
        t1 += order[0] == t
        t3 += t in order[:3]
    n = len(X)
    return ll / n, t1 / n, t3 / n


MIX = ["rational", "selfish", "altruistic", "reckless"]


def by_personality(data):
    """Pooled fit against one fit per personality (peers take the personalities in turn by drone index, as the mixed-fleet arms assign them); the held-out log-likelihood of each."""
    cols = [0, 1, 2]
    train = [d for d in data if d[0] % 2 == 1]
    test = [d for d in data if d[0] % 2 == 0]
    w, mu, sd = fit([d[1][:, cols] for d in train], [d[2] for d in train])
    pooled_ll, pooled_t1, pooled_t3 = score(w, mu, sd, [d[1][:, cols] for d in test], [d[2] for d in test])
    print(f"\n  pooled fit (pickup, delivery, competition): log-lik {pooled_ll:.3f}, top-3 {pooled_t3:.1%}; weights (pickup, delivery, competition) {np.round(w, 2)}")
    tot_ll = tot_n = tot3 = 0.0
    for k, name in enumerate(MIX):
        tr = [d for d in train if d[3] % 4 == k]
        te = [d for d in test if d[3] % 4 == k]
        wk, muk, sdk = fit([d[1][:, cols] for d in tr], [d[2] for d in tr])
        ll, t1, t3 = score(wk, muk, sdk, [d[1][:, cols] for d in te], [d[2] for d in te])
        llp, _, t3p = score(w, mu, sd, [d[1][:, cols] for d in te], [d[2] for d in te])
        tot_ll += ll * len(te)
        tot3 += t3 * len(te)
        tot_n += len(te)
        print(f"  {name:11s} n={len(te):5d} own fit: log-lik {ll:6.3f} top-3 {t3:5.1%} | pooled fit on them: {llp:6.3f} / {t3p:5.1%} | own weights {np.round(wk, 2)}")
    print(f"  all four, own fits: log-lik {tot_ll / tot_n:.3f}, top-3 {tot3 / tot_n:.1%}  (pooled {pooled_ll:.3f}, {pooled_t3:.1%}); a gain would mean personality shapes the choice")


def main(root, arm):
    data = collect(root, arm)
    train = [d for d in data if d[0] % 2 == 1]
    test = [d for d in data if d[0] % 2 == 0]
    chance = np.mean([1.0 / len(d[1]) for d in test])
    chance_ll = np.mean([-np.log(len(d[1])) for d in test])
    print(f"{len(train)} training and {len(test)} test choices; chance: top-1 {chance:.1%}, log-likelihood {chance_ll:.2f}")
    sets = [("M0 pickup distance", [0]), ("M1 + delivery leg (whole trip)", [0, 1]), ("M2 + competition", [0, 1, 2]), ("M3 + distance from the peer's start", [0, 1, 2, 3]),
            ("M4 + rank by pickup distance", [0, 1, 2, 3, 4])]
    for name, cols in sets:
        w, mu, sd = fit([d[1][:, cols] for d in train], [d[2] for d in train])
        ll, t1, t3 = score(w, mu, sd, [d[1][:, cols] for d in test], [d[2] for d in test])
        print(f"  {name:40s} log-lik {ll:6.2f}  top-1 {t1:5.1%}  top-3 {t3:5.1%}   weights (standardised) {np.round(w, 2)}")


def collect_own(root, arm):
    """Each drone's choice at its own decisions, among the tasks it offered itself (its beliefs): the parcel it chose is proposals[policy]. Features as in collect()."""
    data = []
    mission = 0
    for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
        exp = json.load(open(ep))
        st = exp["station"]
        centre = (st["col"] + st["w"] // 2, st["row"] + st["h"] // 2)
        for d in sorted(glob.glob(ep.replace("experiment.json", f"{arm}_instance_*"))):
            mission += 1
            fleet = json.load(open(f"{d}/fleet.json"))
            ppos = {p["parcel_id"]: tuple(p["spawn_cell"]) for p in fleet["parcels"]}
            wbycell = {(p["col"], p["row"]): float(p.get("weight", 1.0)) for p in exp["parcels"]}
            wt = {i: wbycell.get(c, 1.0) for i, c in ppos.items()}
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            cells = {a["agent_id"]: [tuple(t["cell"]) for t in a["trajectory"]] for a in ags}
            for g in ags:
                me = g["agent_id"]
                for dec in g.get("slm_decisions", []):
                    pr = dec.get("proposals") or {}
                    m = pr.get("measured")
                    chosen = pr.get(dec.get("policy"))
                    if not m or not isinstance(chosen, str):
                        continue
                    offered = [x[0] for x in m["tasks"] if x[0] in ppos]
                    if len(offered) < 2 or chosen not in offered:
                        continue
                    t = dec["time"]["step"]
                    pos = cells[me][min(t, len(cells[me])) - 1]
                    others = [cells[o][min(t, len(cells[o])) - 1] for o in cells if o != me]
                    dp = np.array([man(pos, ppos[i]) for i in offered], float)
                    dd = np.array([man(ppos[i], centre) for i in offered], float)
                    comp = np.array([max(0.0, d0 - min(man(o, ppos[i]) for o in others)) for d0, i in zip(dp, offered)], float)
                    dst = np.array([man(cells[me][0], ppos[i]) for i in offered], float)
                    rk = np.argsort(np.argsort(dp)).astype(float)
                    wv = np.array([wt[i] for i in offered], float)
                    data.append((mission, np.stack([dp, dd, comp, dst, rk, wv, dp * wv], axis=1), offered.index(chosen), int(me.split("_")[-1])))
    return data


def main_own(root, arm):
    data = collect_own(root, arm)
    train = [d for d in data if d[0] % 2 == 1]
    test = [d for d in data if d[0] % 2 == 0]
    print(f"{len(train)} training and {len(test)} test decisions, each among the drone's OWN offered tasks; chance top-1 {np.mean([1.0 / len(d[1]) for d in test]):.1%}")
    for name, cols in [("M0 pickup distance", [0]), ("M1 + delivery leg", [0, 1]), ("M2 + competition", [0, 1, 2]), ("M5 + parcel weight", [0, 1, 2, 5]),
                       ("M6 + weight x pickup distance (SPPDEC's own cost form)", [0, 1, 2, 5, 6]), ("M4 + rank by pickup distance", [0, 1, 2, 3, 4])]:
        w, mu, sd = fit([d[1][:, cols] for d in train], [d[2] for d in train])
        ll, t1, t3 = score(w, mu, sd, [d[1][:, cols] for d in test], [d[2] for d in test])
        print(f"  {name:32s} log-lik {ll:6.2f}  top-1 {t1:5.1%}  top-3 {t3:5.1%}   weights {np.round(w, 2)}")
    by_personality(data)


if __name__ == "__main__":
    if not (len(sys.argv) > 3 and sys.argv[3] == "own"):
        main(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3] == "personality":
        by_personality(collect(sys.argv[1], sys.argv[2]))
    if len(sys.argv) > 3 and sys.argv[3] == "own":
        main_own(sys.argv[1], sys.argv[2])
