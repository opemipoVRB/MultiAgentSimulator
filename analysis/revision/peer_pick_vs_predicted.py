# analysis/revision/peer_pick_vs_predicted.py
"""
What a peer picked against what SPPDEC predicted, to find the gap in the reasoning. At each logged decision of the shadow arm, for every peer that was not carrying:
  hungarian / cbba   the parcel the drone's own reasoning model assigns to the peer (`measured.assign`)
  nearest            the open parcel nearest to the peer's last sensed position (the rule of the peer model)
against the truth: the first parcel the peer picks after the decision.
Reports the hit rate of each, and for the misses where the gap is: (a) the true parcel was not among the tasks the drone offered itself (information or feasibility gap),
(b) it was, but the model gave it to another agent (which: self, another peer, nobody), by whether the peer is sensed now, by how long ago, and by the true parcel's
distance rank from the peer.
Usage: python -m analysis.revision.peer_pick_vs_predicted <batch dir> <arm>
"""
import collections
import glob
import json
import sys

import numpy as np


def man(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def main(root, arm):
    hit = collections.defaultdict(lambda: collections.Counter())      # model -> {"hit","miss"} by group
    gap = collections.Counter()
    gap_by = collections.defaultdict(collections.Counter)
    rank_true = []
    rows = 0
    for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
        cfg = ep.split("/")[-3]
        for d in glob.glob(ep.replace("experiment.json", f"{arm}_instance_*")):
            fleet = json.load(open(f"{d}/fleet.json"))
            ppos = {p["parcel_id"]: tuple(p["spawn_cell"]) for p in fleet["parcels"]}
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            picks = {a["agent_id"]: sorted((x["time"]["step"], x["parcel_id"]) for x in a["actions"] if x["type"] == "pick") for a in ags}
            drops = {a["agent_id"]: sorted(x["time"]["step"] for x in a["actions"] if x["type"] == "drop") for a in ags}
            picked_at = {pid: t for pl in picks.values() for t, pid in pl}
            for g in ags:
                me = g["agent_id"]
                for dec in g.get("slm_decisions", []):
                    m = (dec.get("proposals") or {}).get("measured")
                    if not m or not m.get("assign"):
                        continue
                    t = dec["time"]["step"]
                    offered = {x[0] for x in m["tasks"]}
                    for pid, last, _old, since, vis in m["peers"]:
                        if pid not in picks:
                            continue
                        carrying = any(pt <= t < next((dt for dt in drops[pid] if dt >= pt), 10 ** 9) for pt, _ in picks[pid])
                        if carrying:
                            continue
                        nxt = next(((pt, pp) for pt, pp in picks[pid] if pt > t), None)
                        if nxt is None:
                            continue
                        true = nxt[1]
                        open_now = [i for i, pa in picked_at.items() if pa > t] + [i for i in ppos if i not in picked_at]
                        order = sorted(set(open_now), key=lambda i: (man(tuple(last), ppos[i]), i))
                        r = order.index(true) + 1 if true in order else None
                        rows += 1
                        if r:
                            rank_true.append(r)
                        grp = "sensed now" if vis else ("unseen <=100 steps" if since <= 100 else "unseen >100 steps")
                        preds = {"hungarian": m["assign"]["hungarian"].get(pid), "cbba": m["assign"]["cbba"].get(pid), "nearest": order[0] if order else None}
                        for name, pr in preds.items():
                            hit[name][(grp, "hit" if pr == true else "miss")] += 1
                            hit[name][("all", "hit" if pr == true else "miss")] += 1
                        # where the gap is, for the Hungarian model
                        pr = preds["hungarian"]
                        if pr != true:
                            if true not in offered:
                                kind = "true parcel not among the tasks I offered myself"
                            else:
                                owner = next((a for a, p_ in m["assign"]["hungarian"].items() if p_ == true), None)
                                kind = ("true parcel assigned to nobody" if owner is None else
                                        "true parcel assigned to another peer" if owner != pid else "?")
                                if pr is None:
                                    kind += " (and the peer itself got nothing)"
                            gap[kind] += 1
                            gap_by[grp][kind] += 1
    print(f"{rows} (decision, empty peer) pairs with a known next pick")
    for name in ("hungarian", "cbba", "nearest"):
        line = []
        for grp in ("all", "sensed now", "unseen <=100 steps", "unseen >100 steps"):
            h, ms = hit[name][(grp, "hit")], hit[name][(grp, "miss")]
            line.append(f"{grp}: {h / max(1, h + ms):.0%} (n={h + ms})")
        print(f"  {name:10s} exact hit rate | " + " | ".join(line))
    print(f"  rank of the true next parcel by distance from the peer's last sensed position: median {np.median(rank_true):.0f}; rank 1 {np.mean(np.array(rank_true) == 1):.0%}, within 3 {np.mean(np.array(rank_true) <= 3):.0%}")
    tot = sum(gap.values())
    print(f"\nMisses of the Hungarian model (n={tot}): where the gap is")
    for k, v in gap.most_common():
        print(f"   {v / tot:5.0%}  {k}")
    for grp in ("sensed now", "unseen <=100 steps", "unseen >100 steps"):
        tt = sum(gap_by[grp].values())
        if tt:
            print(f"   {grp}: " + "; ".join(f"{v / tt:.0%} {k}" for k, v in gap_by[grp].most_common(3)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
