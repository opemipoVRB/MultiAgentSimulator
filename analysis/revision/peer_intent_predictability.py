# analysis/revision/peer_intent_predictability.py
"""
How predictable is what a working peer does next, from things a drone can observe? From the logged missions of SPPDEC's rule arms:
(a) the cell where a peer drops a parcel against the station cell nearest to the pickup (the shared drop-cell rule, ignoring occupancy);
(b) the parcel a peer takes next after a delivery against the open parcels ranked by distance from its drop cell;
(c) the parcel a drone sets out for against the open parcels ranked by distance from its position at the decision.
Usage: python -m analysis.revision.peer_intent_predictability <batch dir> [...]
"""
import collections
import glob
import json
import re
import statistics as st
import sys

from src.marl.global_state import MissionLog


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def main(roots):
    drop_exact, drop_err = collections.defaultdict(list), collections.defaultdict(list)
    rank_after, rank_dec, region = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for root in roots:
        for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
            cfg = ep.split("/")[-3]
            exp = json.load(open(ep))
            stn = exp["station"]
            cells = [(c, r) for r in range(stn["row"], stn["row"] + stn["h"]) for c in range(stn["col"], stn["col"] + stn["w"])]
            center = (stn["col"] + stn["w"] // 2, stn["row"] + stn["h"] // 2)
            for res in exp["results"]:
                name = res.get("strategy_name") or res.get("name")
                agents_path = ep.replace("experiment.json", f"{name}/agents.json")
                log = MissionLog(agents_path, ep)
                ags = json.load(open(agents_path))
                ags = list(ags.values()) if isinstance(ags, dict) else ags
                for a in ags:
                    ev = sorted(a["actions"], key=lambda e: e["time"]["step"] if "time" in e else e["step"])
                    st_ = lambda e: e["time"]["step"] if "time" in e else e["step"]
                    picks = [e for e in ev if e["type"] == "pick"]
                    drops = [e for e in ev if e["type"] == "drop"]
                    for p, d in zip(picks, drops):                      # the k-th pick is delivered by the k-th drop
                        pred = min(cells, key=lambda c: manh(p["cell"], c))
                        drop_exact[cfg].append(tuple(d["cell"]) == pred)
                        drop_err[cfg].append(manh(d["cell"], pred))
                    for k, d in enumerate(drops):
                        if k + 1 >= len(picks):
                            continue
                        nxt, t = picks[k + 1], st_(d)
                        open_now = [(i, par) for i, par in enumerate(log.parcels) if log.picked_step[i] is None or log.picked_step[i] > t]
                        target = next((i for i, par in open_now if (par[0], par[1]) == tuple(nxt["cell"])), None)
                        if target is None:
                            continue
                        order = sorted(open_now, key=lambda x: (manh(d["cell"], (x[1][0], x[1][1])), x[0]))
                        rank = [i for i, _ in order].index(target) + 1
                        rank_after[cfg].append(rank)
                        order_c = sorted(open_now, key=lambda x: (manh(center, (x[1][0], x[1][1])), x[0]))
                        region[cfg].append([i for i, _ in order_c].index(target) + 1)
                    for dec in a.get("decision_actions", []):
                        t = dec["time"]["step"]
                        me = log._cell(a["agent_id"], t)
                        open_now = [(i, par) for i, par in enumerate(log.parcels) if log.picked_step[i] is None or log.picked_step[i] > t]
                        cell = tuple(dec["decided_cell"]) if dec.get("decided_cell") else None
                        target = next((i for i, par in open_now if cell and (par[0], par[1]) == cell), None)
                        if target is None:
                            continue
                        order = sorted(open_now, key=lambda x: (manh(me, (x[1][0], x[1][1])), x[0]))
                        rank_dec[cfg].append([i for i, _ in order].index(target) + 1)
    def rate(xs, k):
        return sum(x <= k for x in xs) / len(xs) if xs else float("nan")
    for cfg in sorted(drop_exact):
        print(f"\n== {cfg}")
        print(f"(a) drop cell = the station cell nearest to the pickup: exact {sum(drop_exact[cfg]) / len(drop_exact[cfg]):.2f} of {len(drop_exact[cfg])}; mean miss {st.mean(drop_err[cfg]):.2f} cells")
        r = rank_after[cfg]
        print(f"(b) next parcel after a delivery, by distance from the drop cell: nearest {rate(r, 1):.2f}, within 3 nearest {rate(r, 3):.2f}, within 5 {rate(r, 5):.2f}, median rank {st.median(r):.0f} of {len(r)} cases")
        rc = region[cfg]
        print(f"    same, ranked by distance from the station centre instead: nearest {rate(rc, 1):.2f}, within 3 {rate(rc, 3):.2f}")
        rd = rank_dec[cfg]
        print(f"(c) parcel set out for, by distance from the drone at the decision: nearest {rate(rd, 1):.2f}, within 3 nearest {rate(rd, 3):.2f}, median rank {st.median(rd):.0f} of {len(rd)} cases")


if __name__ == "__main__":
    main(sys.argv[1:])
