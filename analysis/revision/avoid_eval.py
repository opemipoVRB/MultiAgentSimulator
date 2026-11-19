# analysis/revision/avoid_eval.py
"""
Avoiding the peers' best parcels (batch experiments/peer_model/avoid.sh). Per config and sensing: matched complete-case makespan against the same fleet without avoidance (ratio, 95% paired
interval), contested plans, plans whose target was gone and distance flown per drone, runs without every parcel delivered; and how good the prediction is: the share of the real next picks
of the peers that were inside the peer's predicted top three.
Usage: python -m analysis.revision.avoid_eval <batch dir>
"""
import collections
import glob
import json
import math
import statistics as st
import sys

from analysis.revision.marl_evaluation import load

CFGS = ("A4_P36", "A8_P24", "A12_P24")
PAIRS = [("homogeneous", "aw_{s}_a0", "avoid_{s}"), ("mixed personalities", "mix_{s}_a0b", "mixavoid_{s}")]


def per_run(root, arm):
    out = {}
    for ep in glob.glob(f"{root}/*/experiment_*/experiment.json"):
        cfg = ep.split("/")[-3]
        seed = json.load(open(ep)).get("seed")
        for d in glob.glob(ep.replace("experiment.json", f"sppdec_{arm}_instance_*")):
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            n = len(ags)
            oc = collections.Counter(str(o.get("outcome")) for a in ags for o in a.get("plan_outcomes", []))
            out[(cfg, seed)] = {"contested": oc["interrupt_contested"] / n, "gone": (oc["target_gone"] + oc["interrupt_target_gone"]) / n,
                                "distance": st.mean((a.get("distance_cells") or 0.0) for a in ags)}
    return out


def top3_hit(root, arm, key="top3"):
    hit = tot = 0
    for ep in glob.glob(f"{root}/*/experiment_*/experiment.json"):
        for d in glob.glob(ep.replace("experiment.json", f"sppdec_{arm}_instance_*")):
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            picks = {a["agent_id"]: sorted((x["time"]["step"], x["parcel_id"]) for x in a["actions"] if x["type"] == "pick") for a in ags}
            drops = {a["agent_id"]: sorted(x["time"]["step"] for x in a["actions"] if x["type"] == "drop") for a in ags}
            for g in ags:
                for dec in g.get("slm_decisions", []):
                    m = (dec.get("proposals") or {}).get("measured")
                    if not m or not m.get(key):
                        continue
                    t = dec["time"]["step"]
                    for pid, top in m[key].items():
                        if pid not in picks or any(pt <= t < next((dt for dt in drops[pid] if dt >= pt), 10 ** 9) for pt, _ in picks[pid]):
                            continue
                        nxt = next((pp for pt, pp in picks[pid] if pt > t), None)
                        if nxt is None:
                            continue
                        tot += 1
                        hit += nxt in top
    return hit / max(1, tot), tot


def main(root, variant="plain"):
    global PAIRS, CFGS
    if variant == "avoid3":
        PAIRS = [("top-3, open set", "aw_{s}_a0", "avoid3_{s}"), ("top-3, believed set", "aw_{s}_a0", "avoid3b_{s}")]
    if variant == "per_peer":
        CFGS = ("A4_P12", "A4_P24", "A8_P24")
        PAIRS = [("pooled fit", "mixsr_{s}", "mixsr_online_{s}"), ("fit per peer", "mixsr_{s}", "mixsr_pp_{s}")]
    if variant == "online":
        CFGS = ("A4_P12", "A4_P24", "A8_P24", "A12_P24")
        PAIRS = [("fixed learned cost", "aw_{s}_a0", "learned_{s}"), ("online learned cost", "aw_{s}_a0", "online_{s}")]
    if variant == "learned":
        CFGS = ("A4_P12", "A4_P24", "A8_P24", "A12_P24")
        PAIRS = [("learned peer cost", "aw_{s}_a0", "learned_{s}")]
    if variant == "avoid2":
        PAIRS = [("top-2, open set", "aw_{s}_a0", "avoid2_{s}"), ("top-2, believed set", "aw_{s}_a0", "avoid2b_{s}")]
    if variant == "trip":
        PAIRS = [("homogeneous", "rt_{s}", "avoidt_{s}"), ("mixed personalities", "mixrt_{s}", "mixavoidt_{s}")]
    top = {"trip": "avoidt", "avoid3": "avoid3", "avoid2": "avoid2"}.get(variant, "avoid")
    rec = load(root)
    for s in ("full", "noisy"):
        for label, base, new in PAIRS:
            b, n = base.format(s=s), new.format(s=s)
            rb, rn = per_run(root, b), per_run(root, n)
            for cfg in CFGS:
                keys = [k for k in rec if k[0] == cfg and b in rec[k] and n in rec[k]]
                done = [k for k in keys if rec[k][b]["delivered"] == rec[k][b]["parcels"] and rec[k][n]["delivered"] == rec[k][n]["parcels"]]
                if len(done) < 3:
                    print(f"{s:5s} {cfg:8s} {label:20s} only {len(done)} complete pairs of {len(keys)} (delivered per run: without {st.mean(rec[k][b]['delivered'] for k in keys):.1f}, with {st.mean(rec[k][n]['delivered'] for k in keys):.1f})")
                    continue
                d = [rec[k][n]["makespan"] / rec[k][b]["makespan"] for k in done]
                f = lambda r, key: st.mean(r[(cfg, rec_seed)][key] for rec_seed in [k[1] for k in done] if (cfg, rec_seed) in r)
                print(f"{s:5s} {cfg:8s} {label:20s} makespan {st.mean(rec[k][b]['makespan'] for k in done):6.0f} -> {st.mean(rec[k][n]['makespan'] for k in done):6.0f} "
                      f"x{st.mean(d):.3f}+-{2.04 * st.stdev(d) / math.sqrt(len(d)):.3f} | contested/drone {f(rb, 'contested'):.2f} -> {f(rn, 'contested'):.2f} | gone/drone {f(rb, 'gone'):.2f} -> {f(rn, 'gone'):.2f} "
                      f"| distance {f(rb, 'distance'):.0f} -> {f(rn, 'distance'):.0f} | n={len(done)}/{len(keys)}")
    if variant in ("learned", "online", "per_peer"):
        return
    if variant in ("avoid3", "avoid2"):
        for arm in (("avoid3", "avoid3b") if variant == "avoid3" else ("avoid2", "avoid2b")):
            for key in ("top3_open", "top3"):
                hf, tf = top3_hit(root, f"{arm}_full", key)
                hn, tn = top3_hit(root, f"{arm}_noisy", key)
                print(f"prediction ({arm}, ranking over {'the open set' if key == 'top3_open' else 'the believed set'}): a peer's real next pick inside its predicted top three in {hf:.0%} of {tf} (full) and {hn:.0%} of {tn} (noisy)")
        return
    h, t = top3_hit(root, f"{top}_noisy")
    h2, t2 = top3_hit(root, f"{top}_full")
    print(f"\nprediction: a peer's real next pick was inside its predicted top three in {h2:.0%} of {t2} cases (full sensing) and {h:.0%} of {t} (noisy sensing)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "plain")
