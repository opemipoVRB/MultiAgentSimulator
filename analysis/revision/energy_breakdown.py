# analysis/revision/energy_breakdown.py
"""Energy of each strategy by source, per mission, in kJ (1 energy unit = 3.6 kJ, the 100 Wh battery anchor of simulated_hardware.md):
motion (flight, pick, drop), hover, radio (messages x the send and receive charge), and onboard computation (ESTIMATED: CPU seconds of the drones' logic, or of the
command-center planner, x a slowdown of 5 for an embedded board x 6.1 W, the Raspberry Pi 5 loaded power 8.8 W minus idle 2.7 W). Computation is not charged in the
simulator; it is reported beside the simulated energy, not added to it.
Usage: python -m analysis.revision.energy_breakdown <batch dir> <cpu log .jsonl> <out csv>"""
import csv, glob, json, sys
from collections import defaultdict

UNIT_KJ, TX, RX, SLOWDOWN, WATTS = 3.6, 8.3e-7, 1.9e-7, 5.0, 6.1

def cpu_per_drone(path):
    """Mean CPU seconds per drone per mission for each strategy name, and the command-center total for the centralised one."""
    d, cc = defaultdict(list), defaultdict(list)
    for line in open(path):
        r = json.loads(line); name = r["strategy"].rsplit("_instance_", 1)[0]
        (cc if r["agent"] == "command_center" else d)[name].append(r["think_cpu_s"])
    return d, cc

def main(batch, cpu, out):
    per_drone, per_cc = cpu_per_drone(cpu)
    rows = []
    for ep in sorted(glob.glob(f"{batch}/*/experiment_*/experiment.json")):
        e = json.load(open(ep)); base = ep.rsplit("/", 1)[0]; cfg = ep.split("/")[-3]
        for r in e["results"]:
            m = r["metrics"]; fam = r["family"]
            ags = json.load(open(f"{base}/{r['artifacts']['agents']}")); ags = list(ags.values()) if isinstance(ags, dict) else ags
            n = m["n_drones"]; total = m["total_energy_consumed"]; hover = sum(a.get("hover_energy", 0.0) for a in ags)
            radio = m["messages_sent"] * (TX + RX)                      # one send and one receive per counted message (ideal radio, no loss)
            name = r["family"].rsplit("_", 1)[0] if r["family"] not in per_drone and r["family"] not in per_cc else r["family"]
            key = next((k for k in list(per_drone) + list(per_cc) if r["family"].startswith(k)), None)
            cpu_s = (sum(per_drone[key]) / len(per_drone[key]) * n) if key in per_drone else (sum(per_cc[key]) / len(per_cc[key]) if key in per_cc else None)
            rows.append(dict(config=cfg, arm=fam, drones=n, delivered=m["tasks_completed"], parcels=m["n_parcels"], messages=m["messages_sent"],
                             motion_kJ=round((total - hover - radio) * UNIT_KJ, 2), hover_kJ=round(hover * UNIT_KJ, 2), radio_kJ=round(radio * UNIT_KJ, 6),
                             cpu_s=None if cpu_s is None else round(cpu_s, 2), computation_kJ_est=None if cpu_s is None else round(cpu_s * SLOWDOWN * WATTS / 1000, 3)))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")

if __name__ == "__main__":
    main(*sys.argv[1:4])
