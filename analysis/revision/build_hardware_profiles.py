# analysis/revision/build_hardware_profiles.py
"""
Build src/hardware_profiles.json from the speed measurements (.ignore/sandbox/hw/measure.py).

Per model and machine setting: prefill speed (tokens/s, prompt read cold), decode speed (tokens/s), cache ratio (time to
read the prompt with its fixed prefix cached, over the cold time). Profiles: the three measured settings of this machine
and a Raspberry Pi 5 estimate, scaled from the 4-thread CPU measurement by one stated factor (published Pi 5 decode speed
for a 1.5B model, about 5-15 tokens/s, midpoint 10, over the measured decode speed of the 1.7B model on 4 threads here).

Usage: python -m analysis.revision.build_hardware_profiles .ignore/sandbox/hw/results_*.json
"""
import glob
import json
import sys
from pathlib import Path

SETTINGS = {"gpu": ("workstation_gpu", "This machine, language model on the GPU (16 GB)."),
            "cpu_all": ("workstation_cpu", "This machine, language model on the CPU only, all threads."),
            "cpu_4": ("cpu_4core", "This machine, CPU only, limited to 4 threads (an upper bound for a 4-core board).")}
# Published llama.cpp figures for a Jetson Orin Nano 8 GB (Super), 25 W mode, Q4_K_M, 4,096 context, six threads, five-run llama-bench
# averages (pp512 = reading, tg128 = writing), from https://turingpi.com/jetson-orin-nano-8gb-llm-benchmark/ :
JETSON_ORIN_NANO_PUBLISHED = {"qwen3:1.7b": (1408.3, 39.42), "qwen3:4b": (609.7, 19.53), "qwen3:8b": (355.5, 12.64)}
PI_PUBLISHED_DECODE_1_5B = 10.0   # tokens/s, midpoint of the 5-15 range reported for 1.5B models on a Raspberry Pi 5 (search summaries)


def entry(m):
    cold, warm = m["cold"], m["prefix_cached"]
    return {"prefill_tps": round(cold["prompt_tokens"] / max(cold["prompt_s"], 1e-3), 1),
            "decode_tps": round(cold["gen_tokens"] / max(cold["gen_s"], 1e-3), 1),
            "cache_ratio": round(min(1.0, warm["prompt_s"] / max(cold["prompt_s"], 1e-3)), 3),
            "prompt_tokens": cold["prompt_tokens"]}


def main(paths):
    results = {}
    for p in paths:
        results.update(json.load(open(p)))
    profiles = {}
    for key, m in results.items():
        if "error" in m or "cold" not in m or "error" in m["cold"]:
            continue
        model, setting = key.split("|")
        name, desc = SETTINGS[setting]
        profiles.setdefault(name, {"description": desc, "models": {}})["models"][model] = entry(m)
    cpu4 = profiles.get("cpu_4core", {}).get("models", {})
    ref = cpu4.get("qwen3:1.7b")
    if ref:
        factor = PI_PUBLISHED_DECODE_1_5B / ref["decode_tps"]
        profiles["pi5_estimate"] = {
            "description": ("ESTIMATE, not measured: the 4-thread CPU measurements scaled by %.2f (published Raspberry Pi 5 "
                            "decode speed for a 1.5B model, midpoint 10 tokens/s, over %.1f measured here for qwen3:1.7b on "
                            "4 threads); the same factor is applied to reading the prompt." % (factor, ref["decode_tps"])),
            "models": {mod: {"prefill_tps": round(e["prefill_tps"] * factor, 1), "decode_tps": round(e["decode_tps"] * factor, 1),
                             "cache_ratio": e["cache_ratio"], "prompt_tokens": e["prompt_tokens"]} for mod, e in cpu4.items()}}
    gpu = profiles.get("workstation_gpu", {}).get("models", {})
    if gpu:
        # A drone with its own Jetson Orin Nano 8 GB: published speeds for the models the benchmark covers; the 0.6B and the
        # vision-language 8B (not in it) are scaled from the nearest measured model by their speed ratio on this machine's GPU.
        # The cache ratio is the measured GPU one (the cache works the same on any GPU).
        models = {}
        for model, (prefill, decode) in JETSON_ORIN_NANO_PUBLISHED.items():
            models[model] = {"prefill_tps": prefill, "decode_tps": decode, "cache_ratio": gpu[model]["cache_ratio"],
                             "prompt_tokens": gpu[model]["prompt_tokens"], "source": "published"}
        for model, ref in (("qwen3:0.6b", "qwen3:1.7b"), ("qwen3-vl:8b-instruct", "qwen3:8b")):
            if model in gpu and ref in models:
                models[model] = {"prefill_tps": round(models[ref]["prefill_tps"] * gpu[model]["prefill_tps"] / gpu[ref]["prefill_tps"], 1),
                                 "decode_tps": round(models[ref]["decode_tps"] * gpu[model]["decode_tps"] / gpu[ref]["decode_tps"], 1),
                                 "cache_ratio": gpu[model]["cache_ratio"], "prompt_tokens": gpu[model]["prompt_tokens"],
                                 "source": f"ESTIMATE: {ref} published speed times the speed ratio on this machine's GPU"}
        profiles["jetson_orin_nano_8gb"] = {
            "description": ("A drone's own Jetson Orin Nano 8 GB (dedicated, no queue behind other drones). Published llama.cpp speeds "
                            "(Q4_K_M, 25 W mode) for qwen3 1.7B, 4B and 8B; the other models scaled as stated per model. "
                            "https://turingpi.com/jetson-orin-nano-8gb-llm-benchmark/"),
            "models": models}
    out = Path(__file__).resolve().parents[2] / "src" / "hardware_profiles.json"
    out.write_text(json.dumps(profiles, indent=1))
    for name, p in profiles.items():
        print(name)
        for mod, e in p["models"].items():
            print(f"  {mod:22s} prefill {e['prefill_tps']:8.1f} tok/s  decode {e['decode_tps']:7.1f} tok/s  cache ratio {e['cache_ratio']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:] or glob.glob(".ignore/sandbox/hw/results_*.json"))
