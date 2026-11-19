# src/hardware.py
"""
Hardware profiles: how long a drone's onboard computer takes to decide, from measured speeds.

A decision that consults the language model takes: reading the prompt (prefill), then writing the answer (decode).
Measured per model and per machine: prefill speed in tokens/s, decode speed in tokens/s, the model load time, and how
much faster the prompt is read when its fixed part (the model descriptions and the evidence tables, identical on every
decision) is already in the model's cache. A drone keeps that cache warm after its first decision.

The profiles live in src/hardware_profiles.json, written by analysis/revision/build_hardware_profiles.py from the
measurements of this machine (GPU, all CPU threads, 4 CPU threads) and, for boards that are not at hand, scaled by a
stated factor from published benchmarks (labelled "estimate" in the file).
"""
import json
import math
from pathlib import Path
from typing import Dict, Optional

PROFILES_PATH = Path(__file__).with_name("hardware_profiles.json")
STEPS_PER_SECOND = 60.0


class HardwareProfile:
    def __init__(self, name: str, models: Dict[str, Dict[str, float]], description: str = ""):
        self.name = name
        self.models = models
        self.description = description

    def decision_seconds(self, model: str, prompt_tokens: int, gen_tokens: int, cached: bool = False) -> float:
        m = self.models.get(model)
        if m is None:
            raise KeyError(f"hardware profile {self.name!r} has no measurement for model {model!r}")
        prefill = prompt_tokens / m["prefill_tps"] * (m["cache_ratio"] if cached else 1.0)
        return prefill + gen_tokens / m["decode_tps"]

    def decision_steps(self, model: str, prompt_tokens: int, gen_tokens: int, cached: bool = False) -> int:
        return max(1, int(math.ceil(self.decision_seconds(model, prompt_tokens, gen_tokens, cached) * STEPS_PER_SECOND)))


def load_profile(name: str, path: Optional[Path] = None) -> HardwareProfile:
    data = json.loads(Path(path or PROFILES_PATH).read_text(encoding="utf-8"))
    p = data[name]
    return HardwareProfile(name, p["models"], p.get("description", ""))


def estimate_tokens(text: str) -> int:
    """Token count of a prompt when the model server did not report it (about 3.8 characters per token for this prompt)."""
    return int(round(len(text) / 3.8))
