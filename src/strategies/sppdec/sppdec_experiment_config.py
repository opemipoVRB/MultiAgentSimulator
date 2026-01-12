# src/strategies/sppdec/sppdec_experiment_config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SPPDECConfig:
    baseline_payload: float
    energy_penalty_alpha: float


CONFIG_REGISTRY = {
    "control": SPPDECConfig(0.0, 0.0),
    "baseline_only": SPPDECConfig(0.15, 0.0),
    "fairness_only": SPPDECConfig(0.0, 0.02),
    "combined": SPPDECConfig(0.15, 0.02),
}


def get_sppdec_config(variant: str) -> SPPDECConfig:
    if variant not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown SPPDEC variant: {variant}")
    return CONFIG_REGISTRY[variant]
