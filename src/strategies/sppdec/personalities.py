SPPDEC_PERSONALITIES = {
    "rational": {
        "baseline_payload": 0.15,
        "energy_penalty_alpha": 0.02,
        "label": "Balanced, team-aware"
    },
    "selfish": {
        "baseline_payload": 0.15,
        "energy_penalty_alpha": 0.0,
        "label": "Optimizes for self only"
    },
    "altruistic": {
        "baseline_payload": 0.0,
        "energy_penalty_alpha": 0.02,
        "label": "Avoids overworked peers strongly"
    },
    "reckless": {
        "baseline_payload": 0.0,
        "energy_penalty_alpha": 0.0,
        "label": "No self modeling, no fairness"
    }
}
