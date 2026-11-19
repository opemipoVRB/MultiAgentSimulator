SPPDEC_PERSONALITIES = {
    "rational": {
        "avoid_width": 1,
        "baseline_payload": 0.15,
        "energy_penalty_alpha": 0.02,
        "label": "Balanced, team-aware"
    },
    "selfish": {
        "avoid_width": 0,
        "baseline_payload": 0.15,
        "energy_penalty_alpha": 0.0,
        "label": "Optimizes for self only"
    },
    "altruistic": {
        "avoid_width": 3,
        "baseline_payload": 0.0,
        "energy_penalty_alpha": 0.02,
        "label": "Avoids overworked peers strongly"
    },
    "reckless": {
        "avoid_width": 0,
        "baseline_payload": 0.0,
        "energy_penalty_alpha": 0.0,
        "label": "No self modeling, no fairness"
    }
}
