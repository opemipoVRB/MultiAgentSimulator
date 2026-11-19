# analysis/revision/names.py
"""`pretty(text)` rewrites the short labels used inside the analysis code ("S: CBBA shortcut + observation",
"cbba_bid_rounds", ...) into the plain names of arm_names.py, so printed tables read clearly."""
import json
from analysis.revision.revision_stats import LABELS
from analysis.revision.arm_names import NAMES_PATH

MODEL_WORDS = {"greedy": "Greedy (nearest task first)", "cbba": "CBBA, one task (bids guessed)", "hungarian": "Hungarian (optimal assignment)",
               "cbba_bid_rounds": "CBBA, 3 tasks (bids guessed)", "bid_rounds": "CBBA, 3 tasks (bids guessed)"}


def _map():
    try:
        names = json.loads(NAMES_PATH.read_text())
    except Exception:
        return {}
    out = {LABELS[a]: n for a, n in names.items() if a in LABELS}
    return dict(sorted(out.items(), key=lambda kv: -len(kv[0])))


_PAIRS = None


def pretty(text: str) -> str:
    global _PAIRS
    if _PAIRS is None:
        _PAIRS = list(_map().items())
    for old, new in _PAIRS:
        text = text.replace(old, new)
    for old, new in MODEL_WORDS.items():
        text = text.replace(f"{old} /", f"{new} /").replace(f"/ {old}", f"/ {new}")
    return text
