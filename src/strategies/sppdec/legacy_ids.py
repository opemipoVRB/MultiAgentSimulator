# src/strategies/sppdec/legacy_ids.py
"""
Identifiers were renamed to say what they mean. Runs exported before the rename carry the old ids;
`modernise` turns an old id into the current one so analysis and replay read both.

  cbba_head*                     -> cbba_assumed_bids*      (CBBA run in the drone, peer bids assumed)
  cbba_in_head                   -> cbba_with_assumed_bids
  proto1 / proto3 / protoauto    -> bid_rounds_bundle1 / bid_rounds_bundle3 / bid_rounds_fair_share
  *_shared*                      -> *_symmetric*            (drone and peers scored by the same estimator)
  *_obs                          -> *_tracking              (peers tracked every step)
  *_mem                          -> *_memory                (remembers who took what)
  hung / clf / slmimit           -> hungarian / classifier / slm_imitation
  *_hetero*                      -> *_unequal_batteries*
  cbba_impl "shortcut"/"protocol" -> "best_pair_first" / "bid_rounds"
"""
import re

RULES = [
    (r"cbba_in_head", "cbba_with_assumed_bids"),
    (r"cbba_head", "cbba_assumed_bids"),
    (r"protoauto", "bid_rounds_fair_share"),
    (r"proto(\d)", r"bid_rounds_bundle\1"),
    (r"_shared(?![a-z])", "_symmetric"),
    (r"_obs(?![a-z])", "_tracking"),
    (r"_mem(?![a-z])", "_memory"),
    (r"(?<![a-z])hung(?=_)", "hungarian"),
    (r"(?<![a-z])clf(?=_)", "classifier"),
    (r"slmimit", "slm_imitation"),
    (r"_hetero_assume_same", "_unequal_batteries_assumed_equal"),
    (r"_hetero_known", "_unequal_batteries_known"),
    (r"_hetero(?![a-z])", "_unequal_batteries"),
]


def modernise(identifier: str) -> str:
    """Current spelling of a strategy / arm id (unchanged if it is already current)."""
    for pat, rep in RULES:
        identifier = re.sub(pat, rep, identifier)
    return identifier
