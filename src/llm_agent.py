# src/llm_agent.py
from __future__ import annotations
import time
import json
import hashlib
from typing import Any, Dict, Optional

from llm_planner import PlannerClient

class LLMAgent:
    """
    Per-agent LLM wrapper.
    - wraps PlannerClient (which already supports LLM/non-LLM)
    - does simple caching and rate-limiting per-agent
    """

    def __init__(self, agent_id: str, use_llm: bool = False, min_interval: float = 0.5):
        self.agent_id = agent_id
        self._planner = PlannerClient(use_llm=use_llm)
        self._min_interval = float(min_interval)
        self._last_call_ts = 0.0
        self._cache = {}  # snapshot_hash -> plan

    def _snapshot_hash(self, snapshot: Dict[str, Any]) -> str:
        # deterministic lightweight hash for caching
        s = json.dumps(snapshot, sort_keys=True)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def propose(self, snapshot: Dict[str, Any], force_refresh: bool = False) -> Dict[str, Any]:
        """
        Request a plan (proposal) for this agent.
        Respects rate limit and caches results for identical snapshots.
        """
        now = time.time()
        if (now - self._last_call_ts) < self._min_interval and not force_refresh:
            # if called too quickly, return cached last plan if any
            if hasattr(self, "_last_plan") and self._last_plan:
                return self._last_plan

        h = self._snapshot_hash(snapshot)
        if not force_refresh and h in self._cache:
            self._last_call_ts = now
            self._last_plan = self._cache[h]
            return self._last_plan

        plan = self._planner.request_plan(snapshot, force_refresh=force_refresh)
        self._cache[h] = plan
        self._last_call_ts = now
        self._last_plan = plan
        return plan

    def clear_cache(self):
        self._cache.clear()
