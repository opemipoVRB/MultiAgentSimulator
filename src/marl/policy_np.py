# src/marl/policy_np.py
"""
The learned task-selection policy in plain numpy: parameters, initialisation, forward pass, save and load.

Execution needs only this file (no torch). The trainer holds the same network in torch (src/marl/policy_torch.py) and writes its
parameters in the layout defined here; a test checks that both give the same outputs.

Architecture (width H): the drone embeds itself, its peers and the tasks it can afford. The drone attends over the peers (a context
vector), then scores each task by a dot product between a query built from [drone, peer context] and the task's key (pointer
style), plus a direct linear term on the task features. The output is one logit per offered task; there is no wait action (the
drone always takes an offered task, or waits when none is offered). A value head on [drone, peer context, mean task embedding]
serves as the critic during training.
"""
from typing import Dict, Tuple

import numpy as np

from src.marl.features import PEER_DIM, SELF_DIM, TASK_DIM

H = 32

PARAM_SHAPES = {
    "Ws1": (SELF_DIM, H), "bs1": (H,),
    "Wt1": (TASK_DIM, H), "bt1": (H,), "Wt2": (H, H), "bt2": (H,),
    "Wp1": (PEER_DIM, H), "bp1": (H,),
    "Wpq": (H, H), "Wpk": (H, H),
    "Wq": (2 * H, H), "Wk": (H, H), "wd": (TASK_DIM,),
    "Wv1": (3 * H, H), "bv1": (H,), "wv2": (H,), "bv2": (1,),
}


def init_params(seed: int = 0) -> Dict[str, np.ndarray]:
    """Small random weights; the direct term starts at zero, so the untrained policy is close to a uniform random choice."""
    rng = np.random.default_rng(seed)
    params = {}
    for name, shape in PARAM_SHAPES.items():
        if name.startswith("b") or name in ("wd",):
            params[name] = np.zeros(shape, dtype=np.float32)
        else:
            fan_in = shape[0] if len(shape) > 1 else shape[0]
            params[name] = (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)
    params["wv2"] = (params["wv2"] * 0.1).astype(np.float32)
    return params


def forward(params: Dict[str, np.ndarray], x_self: np.ndarray, tasks: np.ndarray, peers: np.ndarray
            ) -> Tuple[np.ndarray, float]:
    """Logits over the offered tasks (shape (n,)) and the value of the drone's situation."""
    e_s = np.tanh(x_self @ params["Ws1"] + params["bs1"])
    if len(peers):
        e_p = np.maximum(peers @ params["Wp1"] + params["bp1"], 0.0)
        scores = (e_p @ params["Wpk"]) @ (e_s @ params["Wpq"]) / np.sqrt(H)
        scores = scores - scores.max()
        att = np.exp(scores)
        att = att / att.sum()
        context = att @ e_p
    else:
        context = np.zeros(H, dtype=np.float32)
    ctx = np.concatenate([e_s, context])
    if len(tasks):
        e_t = np.maximum(np.maximum(tasks @ params["Wt1"] + params["bt1"], 0.0) @ params["Wt2"] + params["bt2"], 0.0)
        q = ctx @ params["Wq"]
        logits = (e_t @ params["Wk"]) @ q / np.sqrt(H) + tasks @ params["wd"]
        mean_t = e_t.mean(axis=0)
    else:
        logits = np.zeros(0, dtype=np.float32)
        mean_t = np.zeros(H, dtype=np.float32)
    v_in = np.concatenate([ctx, mean_t])
    value = float(np.maximum(v_in @ params["Wv1"] + params["bv1"], 0.0) @ params["wv2"] + params["bv2"][0])
    return logits.astype(np.float32), value


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    return z - np.log(np.exp(z).sum())


def save(path: str, params: Dict[str, np.ndarray]) -> None:
    np.savez(path, **params)


def load(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as f:
        params = {k: f[k].astype(np.float32) for k in f.files}
    missing = set(PARAM_SHAPES) - set(params)
    if missing:
        raise ValueError(f"policy file {path} lacks parameters {sorted(missing)}")
    for name, shape in PARAM_SHAPES.items():
        if params[name].shape != shape:
            raise ValueError(f"policy file {path}: {name} has shape {params[name].shape}, expected {shape}")
    return params
