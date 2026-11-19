# src/marl/policy_torch.py
"""
The training copy of the policy in src/marl/policy_np.py: the same parameters in the same layout, a batched forward pass with padding masks,
and conversion to the numpy layout used at execution. A test (.ignore/sandbox/tools/test_marl_policy.py) checks that the two agree.
"""
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.marl.features import PEER_DIM, SELF_DIM, TASK_DIM
from src.marl.global_state import DRONE_DIM, PARCEL_DIM, SCALAR_DIM
from src.marl.policy_np import H, init_params


MASKED_LOGIT = -1e9


class PolicyNet(nn.Module):
    def __init__(self, params: Dict[str, np.ndarray] = None):
        super().__init__()
        params = params if params is not None else init_params(0)
        self.p = nn.ParameterDict({k: nn.Parameter(torch.tensor(v, dtype=torch.float32)) for k, v in params.items()})

    def forward(self, x_self, tasks, task_mask, peers, peer_mask):
        """
        x_self (B, S); tasks (B, N, T) with task_mask (B, N) True where a task exists; peers (B, M, P) with peer_mask (B, M).
        Returns logits (B, N) (-inf where there is no task) and values (B,).
        """
        P = self.p
        e_s = torch.tanh(x_self @ P["Ws1"] + P["bs1"])                                   # (B, H)
        if peers.shape[1] > 0:
            e_p = torch.relu(peers @ P["Wp1"] + P["bp1"])                                # (B, M, H)
            scores = ((e_p @ P["Wpk"]) * (e_s @ P["Wpq"]).unsqueeze(1)).sum(-1) / H ** 0.5   # (B, M)
            scores = scores.masked_fill(~peer_mask, -1e9)
            att = torch.softmax(scores, dim=1) * peer_mask.float()
            att = att / att.sum(dim=1, keepdim=True).clamp(min=1e-9)
            context = (att.unsqueeze(-1) * e_p).sum(1)                                   # (B, H); zero with no peers
        else:
            context = torch.zeros_like(e_s)
        ctx = torch.cat([e_s, context], dim=-1)                                          # (B, 2H)
        e_t = torch.relu(torch.relu(tasks @ P["Wt1"] + P["bt1"]) @ P["Wt2"] + P["bt2"])  # (B, N, H)
        q = ctx @ P["Wq"]                                                                # (B, H)
        logits = ((e_t @ P["Wk"]) * q.unsqueeze(1)).sum(-1) / H ** 0.5 + tasks @ P["wd"]  # (B, N)
        # A large finite value, not -inf: with -inf the entropy term (0 times -inf) gives NaN gradients.
        logits = logits.masked_fill(~task_mask, MASKED_LOGIT)
        counts = task_mask.float().sum(1, keepdim=True).clamp(min=1.0)
        mean_t = (e_t * task_mask.unsqueeze(-1).float()).sum(1) / counts                 # (B, H)
        v_in = torch.cat([ctx, mean_t], dim=-1)
        value = torch.relu(v_in @ P["Wv1"] + P["bv1"]) @ P["wv2"] + P["bv2"][0]
        return logits, value

    def numpy_params(self) -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in self.p.items()}


def collate(records: List[dict], device: str = "cpu"):
    """Pad a list of decision records (arrays from the strategy's trace) into batch tensors on `device`."""
    n_max = max(1, max(len(r["tasks"]) for r in records))
    m_max = max(0, max(len(r["peers"]) for r in records))
    B = len(records)
    x_self = np.zeros((B, SELF_DIM), dtype=np.float32)
    tasks = np.zeros((B, n_max, TASK_DIM), dtype=np.float32)
    task_mask = np.zeros((B, n_max), dtype=bool)
    peers = np.zeros((B, m_max, PEER_DIM), dtype=np.float32)
    peer_mask = np.zeros((B, m_max), dtype=bool)
    for i, r in enumerate(records):
        x_self[i] = r["x_self"]
        n, m = len(r["tasks"]), len(r["peers"])
        tasks[i, :n] = r["tasks"]
        task_mask[i, :n] = True
        if m:
            peers[i, :m] = r["peers"]
            peer_mask[i, :m] = True
    return (torch.from_numpy(x_self).to(device), torch.from_numpy(tasks).to(device), torch.from_numpy(task_mask).to(device),
            torch.from_numpy(peers).to(device), torch.from_numpy(peer_mask).to(device))



class CentralCritic(nn.Module):
    """
    Centralised value function for MAPPO: the deciding drone's own features plus the global state (every other drone's position and
    whether it carries, every open parcel, the step and the open share), pooled over the sets. Used during training only; it is not part of
    the deployed policy and is never saved in the policy file.
    """

    def __init__(self):
        super().__init__()
        self.self_enc = nn.Sequential(nn.Linear(SELF_DIM, H), nn.Tanh())
        self.drone_enc = nn.Sequential(nn.Linear(DRONE_DIM, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.parcel_enc = nn.Sequential(nn.Linear(PARCEL_DIM, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * H + SCALAR_DIM, H), nn.ReLU(), nn.Linear(H, 1))

    @staticmethod
    def _pool(e, mask):
        m = mask.unsqueeze(-1).float()
        return (e * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, x_self, drones, drone_mask, parcels, parcel_mask, scalars):
        e_s = self.self_enc(x_self)
        pd = self._pool(self.drone_enc(drones), drone_mask) if drones.shape[1] > 0 else torch.zeros_like(e_s)
        pp = self._pool(self.parcel_enc(parcels), parcel_mask) if parcels.shape[1] > 0 else torch.zeros_like(e_s)
        return self.head(torch.cat([e_s, pd, pp, scalars], dim=-1)).squeeze(-1)


def collate_global(records: List[dict], device: str = "cpu"):
    """Batch tensors for the centralised critic from records that carry 'x_self', 'g_drones', 'g_parcels' and 'g_scalars'."""
    B = len(records)
    md = max(1, max(len(r["g_drones"]) for r in records))
    mp = max(1, max(len(r["g_parcels"]) for r in records))
    x_self = np.zeros((B, SELF_DIM), dtype=np.float32)
    drones = np.zeros((B, md, DRONE_DIM), dtype=np.float32)
    dmask = np.zeros((B, md), dtype=bool)
    parcels = np.zeros((B, mp, PARCEL_DIM), dtype=np.float32)
    pmask = np.zeros((B, mp), dtype=bool)
    scalars = np.zeros((B, SCALAR_DIM), dtype=np.float32)
    for i, r in enumerate(records):
        x_self[i] = r["x_self"]
        nd, npc = len(r["g_drones"]), len(r["g_parcels"])
        if nd:
            drones[i, :nd] = r["g_drones"]
            dmask[i, :nd] = True
        if npc:
            parcels[i, :npc] = r["g_parcels"]
            pmask[i, :npc] = True
        scalars[i] = r["g_scalars"]
    t = lambda a: torch.from_numpy(a).to(device)
    return t(x_self), t(drones), t(dmask), t(parcels), t(pmask), t(scalars)
