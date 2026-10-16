"""Popularity and BPR matrix factorization baselines."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data import Dataset


def popularity_topk(ds: Dataset, seen: list[np.ndarray], k: int) -> np.ndarray:
    pop = ds.item_popularity()
    pop[0] = -np.inf
    order = np.argsort(-pop)
    out = np.zeros((ds.n_users, k), dtype=np.int64)
    for u in range(ds.n_users):
        seen_set = set(seen[u].tolist())
        row = [i for i in order[: k + len(seen_set)] if i not in seen_set][:k]
        out[u] = row
    return out


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.U = nn.Embedding(n_users, dim)
        self.V = nn.Embedding(n_items + 1, dim)
        nn.init.normal_(self.U.weight, std=0.05)
        nn.init.normal_(self.V.weight, std=0.05)


def train_bpr(ds: Dataset, device, epochs: int = 20, batch_size: int = 4096, lr: float = 5e-3,
              reg: float = 1e-5, seed: int = 0, log=print) -> BPRMF:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    users = ds.train["user"].to_numpy()
    items = ds.train["item"].to_numpy()
    model = BPRMF(ds.n_users, ds.n_items).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        order = rng.permutation(len(users))
        total, steps = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            u = torch.as_tensor(users[idx], device=device)
            i = torch.as_tensor(items[idx], device=device)
            j = torch.as_tensor(rng.integers(1, ds.n_items + 1, len(idx)), device=device)
            pu, qi, qj = model.U(u), model.V(i), model.V(j)
            diff = (pu * qi).sum(-1) - (pu * qj).sum(-1)
            loss = -torch.nn.functional.logsigmoid(diff).mean() + reg * (pu.pow(2).sum() + qi.pow(2).sum() + qj.pow(2).sum()) / len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        if (epoch + 1) % 5 == 0:
            log(f"bpr epoch {epoch + 1}/{epochs} loss {total / steps:.4f}")
    return model
