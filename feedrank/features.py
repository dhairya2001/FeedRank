"""Turn user histories and side features into padded tensors."""
from __future__ import annotations

import numpy as np
import torch

from .data import Dataset

HIST_LEN = 50


def pad_history(seq: np.ndarray, length: int = HIST_LEN) -> np.ndarray:
    out = np.zeros(length, dtype=np.int64)
    tail = seq[-length:]
    if len(tail):
        out[-len(tail):] = tail
    return out


def user_batch(ds: Dataset, users: np.ndarray, histories: np.ndarray, device) -> dict[str, torch.Tensor]:
    return {
        "gender": torch.as_tensor(ds.user_gender[users], device=device),
        "age": torch.as_tensor(ds.user_age[users], device=device),
        "occ": torch.as_tensor(ds.user_occ[users], device=device),
        "hist": torch.as_tensor(histories, device=device),
    }


def item_batch(ds: Dataset, items: np.ndarray, device) -> dict[str, torch.Tensor]:
    return {
        "item": torch.as_tensor(items, device=device),
        "genres": torch.as_tensor(ds.item_genres[items], device=device),
        "year": torch.as_tensor(ds.item_year[items], device=device),
    }


class SequenceExamples:
    """Every train position t >= 1 of every user: (user, history before t, target at t)."""

    def __init__(self, seqs: list[np.ndarray], min_hist: int = 1):
        users, pos = [], []
        for u, s in enumerate(seqs):
            for t in range(min_hist, len(s)):
                users.append(u)
                pos.append(t)
        self.users = np.asarray(users, dtype=np.int64)
        self.pos = np.asarray(pos, dtype=np.int64)
        self.seqs = seqs

    def __len__(self) -> int:
        return len(self.users)

    def batch(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = self.users[idx]
        hists = np.stack([pad_history(self.seqs[u][:t]) for u, t in zip(users, self.pos[idx])])
        targets = np.array([self.seqs[u][t] for u, t in zip(users, self.pos[idx])], dtype=np.int64)
        return users, hists, targets
