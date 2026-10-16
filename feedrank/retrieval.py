"""Two tower retrieval model trained with in batch sampled softmax and logQ correction."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Dataset
from .features import SequenceExamples, item_batch, pad_history, user_batch


class ItemTower(nn.Module):
    def __init__(self, n_items: int, n_genres: int, dim: int = 64):
        super().__init__()
        self.id_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.genre = nn.Linear(n_genres, 16)
        self.year = nn.Embedding(11, 8)
        self.mlp = nn.Sequential(nn.Linear(dim + 24, 128), nn.ReLU(), nn.Linear(128, dim))

    def forward(self, item, genres, year):
        x = torch.cat([self.id_emb(item), self.genre(genres), self.year(year)], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    def __init__(self, n_items: int, dim: int = 64):
        super().__init__()
        self.hist_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.gender = nn.Embedding(2, 4)
        self.age = nn.Embedding(7, 8)
        self.occ = nn.Embedding(21, 8)
        self.mlp = nn.Sequential(nn.Linear(2 * dim + 20, 256), nn.ReLU(), nn.Linear(256, dim))

    def forward(self, gender, age, occ, hist):
        mask = (hist > 0).float().unsqueeze(-1)
        h = self.hist_emb(hist)
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        # recency: the last 5 items get their own pooled view
        recent = (h[:, -5:] * mask[:, -5:]).sum(1) / mask[:, -5:].sum(1).clamp(min=1.0)
        x = torch.cat([mean, recent, self.gender(gender), self.age(age), self.occ(occ)], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_items: int, n_genres: int, dim: int = 64, temperature: float = 0.05):
        super().__init__()
        self.user = UserTower(n_items, dim)
        self.item = ItemTower(n_items, n_genres, dim)
        self.temperature = temperature


def in_batch_softmax_loss(u: torch.Tensor, v: torch.Tensor, items: torch.Tensor,
                          log_q: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sampled softmax over in batch negatives.

    logQ correction subtracts each candidate's log sampling probability so popular
    items are not over penalized as negatives. Accidental hits (same item twice in
    the batch) are masked out.
    """
    logits = (u @ v.T) / temperature - log_q[items][None, :]
    same = items[:, None] == items[None, :]
    eye = torch.eye(len(items), dtype=torch.bool, device=u.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    labels = torch.arange(len(items), device=u.device)
    return F.cross_entropy(logits, labels)


def train_two_tower(ds: Dataset, device, epochs: int = 6, batch_size: int = 1024,
                    lr: float = 1e-3, dim: int = 64, seed: int = 0, holdout: int = 0, log=print) -> TwoTower:
    """holdout > 0 drops each user's last `holdout` train items (used to build ranker training lists)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    seqs = ds.train_sequences()
    if holdout:
        seqs = [s[:-holdout] for s in seqs]
    examples = SequenceExamples(seqs)
    pop = ds.item_popularity() + 1.0
    log_q = torch.as_tensor(np.log(pop / pop.sum()), dtype=torch.float32, device=device)

    model = TwoTower(ds.n_items, ds.item_genres.shape[1], dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(examples))
        total, steps = 0.0, 0
        for start in range(0, len(order), batch_size):
            users, hists, targets = examples.batch(order[start:start + batch_size])
            u = model.user(**user_batch(ds, users, hists, device))
            ib = item_batch(ds, targets, device)
            v = model.item(ib["item"], ib["genres"], ib["year"])
            loss = in_batch_softmax_loss(u, v, ib["item"], log_q, model.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        log(f"two tower{' fold' if holdout else ''} epoch {epoch + 1}/{epochs} loss {total / steps:.4f}")
    return model


@torch.no_grad()
def item_embeddings(model: TwoTower, ds: Dataset, device) -> torch.Tensor:
    model.eval()
    items = np.arange(ds.n_items + 1)
    ib = item_batch(ds, items, device)
    emb = model.item(ib["item"], ib["genres"], ib["year"])
    emb[0] = 0.0
    return emb


@torch.no_grad()
def user_embeddings(model: TwoTower, ds: Dataset, histories: list[np.ndarray], device,
                    users: np.ndarray | None = None, batch_size: int = 2048) -> torch.Tensor:
    """One embedding per history; users[q] is the owner of histories[q] (defaults to q)."""
    model.eval()
    users = np.arange(len(histories)) if users is None else users
    out = []
    for start in range(0, len(histories), batch_size):
        q = np.arange(start, min(start + batch_size, len(histories)))
        hists = np.stack([pad_history(histories[i]) for i in q])
        out.append(model.user(**user_batch(ds, users[q], hists, device)))
    return torch.cat(out)


@torch.no_grad()
def retrieve(user_emb: torch.Tensor, item_emb: torch.Tensor, seen: list[np.ndarray], k: int,
             batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Exact top k by dot product over the full catalog, excluding already seen items."""
    ids, scores = [], []
    for start in range(0, len(user_emb), batch_size):
        s = user_emb[start:start + batch_size] @ item_emb.T
        s[:, 0] = float("-inf")
        for row, u in enumerate(range(start, start + len(s))):
            s[row, torch.as_tensor(seen[u], dtype=torch.long, device=s.device)] = float("-inf")
        top = torch.topk(s, k, dim=1)
        ids.append(top.indices.cpu().numpy())
        scores.append(top.values.cpu().numpy())
    return np.concatenate(ids), np.concatenate(scores)
