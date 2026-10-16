"""DCN v2 style ranker that re scores the retrieval candidates with a listwise loss."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Dataset

# genres + retrieval score, log rank, log popularity, genre affinity, recent genre affinity,
# similarity to last item, max similarity to last 5 items, mean similarity to full history
N_DENSE = 18 + 8


class CrossLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x0, x):
        return x0 * self.lin(x) + x


class DCNRanker(nn.Module):
    def __init__(self, n_items: int, n_cross: int = 3, item_dim: int = 16):
        super().__init__()
        self.gender = nn.Embedding(2, 4)
        self.age = nn.Embedding(7, 8)
        self.occ = nn.Embedding(21, 8)
        self.item = nn.Embedding(n_items + 1, item_dim, padding_idx=0)
        self.year = nn.Embedding(11, 8)
        dim = 4 + 8 + 8 + item_dim + 8 + N_DENSE
        self.norm = nn.BatchNorm1d(N_DENSE)
        self.cross = nn.ModuleList(CrossLayer(dim) for _ in range(n_cross))
        self.deep = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU())
        self.head = nn.Linear(dim + 128, 1)

    def forward(self, gender, age, occ, item, year, dense):
        x0 = torch.cat([self.gender(gender), self.age(age), self.occ(occ), self.item(item),
                        self.year(year), self.norm(dense)], dim=-1)
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return self.head(torch.cat([x, self.deep(x0)], dim=-1)).squeeze(-1)


def genre_profiles(ds: Dataset, histories: list[np.ndarray], last: int | None = None) -> np.ndarray:
    prof = np.zeros((ds.n_users, ds.item_genres.shape[1]), dtype=np.float32)
    for u, h in enumerate(histories):
        h = h[-last:] if last else h
        if len(h):
            prof[u] = ds.item_genres[h].mean(0)
    return prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-8)


@dataclass
class RankContext:
    """Per query state the ranker features are computed from. A query is (user, history)."""

    user: np.ndarray        # (n_queries,) user id of each query
    profile: np.ndarray     # (n_users, 18) normalized genre profile over full history
    recent: np.ndarray      # (n_users, 18) genre profile over last 5 items
    last5: np.ndarray       # (n_users, 5) last 5 item ids, left padded with 0
    hist_mean: np.ndarray   # (n_users, d) mean item tower embedding of the history
    item_emb: np.ndarray    # (n_items + 1, d) item tower embeddings, row 0 = zeros
    log_pop: np.ndarray     # (n_items + 1,)


def genre_profiles_q(ds: Dataset, histories: list[np.ndarray], last: int | None = None) -> np.ndarray:
    prof = np.zeros((len(histories), ds.item_genres.shape[1]), dtype=np.float32)
    for q, h in enumerate(histories):
        h = h[-last:] if last else h
        if len(h):
            prof[q] = ds.item_genres[h].mean(0)
    return prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-8)


def build_context(ds: Dataset, histories: list[np.ndarray], item_emb: np.ndarray,
                  users: np.ndarray | None = None) -> RankContext:
    users = np.arange(len(histories)) if users is None else users
    last5 = np.zeros((len(histories), 5), dtype=np.int64)
    hist_mean = np.zeros((len(histories), item_emb.shape[1]), dtype=np.float32)
    for q, h in enumerate(histories):
        tail = h[-5:]
        if len(tail):
            last5[q, -len(tail):] = tail
            hist_mean[q] = item_emb[h].mean(0)
    return RankContext(users, genre_profiles_q(ds, histories), genre_profiles_q(ds, histories, last=5), last5,
                       hist_mean, item_emb, np.log1p(ds.item_popularity()))


def candidate_features(ds: Dataset, ctx: RankContext, queries: np.ndarray, items: np.ndarray,
                       scores: np.ndarray, ranks: np.ndarray, device) -> dict[str, torch.Tensor]:
    g = ds.item_genres[items]
    gnorm = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
    e = ctx.item_emb[items]                                    # (n, d)
    recent_e = ctx.item_emb[ctx.last5[queries]]                  # (n, 5, d), padding rows are 0
    sims = np.einsum("nd,nkd->nk", e, recent_e)
    dense = np.concatenate([
        g,
        scores[:, None],
        np.log1p(ranks)[:, None],
        ctx.log_pop[items][:, None],
        (gnorm * ctx.profile[queries]).sum(1, keepdims=True),
        (gnorm * ctx.recent[queries]).sum(1, keepdims=True),
        sims[:, -1:],
        sims.max(1, keepdims=True),
        (e * ctx.hist_mean[queries]).sum(1, keepdims=True),
    ], axis=1).astype(np.float32)
    users = ctx.user[queries]
    t = lambda a: torch.as_tensor(a, device=device)
    return {"gender": t(ds.user_gender[users]), "age": t(ds.user_age[users]), "occ": t(ds.user_occ[users]),
            "item": t(items), "year": t(ds.item_year[items]), "dense": t(dense)}


def flatten_candidates(cand: np.ndarray, cand_scores: np.ndarray, user_ids: np.ndarray):
    n, k = cand.shape
    return np.repeat(user_ids, k), cand.reshape(-1), cand_scores.reshape(-1), np.tile(np.arange(k), n)


def build_ranker_lists(cand: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Users whose held out target was retrieved, and the target's position in their candidate list."""
    hit = cand == targets[:, None]
    users = np.where(hit.any(1))[0]
    return users, hit[users].argmax(1)


def list_ndcg(model: DCNRanker, ds: Dataset, ctx: RankContext, cand, cand_scores, targets, device, k: int = 10) -> float:
    from .metrics import ndcg_at_k, target_ranks
    return ndcg_at_k(target_ranks(rerank(model, ds, ctx, cand, cand_scores, device), targets), k)


def train_ranker(ds: Dataset, ctx: RankContext, cand: np.ndarray, cand_scores: np.ndarray,
                 targets: np.ndarray, device, val=None, epochs: int = 10, batch_users: int = 256,
                 lr: float = 1e-3, seed: int = 0, log=print) -> DCNRanker:
    """Listwise softmax over each query's full candidate list (the hard negatives retrieval produced).

    val = (ctx, cand, scores, targets) on a disjoint split, used to keep the best epoch by NDCG@10.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    queries, pos = build_ranker_lists(cand, targets)
    log(f"ranker lists {len(queries)} x {cand.shape[1]} candidates")
    model = DCNRanker(ds.n_items).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    k = cand.shape[1]
    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(queries))
        total, steps = 0.0, 0
        for start in range(0, len(order), batch_users):
            b = order[start:start + batch_users]
            q = queries[b]
            flat = flatten_candidates(cand[q], cand_scores[q], q)
            logits = model(**candidate_features(ds, ctx, *flat, device)).view(len(b), k)
            loss = F.cross_entropy(logits, torch.as_tensor(pos[b], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        msg = f"ranker epoch {epoch + 1}/{epochs} listwise ce {total / steps:.4f}"
        if val is not None:
            score = list_ndcg(model, ds, *val, device)
            msg += f" val ndcg@10 {score:.4f}"
            if score > best:
                best, best_state = score, {n: p.detach().clone() for n, p in model.state_dict().items()}
        log(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def rerank(model: DCNRanker, ds: Dataset, ctx: RankContext, cand: np.ndarray, cand_scores: np.ndarray,
           device, batch_users: int = 512) -> np.ndarray:
    model.eval()
    out = np.zeros_like(cand)
    k = cand.shape[1]
    for start in range(0, len(cand), batch_users):
        c, s = cand[start:start + batch_users], cand_scores[start:start + batch_users]
        flat = flatten_candidates(c, s, np.arange(start, start + len(c)))
        logits = model(**candidate_features(ds, ctx, *flat, device)).view(len(c), k)
        order = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
        out[start:start + len(c)] = np.take_along_axis(c, order, axis=1)
    return out
