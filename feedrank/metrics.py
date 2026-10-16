"""Ranking metrics for single held out target per user."""
from __future__ import annotations

import numpy as np


def target_ranks(topk: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0 based rank of target inside each top k row, or -1 if absent."""
    hits = topk == targets[:, None]
    ranks = np.where(hits.any(axis=1), hits.argmax(axis=1), -1)
    return ranks


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean((ranks >= 0) & (ranks < k)))


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    ok = (ranks >= 0) & (ranks < k)
    gains = np.zeros(len(ranks))
    gains[ok] = 1.0 / np.log2(ranks[ok] + 2)
    return float(gains.mean())


def mrr(ranks: np.ndarray) -> float:
    rr = np.zeros(len(ranks))
    ok = ranks >= 0
    rr[ok] = 1.0 / (ranks[ok] + 1)
    return float(rr.mean())


def catalog_coverage(topk: np.ndarray, n_items: int) -> float:
    return float(len(np.unique(topk)) / n_items)


def mean_popularity_percentile(topk: np.ndarray, popularity: np.ndarray) -> float:
    """Average popularity percentile (1.0 = most popular) of recommended items."""
    order = np.argsort(np.argsort(popularity[1:])) / max(len(popularity) - 2, 1)
    pct = np.concatenate([[0.0], order])
    return float(pct[topk].mean())


def report(topk: np.ndarray, targets: np.ndarray, n_items: int, popularity: np.ndarray, ks=(10, 50, 100)) -> dict:
    ranks = target_ranks(topk, targets)
    out = {f"recall@{k}": recall_at_k(ranks, k) for k in ks if k <= topk.shape[1]}
    out.update({f"ndcg@{k}": ndcg_at_k(ranks, k) for k in ks if k <= topk.shape[1]})
    out["mrr"] = mrr(ranks)
    out["coverage@10"] = catalog_coverage(topk[:, :10], n_items)
    out["pop_pct@10"] = mean_popularity_percentile(topk[:, :10], popularity)
    return out
