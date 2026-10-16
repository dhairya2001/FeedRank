"""Online recommender: retrieve 100 candidates, re rank with DCN, return top k.

Run: uvicorn feedrank.serve:app  (after `python -m feedrank.train` writes artifacts/)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_ml1m
from .features import pad_history, user_batch
from .ranking import DCNRanker, build_context, candidate_features
from .retrieval import TwoTower, item_embeddings


class Recommender:
    def __init__(self, ds, two_tower: TwoTower, ranker: DCNRanker, device, n_candidates: int = 100):
        self.ds, self.tt, self.ranker, self.device = ds, two_tower.eval(), ranker.eval(), device
        self.n_candidates = n_candidates
        self.item_emb = item_embeddings(two_tower, ds, device)
        seqs = ds.train_sequences()
        extra = ds.val.set_index("user")["item"]
        self.histories = [np.append(seqs[u], extra.get(u, [])).astype(np.int64) for u in range(ds.n_users)]
        self.ctx = build_context(ds, self.histories, self.item_emb.cpu().numpy())

    @torch.no_grad()
    def recommend(self, user: int, k: int = 10) -> list[int]:
        hist = pad_history(self.histories[user])[None]
        u = self.tt.user(**user_batch(self.ds, np.array([user]), hist, self.device))
        s = (u @ self.item_emb.T)[0]
        s[0] = float("-inf")
        s[torch.as_tensor(self.histories[user], dtype=torch.long, device=self.device)] = float("-inf")
        top = torch.topk(s, self.n_candidates)
        items, scores = top.indices.cpu().numpy(), top.values.cpu().numpy()
        users = np.full(len(items), user)
        feats = candidate_features(self.ds, self.ctx, users, items, scores, np.arange(len(items)), self.device)
        order = torch.argsort(self.ranker(**feats), descending=True).cpu().numpy()
        return items[order[:k]].tolist()

    def benchmark(self, n: int = 1000, k: int = 10, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        users = rng.integers(0, self.ds.n_users, n)
        for u in users[:20]:
            self.recommend(int(u), k)  # warm up
        times = []
        for u in users:
            t0 = time.perf_counter()
            self.recommend(int(u), k)
            times.append((time.perf_counter() - t0) * 1000)
        return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                "p99_ms": float(np.percentile(times, 99))}


def load_recommender(artifacts: str | Path = "artifacts", data_dir: str | Path = "data", device="cpu") -> Recommender:
    ds = load_ml1m(data_dir)
    tt = TwoTower(ds.n_items, ds.item_genres.shape[1])
    tt.load_state_dict(torch.load(Path(artifacts) / "two_tower.pt", map_location=device))
    rk = DCNRanker(ds.n_items)
    rk.load_state_dict(torch.load(Path(artifacts) / "ranker.pt", map_location=device))
    return Recommender(ds, tt.to(device), rk.to(device), device)


def create_app(rec: Recommender | None = None):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="FeedRank")
    state = {"rec": rec}

    def get() -> Recommender:
        if state["rec"] is None:
            state["rec"] = load_recommender(os.environ.get("FEEDRANK_ARTIFACTS", "artifacts"),
                                            os.environ.get("FEEDRANK_DATA", "data"))
        return state["rec"]

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/recommend/{user_id}")
    def recommend(user_id: int, k: int = 10):
        r = get()
        if not 0 <= user_id < r.ds.n_users or not 1 <= k <= r.n_candidates:
            raise HTTPException(status_code=400, detail="bad user_id or k")
        items = r.recommend(user_id, k)
        return {"user_id": user_id, "items": [{"item": i, "movie_id": int(r.ds.raw_item_id[i]),
                                                "title": r.ds.item_title[i]} for i in items]}

    return app


app = create_app()
