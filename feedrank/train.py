"""End to end pipeline: baselines, two tower retrieval, DCN ranking, evaluation, latency.

python -m feedrank.train --data data --out artifacts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import metrics
from .baselines import popularity_topk, train_bpr
from .data import load_ml1m
from .ranking import build_context, rerank, train_ranker
from .retrieval import item_embeddings, retrieve, train_two_tower, user_embeddings
from .serve import Recommender

N_CAND = 100
HOLDOUT = 8


def pick_device(name: str):
    if name != "auto":
        return torch.device(name)
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tt-epochs", type=int, default=6)
    ap.add_argument("--bpr-epochs", type=int, default=20)
    ap.add_argument("--ranker-epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_ml1m(args.data)
    print(f"users {ds.n_users} items {ds.n_items} train {len(ds.train)} val {len(ds.val)} test {len(ds.test)}")
    pop = ds.item_popularity()
    log_pop = np.log1p(pop)

    seqs = ds.train_sequences()
    val_target = ds.val.sort_values("user")["item"].to_numpy()
    test_target = ds.test.sort_values("user")["item"].to_numpy()
    hist_test = [np.append(seqs[u], val_target[u]).astype(np.int64) for u in range(ds.n_users)]
    results: dict[str, dict] = {}

    # Baselines
    results["popularity"] = metrics.report(popularity_topk(ds, hist_test, N_CAND), test_target, ds.n_items, pop)

    t0 = time.time()
    bpr = train_bpr(ds, device, epochs=args.bpr_epochs, seed=args.seed)
    bpr_top, _ = retrieve(bpr.U.weight.detach(), bpr.V.weight.detach(), hist_test, N_CAND)
    results["bpr_mf"] = metrics.report(bpr_top, test_target, ds.n_items, pop)
    print(f"bpr done in {time.time() - t0:.0f}s")

    # Stage 1: two tower retrieval
    t0 = time.time()
    tt = train_two_tower(ds, device, epochs=args.tt_epochs, seed=args.seed)
    print(f"two tower done in {time.time() - t0:.0f}s")
    item_emb = item_embeddings(tt, ds, device)

    val_cand, val_scores = retrieve(user_embeddings(tt, ds, seqs, device), item_emb, seqs, N_CAND)
    results["two_tower_val"] = metrics.report(val_cand, val_target, ds.n_items, pop)
    test_cand, test_scores = retrieve(user_embeddings(tt, ds, hist_test, device), item_emb, hist_test, N_CAND)
    results["two_tower"] = metrics.report(test_cand, test_target, ds.n_items, pop)

    # Stage 2: listwise ranker. Training lists come from a fold retrieval model trained without each
    # user's last HOLDOUT train items, so candidate scores for those targets are not memorized.
    fold_tt = train_two_tower(ds, device, epochs=args.tt_epochs, seed=args.seed + 1, holdout=HOLDOUT)
    fold_emb = item_embeddings(fold_tt, ds, device)
    q_users, q_hist, q_target = [], [], []
    for u, s in enumerate(seqs):
        for j in range(HOLDOUT):
            cut = len(s) - HOLDOUT + j
            q_users.append(u); q_hist.append(s[:cut]); q_target.append(s[cut])
    q_users, q_target = np.array(q_users), np.array(q_target)
    q_emb = user_embeddings(fold_tt, ds, q_hist, device, users=q_users)
    fold_cand, fold_scores = retrieve(q_emb, fold_emb, q_hist, N_CAND)
    emb_np = item_emb.cpu().numpy()
    val_ctx = build_context(ds, seqs, emb_np)
    ranker = train_ranker(ds, build_context(ds, q_hist, fold_emb.cpu().numpy(), q_users),
                          fold_cand, fold_scores, q_target, device,
                          val=(val_ctx, val_cand, val_scores, val_target),
                          epochs=args.ranker_epochs, seed=args.seed)
    results["two_tower+dcn_ranker_val"] = metrics.report(rerank(ranker, ds, val_ctx, val_cand, val_scores, device),
                                                         val_target, ds.n_items, pop)
    reranked = rerank(ranker, ds, build_context(ds, hist_test, emb_np), test_cand, test_scores, device)
    results["two_tower+dcn_ranker"] = metrics.report(reranked, test_target, ds.n_items, pop)

    torch.save(tt.state_dict(), out / "two_tower.pt")
    torch.save(ranker.state_dict(), out / "ranker.pt")

    rec = Recommender(ds, tt, ranker, device, n_candidates=N_CAND)
    results["latency"] = rec.benchmark()

    (out / "results.json").write_text(json.dumps(results, indent=2))
    keys = ["recall@10", "ndcg@10", "recall@50", "recall@100", "mrr", "coverage@10", "pop_pct@10"]
    lines = ["| model | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for name, r in results.items():
        if name == "latency":
            continue
        lines.append(f"| {name} | " + " | ".join(f"{r.get(k, float('nan')):.4f}" for k in keys) + " |")
    lat = results["latency"]
    lines.append(f"\nend to end recommend() latency on {device}: p50 {lat['p50_ms']:.2f} ms, "
                 f"p95 {lat['p95_ms']:.2f} ms, p99 {lat['p99_ms']:.2f} ms")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
