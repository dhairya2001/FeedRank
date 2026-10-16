# FeedRank

A two stage recommender on MovieLens 1M (6,040 users, 3,706 movies, 1M interactions), built the way
large feed and video recommenders are: a fast retrieval stage narrows the full catalog to 100
candidates, then a heavier ranking model orders them.

* **Retrieval**: two tower model in PyTorch. The user tower pools the watch history (overall and
  last 5 items) with demographic features; the item tower combines an ID embedding with genres and
  release decade. Trained with in batch sampled softmax, logQ popularity correction, and accidental
  hit masking. Serving is an exact dot product top k over L2 normalized embeddings.
* **Ranking**: DCN v2 style cross network plus a deep MLP over user, item, retrieval score, rank,
  popularity, genre affinity, and embedding similarity to recent history. Trained with a listwise
  softmax over all 100 retrieved candidates (hard negatives). Training lists come from a fold
  retrieval model that never saw each user's last 8 train items, so the ranker does not learn from
  memorized retrieval scores; the validation split picks the best epoch.
* **Baselines**: popularity and BPR matrix factorization (collaborative filtering).
* **Evaluation**: leave last out split by timestamp (no future leakage), full catalog ranking with
  already seen items excluded. Reports Recall@K, NDCG@K, MRR, catalog coverage, and the average
  popularity percentile of recommendations (to watch for popularity bias).
* **Serving**: FastAPI endpoint `/recommend/{user_id}` and a latency benchmark of the full
  retrieve plus rerank path.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m feedrank.train --data data --out artifacts   # downloads MovieLens 1M on first run
uvicorn feedrank.serve:app --port 8000                 # GET /recommend/42?k=10
```

Results are written to `artifacts/RESULTS.md` and `artifacts/results.json`.

## Results (seed 0, CPU, test split, full catalog, seen items excluded)

| model | recall@10 | ndcg@10 | recall@50 | recall@100 | mrr | coverage@10 | pop_pct@10 |
|---|---|---|---|---|---|---|---|
| popularity | 0.0353 | 0.0174 | 0.1290 | 0.2132 | 0.0172 | 0.0534 | 0.9965 |
| bpr_mf | 0.0627 | 0.0300 | 0.2179 | 0.3533 | 0.0288 | 0.7277 | 0.8422 |
| two_tower_val | 0.2505 | 0.1356 | 0.5267 | 0.6439 | 0.1155 | 0.5734 | 0.8665 |
| two_tower | 0.2313 | 0.1305 | 0.4833 | 0.6008 | 0.1135 | 0.5696 | 0.8686 |
| two_tower+dcn_ranker_val | 0.2644 | 0.1434 | 0.5349 | 0.6439 | 0.1213 | 0.5210 | 0.8857 |
| two_tower+dcn_ranker | 0.2376 | 0.1324 | 0.4974 | 0.6008 | 0.1144 | 0.5157 | 0.8875 |

end to end recommend() latency on cpu: p50 0.24 ms, p95 0.25 ms, p99 0.26 ms

The two tower model reaches 3.7x the Recall@10 of BPR matrix factorization, and the DCN ranker adds a
further lift on top of retrieval (Recall@10 0.2313 to 0.2376, NDCG@10 0.1305 to 0.1324).
