import numpy as np
import pandas as pd
import torch

from feedrank import metrics
from feedrank.data import build_dataset, leave_last_out
from feedrank.features import pad_history
from feedrank.ranking import DCNRanker, N_DENSE, build_context, build_ranker_lists, candidate_features
from feedrank.retrieval import TwoTower, in_batch_softmax_loss, retrieve


def toy():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 21):
        items = rng.choice(np.arange(1, 31), 8, replace=False)
        rows += [(u, int(i), 4, t) for t, i in enumerate(items)]
    ratings = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    users = pd.DataFrame({"user": range(1, 21), "gender": ["M", "F"] * 10, "age": [25] * 20,
                          "occupation": [3] * 20, "zip": ["0"] * 20})
    movies = pd.DataFrame({"item": range(1, 31), "title": [f"Movie {i} (1995)" for i in range(1, 31)],
                           "genres": ["Comedy|Drama"] * 30})
    return build_dataset(ratings, users, movies)


def test_split_has_no_time_leakage():
    ds = toy()
    last_train = ds.train.groupby("user")["ts"].max()
    assert (ds.val.set_index("user")["ts"] >= last_train).all()
    assert (ds.test.set_index("user")["ts"] >= ds.val.set_index("user")["ts"]).all()
    assert len(ds.val) == len(ds.test) == ds.n_users


def test_leave_last_out_sizes():
    df = pd.DataFrame({"user": [0] * 5, "item": [1, 2, 3, 4, 5], "ts": [1, 2, 3, 4, 5]})
    tr, va, te = leave_last_out(df)
    assert tr["item"].tolist() == [1, 2, 3] and va["item"].tolist() == [4] and te["item"].tolist() == [5]


def test_metrics():
    topk = np.array([[5, 3, 9], [1, 2, 4]])
    ranks = metrics.target_ranks(topk, np.array([5, 4]))
    assert ranks.tolist() == [0, 2]
    assert metrics.recall_at_k(ranks, 1) == 0.5
    assert np.isclose(metrics.ndcg_at_k(ranks, 3), (1 + 1 / np.log2(4)) / 2)
    assert metrics.target_ranks(topk, np.array([7, 7])).tolist() == [-1, -1]


def test_pad_history_left_pads_and_truncates():
    assert pad_history(np.array([1, 2]), 4).tolist() == [0, 0, 1, 2]
    assert pad_history(np.arange(1, 10), 3).tolist() == [7, 8, 9]


def test_two_tower_outputs_unit_vectors():
    ds = toy()
    m = TwoTower(ds.n_items, 18, dim=16)
    hist = torch.as_tensor(np.stack([pad_history(np.array([1, 2, 3]))] * 4))
    u = m.user(torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long), hist)
    assert u.shape == (4, 16)
    assert torch.allclose(u.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_in_batch_loss_masks_accidental_hits():
    u = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    items = torch.tensor([4, 4, 7])
    log_q = torch.zeros(10)
    loss = in_batch_softmax_loss(u, u.clone(), items, log_q, 0.1)
    assert torch.isfinite(loss)


def test_retrieve_excludes_seen_items():
    user = torch.eye(3)
    items = torch.eye(3)
    items = torch.cat([torch.zeros(1, 3), items])  # item 0 is padding
    top, _ = retrieve(user, items, [np.array([1]), np.array([]), np.array([3])], k=2)
    assert 1 not in top[0] and 3 not in top[2] and 0 not in top


def test_ranker_lists_and_features():
    ds = toy()
    cand = np.array([[3, 5, 7, 9, 11]] * ds.n_users)
    targets = np.full(ds.n_users, 7)
    targets[0] = 30  # not retrieved, user dropped
    users, pos = build_ranker_lists(cand, targets)
    assert 0 not in users and (pos == 2).all()
    emb = np.random.default_rng(0).normal(size=(ds.n_items + 1, 8)).astype(np.float32)
    emb[0] = 0
    ctx = build_context(ds, ds.train_sequences(), emb)
    feats = candidate_features(ds, ctx, np.zeros(5, dtype=np.int64), cand[0], np.ones(5, dtype=np.float32), np.arange(5), "cpu")
    assert feats["dense"].shape == (5, N_DENSE)
    out = DCNRanker(ds.n_items).eval()(**feats)
    assert out.shape == (5,)
