"""Download, parse, and split MovieLens 1M into leave last out train / val / test."""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
AGE_BUCKETS = [1, 18, 25, 35, 45, 50, 56]


def download_ml1m(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    target = data_dir / "ml-1m"
    if (target / "ratings.dat").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {ML1M_URL}")
    with urllib.request.urlopen(ML1M_URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(data_dir)
    return target


def _read(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep="::", engine="python", names=names, encoding="latin-1")


@dataclass
class Dataset:
    """Contiguous ids: users 0..n_users-1, items 1..n_items (0 is padding)."""

    n_users: int
    n_items: int
    train: pd.DataFrame          # user, item, ts
    val: pd.DataFrame            # one row per user
    test: pd.DataFrame           # one row per user
    user_gender: np.ndarray      # (n_users,)
    user_age: np.ndarray         # (n_users,)
    user_occ: np.ndarray         # (n_users,)
    item_genres: np.ndarray      # (n_items + 1, 18) multi hot, row 0 = padding
    item_year: np.ndarray        # (n_items + 1,) decade bucket
    item_title: list[str]        # index aligned with item ids
    raw_item_id: np.ndarray      # (n_items + 1,) original MovieID

    def item_popularity(self) -> np.ndarray:
        counts = np.bincount(self.train["item"].to_numpy(), minlength=self.n_items + 1)
        return counts.astype(np.float64)

    def train_sequences(self) -> list[np.ndarray]:
        seqs = [np.empty(0, dtype=np.int64)] * self.n_users
        for u, g in self.train.sort_values(["user", "ts"]).groupby("user"):
            seqs[u] = g["item"].to_numpy()
        return seqs


def leave_last_out(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per user, by time: last interaction is test, second to last is val, rest is train."""
    ratings = ratings.sort_values(["user", "ts", "item"], kind="mergesort")
    rank_from_end = ratings.groupby("user").cumcount(ascending=False)
    test = ratings[rank_from_end == 0]
    val = ratings[rank_from_end == 1]
    train = ratings[rank_from_end >= 2]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_dataset(ratings: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame) -> Dataset:
    ratings = ratings.copy()
    user_ids = np.sort(ratings["user"].unique())
    item_ids = np.sort(ratings["item"].unique())
    u_map = {u: i for i, u in enumerate(user_ids)}
    i_map = {m: i + 1 for i, m in enumerate(item_ids)}
    ratings["user"] = ratings["user"].map(u_map)
    ratings["item"] = ratings["item"].map(i_map)
    train, val, test = leave_last_out(ratings[["user", "item", "ts"]])

    n_users, n_items = len(user_ids), len(item_ids)
    users = users.set_index("user").reindex(user_ids)
    gender = (users["gender"] == "M").astype(np.int64).to_numpy()
    age = users["age"].map({a: i for i, a in enumerate(AGE_BUCKETS)}).fillna(0).astype(np.int64).to_numpy()
    occ = users["occupation"].fillna(0).astype(np.int64).to_numpy()

    movies = movies.set_index("item").reindex(item_ids)
    genres = np.zeros((n_items + 1, len(GENRES)), dtype=np.float32)
    years = np.zeros(n_items + 1, dtype=np.int64)
    titles = ["<pad>"]
    for row_idx, (title, gstr) in enumerate(zip(movies["title"], movies["genres"]), start=1):
        for g in str(gstr).split("|"):
            if g in GENRES:
                genres[row_idx, GENRES.index(g)] = 1.0
        m = re.search(r"\((\d{4})\)\s*$", str(title))
        year = int(m.group(1)) if m else 1990
        years[row_idx] = int(np.clip((year - 1910) // 10, 0, 9)) + 1
        titles.append(str(title))
    raw = np.concatenate([[0], item_ids])
    return Dataset(n_users, n_items, train, val, test, gender, age, occ, genres, years, titles, raw)


def load_ml1m(data_dir: str | Path) -> Dataset:
    root = download_ml1m(data_dir)
    ratings = _read(root / "ratings.dat", ["user", "item", "rating", "ts"])
    users = _read(root / "users.dat", ["user", "gender", "age", "occupation", "zip"])
    movies = _read(root / "movies.dat", ["item", "title", "genres"])
    return build_dataset(ratings, users, movies)
