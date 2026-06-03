"""
GPU-friendly hybrid recommender for the prepared Amazon Electronics dataset.

The hybrid score is:
    alpha * normalized_collaborative_score
    + (1 - alpha) * normalized_content_score

alpha=1.0 is pure collaborative filtering. alpha=0.0 is pure content-based.

Example:
    python hybrid_recommender.py --data-dir data\\processed --epochs 4
    python hybrid_recommender.py --data-dir data\\processed_sample --quick
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Config:
    factors: int = 96
    content_dim: int = 128
    max_tfidf_features: int = 50000
    batch_size: int = 16384
    epochs: int = 4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    eval_negatives: int = 100
    top_k: int = 10
    max_val_users: int = 30000
    max_test_users: int = 30000
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a CF/content hybrid.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--factors", type=int, default=Config.factors)
    parser.add_argument("--content-dim", type=int, default=Config.content_dim)
    parser.add_argument("--eval-negatives", type=int, default=Config.eval_negatives)
    parser.add_argument("--max-val-users", type=int, default=Config.max_val_users)
    parser.add_argument("--max-test-users", type=int, default=Config.max_test_users)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
        help="Comma-separated alpha values. alpha=1 means pure CF.",
    )
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke test: 1 epoch and 5,000 validation/test users.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, factors)
        self.item_factors = nn.Embedding(n_items, factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_factors(users)
        item_vec = self.item_factors(items)
        dot = (user_vec * item_vec).sum(dim=-1)
        return dot + self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1)


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    items = pd.read_csv(data_dir / "items.csv")
    n_users = int(max(train["user_idx"].max(), val["user_idx"].max(), test["user_idx"].max()) + 1)
    n_items = int(max(items["item_idx"].max(), train["item_idx"].max(), val["item_idx"].max(), test["item_idx"].max()) + 1)
    return train, val, test, items, n_users, n_items


def build_content_vectors(
    train: pd.DataFrame,
    items: pd.DataFrame,
    n_users: int,
    n_items: int,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer, TruncatedSVD]:
    print("Building content vectors from product metadata...")
    text = (
        items.get("content_text", "").fillna("")
        + " "
        + items.get("title", "").fillna("")
        + " "
        + items.get("brand", "").fillna("")
        + " "
        + items.get("categories", "").fillna("")
    )
    vectorizer = TfidfVectorizer(
        max_features=config.max_tfidf_features,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        strip_accents="unicode",
    )
    tfidf = vectorizer.fit_transform(text)
    n_components = min(config.content_dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if n_components < 2:
        raise RuntimeError("Not enough item text to create content vectors.")
    svd = TruncatedSVD(n_components=n_components, random_state=config.seed)
    item_vectors = svd.fit_transform(tfidf).astype(np.float32)
    item_vectors = normalize(item_vectors, norm="l2").astype(np.float32)

    if len(item_vectors) < n_items:
        padded = np.zeros((n_items, item_vectors.shape[1]), dtype=np.float32)
        padded[items["item_idx"].to_numpy(np.int64)] = item_vectors
        item_vectors = padded

    users = train["user_idx"].to_numpy(np.int64)
    item_idx = train["item_idx"].to_numpy(np.int64)
    ratings = train["rating"].to_numpy(np.float32)
    weights = np.where(ratings >= 4.0, ratings - 3.0, np.where(ratings <= 2.0, -0.5 * (3.0 - ratings), 0.0))
    user_vectors = np.zeros((n_users, item_vectors.shape[1]), dtype=np.float32)
    weight_sums = np.zeros(n_users, dtype=np.float32)
    np.add.at(user_vectors, users, item_vectors[item_idx] * weights[:, None])
    np.add.at(weight_sums, users, np.abs(weights))
    user_vectors /= np.maximum(weight_sums[:, None], 1e-6)
    user_vectors = normalize(user_vectors, norm="l2").astype(np.float32)
    explained = float(svd.explained_variance_ratio_.sum())
    print(f"Content shape | users: {user_vectors.shape} | items: {item_vectors.shape} | SVD variance: {explained:.3f}")
    return user_vectors, item_vectors, vectorizer, svd


def train_cf_model(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    config: Config,
    device: torch.device,
) -> MatrixFactorization:
    positives = train.loc[train["rating"] >= 4.0, ["user_idx", "item_idx"]].copy()
    if positives.empty:
        raise RuntimeError("No positive training rows with rating >= 4.0.")
    dataset = TensorDataset(
        torch.from_numpy(positives["user_idx"].to_numpy(np.int64)),
        torch.from_numpy(positives["item_idx"].to_numpy(np.int64)),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    model = MatrixFactorization(n_users, n_items, config.factors).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    print(f"Training collaborative model on {device} with {len(positives):,} positive interactions...")
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        row_count = 0
        model.train()
        for users_cpu, pos_items_cpu in loader:
            users = users_cpu.to(device, non_blocking=True)
            pos_items = pos_items_cpu.to(device, non_blocking=True)
            neg_items = torch.randint(0, n_items, pos_items.shape, device=device, generator=generator)
            pos_scores = model.score(users, pos_items)
            neg_scores = model.score(users, neg_items)
            loss = F.softplus(-(pos_scores - neg_scores)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(users)
            row_count += len(users)
        print(f"Epoch {epoch:02d} | BPR loss: {total_loss / max(row_count, 1):.4f}")
    return model


def build_seen(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("user_idx")["item_idx"].apply(lambda values: set(map(int, values))).to_dict()


def sample_eval_rows(frame: pd.DataFrame, max_users: int, seed: int) -> pd.DataFrame:
    positives = frame.loc[frame["rating"] >= 4.0, ["user_idx", "item_idx", "rating"]].copy()
    if max_users and len(positives) > max_users:
        positives = positives.sample(max_users, random_state=seed)
    return positives.reset_index(drop=True)


def make_candidates(
    rows: pd.DataFrame,
    seen: dict[int, set[int]],
    n_items: int,
    negatives: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    users = rows["user_idx"].to_numpy(np.int64)
    positives = rows["item_idx"].to_numpy(np.int64)
    candidates = np.empty((len(rows), negatives + 1), dtype=np.int64)
    candidates[:, 0] = positives
    all_items = np.arange(n_items, dtype=np.int64)
    for row_idx, (user, positive) in enumerate(zip(users, positives)):
        blocked = seen.get(int(user), set()) | {int(positive)}
        picked: list[int] = []
        while len(picked) < negatives:
            draw = rng.choice(all_items, size=(negatives - len(picked)) * 2, replace=True)
            picked.extend(int(item) for item in draw if int(item) not in blocked)
        candidates[row_idx, 1:] = picked[:negatives]
    return users, candidates


def zscore(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True).clamp_min(1e-6)


def evaluate_alphas(
    name: str,
    frame: pd.DataFrame,
    model: MatrixFactorization,
    content_user_vectors: np.ndarray,
    content_item_vectors: np.ndarray,
    seen: dict[int, set[int]],
    n_items: int,
    alphas: list[float],
    config: Config,
    device: torch.device,
    max_users: int,
) -> pd.DataFrame:
    rows = sample_eval_rows(frame, max_users, config.seed)
    if rows.empty:
        raise RuntimeError(f"{name} evaluation has no held-out ratings >= 4.0.")
    rng = np.random.default_rng(config.seed)
    users_np, candidates_np = make_candidates(rows, seen, n_items, config.eval_negatives, rng)
    users = torch.from_numpy(users_np).to(device)
    candidates = torch.from_numpy(candidates_np).to(device)
    content_users = torch.from_numpy(content_user_vectors).to(device)
    content_items = torch.from_numpy(content_item_vectors).to(device)

    metrics = {alpha: {"hits": [], "mrr": [], "ndcg": []} for alpha in alphas}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), config.batch_size):
            end = min(start + config.batch_size, len(rows))
            batch_users = users[start:end]
            batch_candidates = candidates[start:end]
            expanded_users = batch_users[:, None].expand_as(batch_candidates)
            cf_scores = model.score(expanded_users.reshape(-1), batch_candidates.reshape(-1)).view_as(batch_candidates)
            cb_scores = (content_users[batch_users][:, None, :] * content_items[batch_candidates]).sum(dim=-1)
            cf_scores = zscore(cf_scores)
            cb_scores = zscore(cb_scores)
            for alpha in alphas:
                scores = alpha * cf_scores + (1.0 - alpha) * cb_scores
                ranks = (scores[:, 1:] >= scores[:, [0]]).sum(dim=1) + 1
                ranks_cpu = ranks.detach().cpu().numpy()
                hit = ranks_cpu <= config.top_k
                metrics[alpha]["hits"].extend(hit.astype(np.float32))
                metrics[alpha]["mrr"].extend((1.0 / ranks_cpu).astype(np.float32))
                metrics[alpha]["ndcg"].extend((1.0 / np.log2(ranks_cpu + 1)).astype(np.float32) * hit)

    summary = []
    for alpha in alphas:
        row = {
            "split": name,
            "alpha": alpha,
            f"hit_rate@{config.top_k}": float(np.mean(metrics[alpha]["hits"])),
            "mrr": float(np.mean(metrics[alpha]["mrr"])),
            f"ndcg@{config.top_k}": float(np.mean(metrics[alpha]["ndcg"])),
            "users_evaluated": int(len(rows)),
            "negatives_per_user": int(config.eval_negatives),
        }
        summary.append(row)
    return pd.DataFrame(summary).sort_values([f"hit_rate@{config.top_k}", "mrr"], ascending=False)


def save_outputs(
    output_dir: Path,
    config: Config,
    val_results: pd.DataFrame,
    test_results: pd.DataFrame,
    best_alpha: float,
    model: MatrixFactorization,
    content_user_vectors: np.ndarray,
    content_item_vectors: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    val_results.to_csv(output_dir / "hybrid_alpha_sweep_val.csv", index=False)
    test_results.to_csv(output_dir / "hybrid_test_results.csv", index=False)
    torch.save(model.state_dict(), output_dir / "hybrid_cf_model.pt")
    np.save(output_dir / "hybrid_content_user_vectors.npy", content_user_vectors)
    np.save(output_dir / "hybrid_content_item_vectors.npy", content_item_vectors)
    with (output_dir / "hybrid_config.json").open("w", encoding="utf-8") as target:
        json.dump({"config": asdict(config), "best_alpha": best_alpha}, target, indent=2)


def main() -> None:
    args = parse_args()
    config = Config(
        factors=args.factors,
        content_dim=args.content_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_negatives=args.eval_negatives,
        max_val_users=args.max_val_users,
        max_test_users=args.max_test_users,
        seed=args.seed,
    )
    if args.quick:
        config.epochs = 1
        config.max_val_users = 5000
        config.max_test_users = 5000
        config.max_tfidf_features = 12000
        config.content_dim = min(config.content_dim, 64)

    seed_everything(config.seed)
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "hybrid_output").resolve()
    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available; running on CPU.")

    train, val, test, items, n_users, n_items = load_data(data_dir)
    print(f"Data | train: {len(train):,} | val: {len(val):,} | test: {len(test):,} | users: {n_users:,} | items: {n_items:,}")
    content_user_vectors, content_item_vectors, _, _ = build_content_vectors(train, items, n_users, n_items, config)
    model = train_cf_model(train, n_users, n_items, config, device)
    seen = build_seen(train)

    val_results = evaluate_alphas(
        "val", val, model, content_user_vectors, content_item_vectors,
        seen, n_items, alphas, config, device, config.max_val_users
    )
    best_alpha = float(val_results.iloc[0]["alpha"])
    print("\nValidation alpha sweep:")
    print(val_results.to_string(index=False))
    print(f"\nBest alpha from validation: {best_alpha:.2f}")

    test_results = evaluate_alphas(
        "test", test, model, content_user_vectors, content_item_vectors,
        seen, n_items, [best_alpha, 0.0, 1.0], config, device, config.max_test_users
    )
    print("\nTest results:")
    print(test_results.to_string(index=False))
    save_outputs(output_dir, config, val_results, test_results, best_alpha, model, content_user_vectors, content_item_vectors)
    print(f"\nSaved hybrid artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
