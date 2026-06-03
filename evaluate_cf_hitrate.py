from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DATA_DIR = Path(".")
OUTPUT_DIR = Path("collaborative_output")
SEED = 42
TOP_K = 10


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, global_mean):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self.global_mean = nn.Parameter(
            torch.tensor(global_mean, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, user_idx, item_idx):
        user_vector = self.user_embedding(user_idx)
        item_vector = self.item_embedding(item_idx)

        dot_product = torch.sum(user_vector * item_vector, dim=1)
        user_bias = self.user_bias(user_idx).squeeze(1)
        item_bias = self.item_bias(item_idx).squeeze(1)

        prediction = self.global_mean + user_bias + item_bias + dot_product
        return torch.clamp(prediction, 1.0, 5.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate HitRate@10 for saved collaborative filtering model."
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--negatives", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    with open(OUTPUT_DIR / "collaborative_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    items = pd.read_csv(DATA_DIR / "items.csv")

    num_users = max(
        train["user_idx"].max(),
        val["user_idx"].max(),
        test["user_idx"].max(),
    ) + 1

    num_items = max(
        train["item_idx"].max(),
        val["item_idx"].max(),
        test["item_idx"].max(),
        items["item_idx"].max(),
    ) + 1

    model = MatrixFactorization(
        num_users=int(num_users),
        num_items=int(num_items),
        embedding_dim=int(cfg["embedding_dim"]),
        global_mean=float(cfg["global_mean"]),
    ).to(device)

    model.load_state_dict(
        torch.load(
            OUTPUT_DIR / "matrix_factorization_model.pt",
            map_location=device,
        )
    )
    model.eval()

    rng = np.random.default_rng(SEED)

    test_pairs = test[["user_idx", "item_idx"]].values
    sample_size = min(args.sample_size, len(test_pairs))
    sampled_indices = rng.choice(len(test_pairs), size=sample_size, replace=False)
    test_pairs = test_pairs[sampled_indices]

    all_items = np.arange(int(num_items))

    hits = 0
    reciprocal_ranks = []
    ndcg_scores = []

    for i, (user_idx, positive_item_idx) in enumerate(test_pairs):
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{sample_size}")

        negative_candidates = all_items[all_items != positive_item_idx]
        negative_samples = rng.choice(
            negative_candidates,
            size=args.negatives,
            replace=False,
        )

        candidates = np.concatenate([[positive_item_idx], negative_samples])

        user_tensor = torch.full(
            (len(candidates),),
            int(user_idx),
            dtype=torch.long,
            device=device,
        )
        item_tensor = torch.tensor(
            candidates,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            scores = model(user_tensor, item_tensor).cpu().numpy()

        ranked = candidates[np.argsort(scores)[::-1]]
        positive_rank = np.where(ranked == positive_item_idx)[0]

        if len(positive_rank) == 0:
            continue

        rank = int(positive_rank[0]) + 1

        if rank <= args.top_k:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
            ndcg_scores.append(1.0 / math.log2(rank + 1))
        else:
            reciprocal_ranks.append(0.0)
            ndcg_scores.append(0.0)

    total = len(reciprocal_ranks)

    print(f"\nHitRate@{args.top_k}: {hits / total:.4f}")
    print(f"Precision@{args.top_k}: {hits / total:.4f}")
    print(f"MRR@{args.top_k}:     {np.mean(reciprocal_ranks):.4f}")
    print(f"NDCG@{args.top_k}:    {np.mean(ndcg_scores):.4f}")
    print(f"Evaluated:    {total}")
    print(f"Negatives:    {args.negatives}")


if __name__ == "__main__":
    main()
