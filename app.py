import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Electronics Recommender",
    page_icon="",
    layout="centered",
)

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(".")
OUTPUT_DIR = Path("collaborative_output")

# ── model definition (must match training) ───────────────────────────────────
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, global_mean):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias      = nn.Embedding(num_users, 1)
        self.item_bias      = nn.Embedding(num_items, 1)
        self.global_mean    = nn.Parameter(
            torch.tensor(global_mean, dtype=torch.float32), requires_grad=False
        )

    def forward(self, user_idx, item_idx):
        u   = self.user_embedding(user_idx)
        v   = self.item_embedding(item_idx)
        dot = torch.sum(u * v, dim=1)
        ub  = self.user_bias(user_idx).squeeze(1)
        ib  = self.item_bias(item_idx).squeeze(1)
        return torch.clamp(self.global_mean + ub + ib + dot, 1.0, 5.0)


# ── load everything once (cached) ───────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(OUTPUT_DIR / "collaborative_config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    with open(DATA_DIR / "user_encoder.json", encoding="utf-8") as f:
        user_encoder = json.load(f)

    with open(DATA_DIR / "item_encoder.json", encoding="utf-8") as f:
        item_encoder = json.load(f)

    item_decoder = {v: k for k, v in item_encoder.items()}

    items_df  = pd.read_csv(DATA_DIR / "items.csv")
    title_col = "title" if "title" in items_df.columns else (
                "asin"  if "asin"  in items_df.columns else items_df.columns[0])
    idx_col   = "item_idx" if "item_idx" in items_df.columns else items_df.columns[0]
    item_title_map = dict(zip(items_df[idx_col], items_df[title_col]))

    train    = pd.read_csv(DATA_DIR / "train.csv")
    val      = pd.read_csv(DATA_DIR / "val.csv")
    test     = pd.read_csv(DATA_DIR / "test.csv")
    all_data = pd.concat([train, val, test], ignore_index=True)

    num_users = int(all_data["user_idx"].max() + 1)
    num_items = int(max(all_data["item_idx"].max(), items_df[idx_col].max()) + 1)

    model = MatrixFactorization(
        num_users    = num_users,
        num_items    = num_items,
        embedding_dim= int(cfg["embedding_dim"]),
        global_mean  = float(cfg["global_mean"]),
    ).to(device)

    model.load_state_dict(
        torch.load(OUTPUT_DIR / "matrix_factorization_model.pt", map_location=device)
    )
    model.eval()

    user_seen = (
        all_data.groupby("user_idx")["item_idx"]
        .apply(set).to_dict()
    )

    all_item_indices = np.arange(num_items)

    return (model, device, user_encoder, item_decoder,
            item_title_map, user_seen, all_item_indices)


(model, device, user_encoder, item_decoder,
 item_title_map, user_seen, all_item_indices) = load_model_and_data()

valid_user_ids = sorted(user_encoder.keys())

# ── UI ───────────────────────────────────────────────────────────────────────
st.title(" Amazon Electronics Recommender")
st.markdown(
    "**Collaborative Filtering · Matrix Factorization**  \n"
    "HitRate@10 = `0.4046` &nbsp;|&nbsp; NDCG@10 = `0.2501`"
)
st.divider()

# sidebar
with st.sidebar:
    st.header(" Settings")
    n_recs = st.slider("Number of Recommendations", 1, 20, 10)
    exclude_seen = st.checkbox("Exclude already-rated items", value=True)

    st.divider()
    st.subheader(" Model Metrics")
    st.dataframe(
        pd.DataFrame({
            "Metric":  ["HitRate@10", "Precision@10", "MRR@10", "NDCG@10"],
            "Score":   [0.4046, 0.4046, 0.2024, 0.2501],
        }),
        hide_index=True,
        use_container_width=True,
    )

# main input
user_id_input = st.text_input(
    "Enter User ID",
    placeholder="e.g. A1BCDXYZ...",
    help="Enter a valid Amazon User ID from the dataset"
)

st.markdown("**Try an example user:**")
example_cols = st.columns(min(5, len(valid_user_ids[:5])))
for col, uid in zip(example_cols, valid_user_ids[:5]):
    if col.button(uid[:10] + "…", key=uid):
        user_id_input = uid

if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
    if not user_id_input.strip():
        st.warning("Please enter a User ID.")
    elif user_id_input.strip() not in user_encoder:
        st.error(
            f"User ID **{user_id_input}** not found.  \n"
            f"Example IDs: `{'`, `'.join(valid_user_ids[:3])}`"
        )
    else:
        with st.spinner("Generating recommendations..."):
            user_idx  = user_encoder[user_id_input.strip()]
            seen      = user_seen.get(user_idx, set()) if exclude_seen else set()
            candidates= np.array([i for i in all_item_indices if i not in seen])
            if len(candidates) == 0:
                candidates = all_item_indices.copy()

            u_tensor = torch.full((len(candidates),), user_idx, dtype=torch.long, device=device)
            i_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

            with torch.no_grad():
                scores = model(u_tensor, i_tensor).cpu().numpy()

            top_idx    = np.argsort(scores)[::-1][:n_recs]
            top_items  = candidates[top_idx]
            top_scores = scores[top_idx]

            rows = []
            for rank, (item_idx, score) in enumerate(zip(top_items, top_scores), 1):
                orig_id = item_decoder.get(int(item_idx), str(item_idx))
                title   = item_title_map.get(int(item_idx), orig_id)
                rows.append({
                    "Rank": rank,
                    "Item ID": orig_id,
                    "Title / Name": str(title),
                    " Predicted Rating": round(float(score), 2),
                })

            result_df = pd.DataFrame(rows)

        st.success(f"Top {n_recs} recommendations for `{user_id_input}`")
        st.dataframe(result_df, hide_index=True, use_container_width=
