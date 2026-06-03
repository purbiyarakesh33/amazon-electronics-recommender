#  Amazon Electronics Recommender

A production-grade **hybrid recommender system** trained on the Amazon Electronics dataset, combining **Collaborative Filtering (Matrix Factorization)** and **Content-Based Filtering (Two Tower Neural Network)** into a unified pipeline — with rigorous experimentation to find the best-performing model.

 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/rakesh9773/amazon-electronics-recommender)**

---

##  Project Overview

This project was built as part of a machine learning portfolio, applying concepts from Andrew Ng's Machine Learning Specialization to construct an **industry-grade recommendation system** from scratch.

The full pipeline includes:

1. **Collaborative Filtering (CF)** — Matrix Factorization trained in PyTorch with GPU acceleration, mean normalization, cold-start filtering, temporal train/val/test splits, and grid search over hyperparameters with early stopping.
2. **Content-Based Filtering (CBF)** — Two Tower Neural Network trained via Triplet Loss on item metadata (category, brand, price, etc.), producing dense item embeddings.
3. **Hybrid Fusion** — Alpha-weighted score combination of CF and CBF predictions, with alpha tuned via grid search.

**Key finding:** After exhaustive alpha search, `alpha = 1.0` (pure CF) was optimal — the CBF model consistently reduced hybrid performance, making Collaborative Filtering the final selected model.

---

##  Model Performance

| Model | HitRate@10 | Precision@10 | MRR@10 | NDCG@10 |
|---|---|---|---|---|
| Content-Based Filtering (Two Tower) | 0.1968 | 0.1968 | 0.0689 | 0.0984 |
| **Collaborative Filtering (MF)** | **0.4046** | **0.4046** | **0.2024** | **0.2501** |
| Hybrid Fusion (α · CF + (1-α) · CBF) | 0.4046 | 0.4046 | 0.2024 | 0.2501 |

> **Best Model: Collaborative Filtering** — Hybrid fusion with `alpha = 1.0` matched pure CF, confirming that the CBF component did not add signal over CF on this dataset.

---

##  Architecture

```
Amazon Electronics Dataset
         │
         ▼
┌─────────────────────┐     ┌──────────────────────────┐
│  Collaborative      │     │  Content-Based Filtering  │
│  Filtering          │     │  (Two Tower Network)      │
│                     │     │                           │
│  Matrix             │     │  User Tower + Item Tower  │
│  Factorization      │     │  trained via Triplet Loss │
│  (PyTorch, GPU)     │     │  → Dense Item Vectors     │
└────────┬────────────┘     └────────────┬──────────────┘
         │                               │
         └──────────────┬────────────────┘
                        ▼
              Hybrid Fusion Layer
              α · CF + (1-α) · CBF
              (Grid search → α = 1.0)
                        │
                        ▼
               Final: Pure CF Model
```

---

##  Technical Highlights

- **GPU-accelerated training** on PyTorch (RTX 3050, CUDA)
- **Mean normalization** of user ratings before matrix factorization
- **Cold-start filtering** to exclude users/items with insufficient interactions
- **Temporal train/val/test splits** to prevent data leakage
- **Grid search** over all hyperparameters (learning rate, regularization, rank, etc.) — no arbitrary hardcoded values
- **Early stopping** based on validation loss
- **Triplet Loss** with hard negative mining for Two Tower CBF training
- **Alpha grid search** over hybrid fusion weight to empirically determine the best blend

---

##  How It Works

1. Enter an Amazon **User ID** from the dataset
2. The model scores all items using learned **user & item embeddings**
3. Top-N items ranked by **predicted rating** are returned

---

##  File Structure

```
amazon-electronics-recommender/
├── app.py                          # Streamlit application
├── requirements.txt
├── user_encoder.json               # User ID → index mapping
├── item_encoder.json               # Item ID → index mapping
├── items.csv                       # Item metadata
├── train.csv                       # Temporal training split
├── val.csv                         # Validation split
├── test.csv                        # Test split
└── collaborative_output/
    ├── matrix_factorization_model.pt   # Trained CF model weights
    └── collaborative_config.json       # Best hyperparameters from grid search
```

---

##  Tech Stack

| Component | Tools |
|---|---|
| Language | Python 3.13 |
| Deep Learning | PyTorch (CUDA) |
| Data Processing | Pandas, NumPy, SciPy (sparse matrices) |
| CF Model | Matrix Factorization (custom PyTorch) |
| CBF Model | Two Tower Neural Network + Triplet Loss |
| Evaluation | HitRate@K, Precision@K, MRR@K, NDCG@K |
| App | Streamlit |
| Deployment | Hugging Face Spaces |

---

##  Installation & Local Run

```bash
# Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

##  Dataset

**Amazon Electronics Reviews** — a widely used benchmark dataset for recommender systems research, containing user ratings for electronics products on Amazon.

---

##  Experiment Log

| Experiment | Result |
|---|---|
| CF alone | HitRate@10 = 0.4046  |
| CBF alone (Two Tower) | HitRate@10 = 0.1968 |
| Hybrid α=0.8 | < 0.4046 |
| Hybrid α=0.9 | < 0.4046 |
| Hybrid α=1.0 (pure CF) | HitRate@10 = 0.4046  |

> CBF item vectors did not generalize well enough to improve over CF, making pure Matrix Factorization the optimal choice.

---

##  Author

**Rakesh Purbiya**
[LinkedIn](https://www.linkedin.com/in/rakesh-purbiya-0b7091317/) ·
[Hugging Face](https://huggingface.co/spaces/rakesh9773/amazon-electronics-recommender)

---

##  If you found this useful, give it a star!
