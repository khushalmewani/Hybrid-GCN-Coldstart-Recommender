# Hybrid GCN Collaborative-Semantic Recommender

A PyTorch implementation of a Graph Convolutional Network (GCN) for movie recommendations that addresses the cold-start problem by combining collaborative filtering with semantic embeddings from BERT and self-supervised contrastive learning.

---

## 📋 Three-Line Description

**Technical:** A hybrid recommendation system combining LightGCN graph collaborative filtering with BERT semantic embeddings, trained using BPR loss and self-supervised graph learning (SGL) with InfoNCE contrastive loss to handle cold-start items.

**Simple:** This system recommends movies by learning from user-item interactions through a graph neural network, while also understanding movie content through text embeddings, enabling recommendations even for new movies with no user history.

**Academic:** Implementation of a heterogeneous graph neural network architecture that integrates content-based semantic representations with collaborative filtering signals, employing data augmentation via edge dropout and contrastive learning to improve generalization to cold-start scenarios.

---

## 🎯 Key Features

- **Hybrid Architecture**: Combines graph-based collaborative filtering (GCN) with content-based semantic embeddings (BERT)
- **Cold-Start Handling**: Temporal data splitting to simulate and evaluate zero-shot recommendation performance on new items
- **Self-Supervised Learning**: Graph contrastive learning (SGL) with edge dropout augmentation
- **LightGCN Backbone**: Simplified GCN architecture with symmetric normalization and layer-wise aggregation
- **Scalable Design**: Efficient batch training with BPR loss for implicit feedback

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid GCN Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Embeddings (Learnable)    Item Text → BERT → Linear   │
│         │                                      │             │
│         └──────────────┬───────────────────────┘             │
│                        │                                     │
│                 Combined Embedding                           │
│                        │                                     │
│         ┌──────────────┴──────────────┐                     │
│         │  LightGCN Layer 1           │                     │
│         │  LightGCN Layer 2           │                     │
│         │  LightGCN Layer 3           │                     │
│         └──────────────┬──────────────┘                     │
│                        │                                     │
│              Layer-wise Mean Pooling                         │
│                        │                                     │
│         ┌──────────────┴──────────────┐                     │
│    User Embeddings           Item Embeddings                │
│         │                              │                     │
│         └──────────────┬───────────────┘                     │
│                        │                                     │
│              BPR Loss + SGL Loss                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**MovieLens-100K**
- 100,000 ratings from 943 users on 1,682 movies
- Temporal split: 80% warm items (training graph) / 20% cold items (zero-shot test)
- Items sorted by first appearance timestamp to simulate real-world deployment

The dataset is automatically downloaded on first run.

---

## 🔧 Installation

### Prerequisites
```bash
Python 3.8+
CUDA-compatible GPU (optional, but recommended)
```

### Dependencies
```bash
pip install torch torchvision
pip install torch-geometric
pip install sentence-transformers
pip install pandas scikit-learn
pip install requests
```

Or install all at once:
```bash
pip install torch torch-geometric sentence-transformers pandas scikit-learn requests
```

---

## 🚀 Usage

### Quick Start

1. **Clone or download the notebook**
   ```bash
   jupyter notebook hybrid_gcn_collaborative_semantic_recommender.ipynb
   ```

2. **Run all cells sequentially**
   - The dataset will download automatically
   - BERT model (`all-MiniLM-L6-v2`) downloads on first use
   - Training takes ~5-15 minutes on GPU, longer on CPU

### Running from Command Line

If you prefer a Python script, convert the notebook:
```bash
jupyter nbconvert --to script hybrid_gcn_collaborative_semantic_recommender.ipynb
python hybrid_gcn_collaborative_semantic_recommender.py
```

---

## ⚙️ Configuration

Modify the `Config` class to adjust hyperparameters:

```python
class Config:
    DATA_PATH = './ml-100k'      # Dataset location
    SPLIT_RATIO = 0.8            # Train/test split (0.8 = 80% warm, 20% cold)
    BERT_DIM = 384               # BERT embedding dimension
    EMBED_DIM = 64               # GCN latent dimension
    N_LAYERS = 3                 # Number of GCN layers
    DROPOUT = 0.1                # Dropout rate during training
    LR = 1e-3                    # Learning rate
    WEIGHT_DECAY = 1e-4          # L2 regularization
    EPOCHS = 50                  # Training epochs
    BATCH_SIZE = 2048            # Batch size for sampling
    SSL_REG = 0.1                # Weight for contrastive loss
    SSL_TEMP = 0.2               # Temperature for InfoNCE
```

---

## 📐 Model Components

### 1. **LightGCN Convolution Layer**
Simplified graph convolution with symmetric normalization:
```
h^(l+1) = D^(-1/2) A D^(-1/2) h^(l)
```
Where `A` is the adjacency matrix and `D` is the degree matrix.

### 2. **BERT Semantic Encoder**
- Model: `all-MiniLM-L6-v2` (384-dim embeddings)
- Encodes movie titles into semantic vectors
- Projects to GCN embedding space via learnable linear layer

### 3. **Loss Functions**

**BPR (Bayesian Personalized Ranking) Loss:**
```
L_BPR = -log σ(s_pos - s_neg)
```
Optimizes ranking of positive items over negative samples.

**InfoNCE Contrastive Loss:**
```
L_SSL = -log(exp(z_i · z_j / τ) / Σ_k exp(z_i · z_k / τ))
```
Learns robust representations via data augmentation (edge dropout).

**Total Loss:**
```
L_total = L_BPR + λ * L_SSL
```

---

## 🧪 Training Process

1. **Graph Construction**: Build bipartite user-item graph from warm item interactions
2. **Forward Pass**: Propagate embeddings through GCN layers with dropout
3. **BPR Sampling**: Sample user-positive-negative triplets
4. **Augmentation**: Create two graph views via edge dropout (10%)
5. **Contrastive Learning**: Maximize agreement between augmented views
6. **Optimization**: Update parameters via Adam optimizer

---

## 📈 Expected Results

- **Training Loss**: Should decrease from ~0.7 to ~0.3-0.4 over 50 epochs
- **Cold-Start Performance**: Model should generate meaningful embeddings for cold items using BERT features
- **Computational Cost**: ~2-5 seconds per epoch on GPU, ~20-30 seconds on CPU

---

## 🔬 Technical Details

### Cold-Start Strategy
- **Problem**: New items have no interaction history
- **Solution**: Initialize item embeddings from BERT text encodings
- **Evaluation**: Temporal split ensures test items are unseen during graph construction

### Self-Supervised Learning (SGL)
- **Augmentation**: Random edge dropout (10%) creates correlated graph views
- **Objective**: Maximize mutual information between user embeddings from different views
- **Benefit**: Improves robustness and generalization beyond observed interactions

### Graph Neural Network
- **Type**: Spectral-based GCN (LightGCN variant)
- **Aggregation**: Mean pooling across all layer outputs (layer 0 to layer N)
- **Normalization**: Symmetric degree normalization prevents over-smoothing

---

## 📂 Project Structure

```
.
├── hybrid_gcn_collaborative_semantic_recommender.ipynb  # Main notebook
├── README.md                                             # This file
├── ml-100k/                                              # Downloaded dataset
│   ├── u.data                                            # User-item ratings
│   ├── u.item                                            # Item metadata
│   └── ...
└── models/                                               # (Optional) Saved checkpoints
```

---

## 🛠️ Extending the Code

### Add Evaluation Metrics
```python
def evaluate(model, dataset, k=10):
    model.eval()
    with torch.no_grad():
        users_emb, items_emb = model(edge_index, bert_feats)
        # Compute Recall@K, NDCG@K for cold items
        # ...
```

### Save/Load Model
```python
# Save
torch.save(model.state_dict(), 'hybrid_gcn_model.pt')

# Load
model.load_state_dict(torch.load('hybrid_gcn_model.pt'))
```

### Use Different BERT Models
```python
# Replace in dataset class
bert = SentenceTransformer('bert-base-nli-mean-tokens')  # 768-dim
# Update cfg.BERT_DIM = 768
```

---

## 📚 References

### Papers
1. **LightGCN**: He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" (SIGIR 2020)
2. **BPR**: Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
3. **SGL**: Wu et al. "Self-supervised Graph Learning for Recommendation" (SIGIR 2021)
4. **Sentence-BERT**: Reimers & Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (EMNLP 2019)

### Libraries
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Sentence Transformers](https://www.sbert.net/)

---

## ⚠️ Known Limitations

1. **Sampling Strategy**: Current implementation uses random negative sampling. Production systems should use hard negative mining or popularity-based sampling.

2. **No Validation Set**: The code doesn't include validation split for hyperparameter tuning. Consider adding k-fold cross-validation.

3. **Memory Constraints**: Full-batch graph operations may fail on very large datasets. Consider mini-batch subgraph sampling for scalability.

4. **Evaluation**: No metrics (Recall@K, NDCG@K, Hit Rate) are computed. Add evaluation functions to assess cold-start performance.

---

## 🤝 Contributing

Suggestions for improvement:
- [ ] Add comprehensive evaluation metrics
- [ ] Implement validation set and early stopping
- [ ] Add model checkpointing
- [ ] Support for additional datasets (Amazon, Yelp)
- [ ] Hyperparameter tuning with Optuna/Ray Tune
- [ ] Visualization of embeddings (t-SNE, UMAP)
- [ ] Inference API for real-time recommendations

---

## 📝 License

This project is provided as-is for educational and research purposes. The MovieLens dataset is provided by GroupLens Research and has its own usage terms.

---

## 🙋 FAQ

**Q: Why does training take so long on CPU?**  
A: GCN operations on large graphs are computationally intensive. Use a GPU or reduce `N_LAYERS` and `EPOCHS`.

**Q: Can I use this for other datasets?**  
A: Yes, but you'll need to adapt the `MovieLensColdStartDataset` class to load your data format.

**Q: How do I get recommendations for a user?**  
A: Compute dot product between user embedding and all item embeddings, then rank by score:
```python
user_scores = torch.matmul(users_emb[user_id], items_emb.T)
top_k_items = torch.topk(user_scores, k=10).indices
```

**Q: Why use BERT for movie titles only?**  
A: This is a minimal example. You can extend to include genres, descriptions, tags, or even poster images (via vision transformers).

---

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue in the repository or reach out via email.
khushalmewani.work@gmail.com

---

**Happy Recommending! 🎬🍿**
