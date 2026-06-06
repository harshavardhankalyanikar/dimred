# 📊 PCA Dimensionality Reduction — Credit Card Dataset

## Overview
This project applies **Principal Component Analysis (PCA)** to the CC General credit card dataset (8,950 customers, 17 features) to reduce dimensionality, visualize customer behaviour, and identify customer segments via KMeans clustering.

---

## 📁 Project Structure
```
PCA_Project/
├── notebooks/
│   └── CC_general_pca.ipynb    # Full analysis notebook
├── data/
│   └── CC_GENERAL.csv          # Dataset (8950 samples, 17 features)
├── models/
│   ├── pca_metrics.json        # PCA results & KMeans metrics
│   └── pca_reduced.csv         # PCA-reduced data + cluster labels
├── app.py                      # Streamlit web application
├── style.css                   # Custom styling
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🔬 Key Results

| Metric | Value |
|--------|-------|
| Original Features | 17 |
| Components for 90% Variance | **10** |
| Components for 95% Variance | **12** |
| PC1 Explained Variance | **27.30%** |
| PC2 Explained Variance | **20.31%** |
| Best KMeans Clusters (k) | **3** |
| Best Silhouette Score | **0.2586** |

### Cluster Distribution
| Cluster | Customers |
|---------|-----------|
| 0 | 6,107 |
| 1 | 1,601 |
| 2 | 1,242 |

---

## 🚀 Running the App
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📓 Running the Notebook
```bash
jupyter notebook notebooks/CC_general_pca.ipynb
```

---

## 📌 Method
1. **Preprocessing** — Median imputation for missing values, StandardScaler normalization
2. **PCA** — Full decomposition → select components explaining ≥90% variance (10 PCs)
3. **Visualization** — Scree plot, cumulative variance, biplot, 2D/3D scatter
4. **Clustering** — KMeans on PCA-reduced space, silhouette analysis (k=2–8)
