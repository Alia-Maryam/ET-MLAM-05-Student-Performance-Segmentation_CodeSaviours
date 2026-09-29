# ET-MLAM-05-Student-Performance-Segmentation_CodeSaviours
# 🎓 Student Performance Segmentation using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Unsupervised%20ML-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An unsupervised machine learning project that groups students into performance segments using **K-Means clustering**, helping identify high achievers, average performers, and students who may need academic support.

## 🔍 Overview
Instead of relying on grades alone, this project clusters students by their scores and related features to reveal natural performance groups. These segments can support targeted teaching strategies and early intervention.

## 🎯 Objectives
- Explore and preprocess student performance data
- Scale features for distance-based clustering
- Find the optimal number of clusters (Elbow Method and Silhouette Score)
- Train K-Means and interpret each cluster
- Visualize the segments

## 📊 Dataset
| Property | Details |
|---|---|
| Records | `<add>` |
| Features | `<add>` (e.g., math score, reading score, writing score) |
| Missing values | `<add>` |

## 🛠 Tech Stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## ⚙ Installation
```bash
git clone https://github.com/<your-username>/student-performance-kmeans.git
cd student-performance-kmeans
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

## 💻 Code

### Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
```

### Load and Explore
```python
df = pd.read_csv("student_performance.csv")  # update filename
print(df.shape)
print(df.isnull().sum())
df.describe()
```

### Feature Scaling
```python
features = ["math_score", "reading_score", "writing_score"]  # update columns
X = df[features]
X_scaled = StandardScaler().fit_transform(X)
```

### Elbow Method and Silhouette Score
```python
inertia, sil = [], []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, labels))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(K, inertia, marker="o"); ax[0].set_title("Elbow Method")
ax[1].plot(K, sil, marker="o");     ax[1].set_title("Silhouette Score")
plt.show()
```

### Train Final Model
```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # set best k
df["cluster"] = kmeans.fit_predict(X_scaled)
print("Silhouette:", silhouette_score(X_scaled, df["cluster"]))
```

### Cluster Profiling
```python
df.groupby("cluster")[features].mean().round(2)
```

### Visualization (PCA)
```python
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=df["cluster"], palette="Set2")
plt.title("Student Segments (PCA)")
plt.show()
```

## 📈 Results
| Cluster | Profile | Size |
|---|---|---|
| 0 | `<add>` | `<add>` |
| 1 | `<add>` | `<add>` |
| 2 | `<add>` | `<add>` |

Optimal k: `<add>` · Silhouette Score: `<add>`

## 💡 Insights
- `<add key findings from your notebook>`

## 📁 Project Structure
```
├── Project_5_K_Means_Clustering_Student_Performance_Segmentation.ipynb
├── data/student_performance.csv
└── README.md
```

## 🚀 Future Improvements
- Compare with Hierarchical Clustering and DBSCAN
- Add more features (attendance, study time)
- Build an interactive dashboard with Streamlit
- Deploy a tool that assigns new students to a segment

## 👤 Author
Alia Maryam
