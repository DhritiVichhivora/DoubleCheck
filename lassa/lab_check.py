"""Cross-check the hand-written PCA, kNN and k-means against scikit-learn (the same way the FOML lab notebooks compare them)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from engine import build_entities
from names import Vectoriser
from lab_models import LabPCA, LabKNN, LabKMeans

df = pd.read_parquet(os.path.join(ROOT, "data", "indian_medicines.parquet"))
names, *_ = build_entities(df)
X = Vectoriser().fit(names).transform(names)

mine = LabPCA(20).fit(X)
sk = PCA(n_components=20, svd_solver="full").fit(X)
print("PCA: biggest difference in explained-variance ratio:", float(np.abs(mine.explained_variance_ratio_[:20] - sk.explained_variance_ratio_).max()))
cos = [abs(float(mine.components_[i] @ sk.components_[i])) for i in range(20)]
print("PCA: smallest |cosine| between my component and sklearn's (1.0 = same direction):", round(min(cos), 4))

Z = mine.transform(X)
knn = LabKNN(Z)
nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(Z)
same = 0
for q in range(0, 600, 12):
    a, _ = knn.query(Z[q], k=6)
    b = nn.kneighbors(Z[q:q + 1], return_distance=False)[0]
    same += set(a) == set(b)
print("kNN: queries (of 50) where my 6 nearest = sklearn's 6 nearest:", same)

km = LabKMeans(40, seed=0).fit(Z)
best = min(KMeans(40, n_init=1, random_state=s).fit(Z).inertia_ for s in range(3))
print("k-means inertia: mine %.1f | sklearn (best of 3) %.1f" % (km.inertia_, best))
