"""DoubleCheck engine: find medicine names that look or sound like a given name, and say how dangerous the confusion is."""
import pickle
import numpy as np
import pandas as pd
from names import brand_stem, parse_ingredients, sound_key, Vectoriser, EditIndex
from lab_models import LabPCA, LabKNN, LabKMeans


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def same_drug(a, b):
    """Two ingredient names count as the same drug if they sound almost the same (amoxycillin / amoxicillin)."""
    ka, kb = sound_key(a), sound_key(b)
    return 1 - lev(ka, kb) / max(len(ka), len(kb), 1) >= 0.85


def ingredient_overlap(A, B):
    """Share of the smaller ingredient list that also appears in the other list (0 = completely different drugs)."""
    if not A or not B:
        return 0.0
    small, big = (A, B) if len(A) <= len(B) else (B, A)
    hit = sum(any(same_drug(a, b) for b in big) for a in small)
    return hit / len(small)


def canonical_ids(ingredients):
    """Group spellings of the same drug (amoxycillin / amoxicillin) into one id: sound-key edit similarity >= 0.85."""
    ings = sorted(set(ingredients))
    keys = [sound_key(x) for x in ings]
    ed = EditIndex(keys)
    parent = list(range(len(ings)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, k in enumerate(keys):
        sim = ed.similarity(k)
        for j in np.where(sim >= 0.85)[0]:
            if j != i:
                parent[find(int(j))] = find(i)
    return {ing: find(i) for i, ing in enumerate(ings)}, ed, ings


def build_entities(df):
    brands = {}
    for name, comp in zip(df["name"], df["composition"]):
        st = brand_stem(name)
        if len(st) < 3:
            continue
        b = brands.setdefault(st, {"ing": set(), "products": []})
        b["ing"].update(parse_ingredients(comp))
        if len(b["products"]) < 6:
            b["products"].append(str(name))
    names, kinds, ings, prods = [], [], [], []
    for st, b in brands.items():
        names.append(st); kinds.append("brand"); ings.append(sorted(b["ing"])); prods.append(b["products"])
    generics = sorted({g for b in brands.values() for g in b["ing"] if " " not in g and len(g) >= 4} - set(names))
    for g in generics:
        names.append(g); kinds.append("generic"); ings.append([g]); prods.append([])
    return names, kinds, ings, prods


class Engine:
    def __init__(self, k=64, w=0.5):
        self.k, self.w = k, w

    def fit(self, df, kmeans_k=40):
        self.names, self.kinds, self.ings, self.prods = build_entities(df)
        self.canon, self.canon_edit, self.canon_names = canonical_ids([g for L in self.ings for g in L])
        self.ing_ids = [set(self.canon[g] for g in L) for L in self.ings]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.vec = Vectoriser().fit(self.names)
        self.X = self.vec.transform(self.names)
        self.pca = LabPCA(self.k).fit(self.X)
        self.Z = self.pca.transform(self.X)
        self.knn = LabKNN(self.Z)
        self.edit = EditIndex([sound_key(n) for n in self.names])
        self.km = LabKMeans(kmeans_k).fit(self.Z)
        return self

    def scores(self, name):
        stem = brand_stem(name) or "".join(c for c in name.lower() if c.isalpha())
        z = self.pca.transform(self.vec.transform([stem]))[0]
        cos = np.clip(self.knn.similarity(z), 0, 1)
        ed = self.edit.similarity(sound_key(stem))
        return stem, cos, ed, self.w * cos + (1 - self.w) * ed

    def neighbours(self, name, topn=10, ing=None, kinds=("brand", "generic")):
        """Most look/sound-alike entries, each with its ingredient overlap and a danger label."""
        stem, cos, ed, sim = self.scores(name)
        me = self.index.get(stem)
        if me is not None:
            sim = sim.copy(); sim[me] = -1
            ing = ing if ing is not None else self.ings[me]
        order = np.argsort(-sim)
        out = []
        for j in order:
            if self.kinds[j] not in kinds:
                continue
            ov = self.overlap(ing, j) if ing else None
            out.append(dict(name=self.names[j], kind=self.kinds[j], sim=float(sim[j]), cos=float(cos[j]), edit=float(ed[j]),
                            ingredients=self.ings[j], overlap=ov, products=self.prods[j], idx=int(j)))
            if len(out) == topn:
                break
        return stem, out

    def ids_of(self, ing):
        """Drug ids for a typed list of ingredient names (unknown spellings are matched to the closest known one)."""
        out = set()
        for g in ing:
            if g in self.canon:
                out.add(self.canon[g]); continue
            sim = self.canon_edit.similarity(sound_key(g))
            j = int(sim.argmax())
            out.add(self.canon[self.canon_names[j]] if sim[j] >= 0.85 else ("new", g))
        return out

    def overlap(self, ing, j):
        A, B = self.ids_of(ing), self.ing_ids[j]
        return len(A & B) / max(min(len(A), len(B)), 1) if A and B else 0.0

    def slim(self):
        """Drop the big training matrix so the saved file is small."""
        self.X = None
        n = self.k
        self.pca.eigenvectors_ = self.pca.eigenvectors_[:, :n]
        return self

    def save(self, path):
        """Save as plain arrays (.npz) plus a JSON file, so it loads on any numpy version."""
        import json
        np.savez_compressed(path + ".npz", Z=self.Z.astype(np.float32), idf=self.vec.idf_, pca_mean=self.pca.mean_,
                            pca_vecs=self.pca.eigenvectors_[:, : self.k].astype(np.float32), km_centres=self.km.centres_.astype(np.float32),
                            km_labels=self.km.labels_)
        json.dump(dict(names=self.names, kinds=self.kinds, ings=self.ings, prods=self.prods, vocab=self.vec.feature_names,
                       canon=self.canon, canon_names=self.canon_names, k=self.k, w=self.w, km_k=int(self.km.k), inertia=float(self.km.inertia_)),
                  open(path + ".json", "w"))

    @classmethod
    def load(cls, path):
        import json
        a, j = np.load(path + ".npz"), json.load(open(path + ".json"))
        e = cls(k=j["k"], w=j["w"])
        e.names, e.kinds, e.ings, e.prods = j["names"], j["kinds"], j["ings"], j["prods"]
        e.index = {n: i for i, n in enumerate(e.names)}
        e.vec = Vectoriser(); e.vec.vocab_ = {g: i for i, g in enumerate(j["vocab"])}; e.vec.idf_ = a["idf"]
        e.pca = LabPCA(e.k); e.pca.mean_ = a["pca_mean"]; e.pca.eigenvectors_ = a["pca_vecs"].astype(np.float64)
        e.Z = a["Z"]; e.knn = LabKNN(e.Z)
        e.edit = EditIndex([sound_key(n) for n in e.names])
        e.canon, e.canon_names = j["canon"], j["canon_names"]
        e.canon_edit = EditIndex([sound_key(x) for x in e.canon_names])
        e.ing_ids = [set(e.canon[g] for g in L) for L in e.ings]
        e.km = LabKMeans(j["km_k"]); e.km.centres_, e.km.labels_, e.km.inertia_ = a["km_centres"], a["km_labels"], j["inertia"]
        return e
