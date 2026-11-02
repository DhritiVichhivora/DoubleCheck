"""Build the final DoubleCheck model and every number shown in the app. Run:  python lassa/build_model.py"""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from engine import Engine
from names import sound_key
import validate as V

K, W = 256, 0.5
t0 = time.time()
df = pd.read_parquet(os.path.join(ROOT, "data", "indian_medicines.parquet"))
e = Engine(k=K, w=W).fit(df); e._cache = {}
N = len(e.names)
print("entities", N, "| brands", e.kinds.count("brand"), "| generics", e.kinds.count("generic"), "|", round(time.time() - t0), "s")

# ---- similarity of every pair of names (spelling + sound, PCA space)
cos = np.clip(e.knn.Z @ e.knn.Z.T, 0, 1).astype(np.float32)
ed = np.stack([e.edit.similarity(sound_key(n)) for n in e.names]).astype(np.float32)
S = W * cos + (1 - W) * ed; S = (S + S.T) / 2; np.fill_diagonal(S, 0)
iu = np.triu_indices(N, 1); vals = S[iu]
cut = {"watch": float(np.quantile(vals, .99)), "high": float(np.quantile(vals, .999)), "very_high": float(np.quantile(vals, .9999))}
print("cut-offs (top 1%, 0.1%, 0.01% of", len(vals), "pairs):", {k: round(v, 3) for k, v in cut.items()})

def band(s):
    return "Very high" if s >= cut["very_high"] else "High" if s >= cut["high"] else "Watch" if s >= cut["watch"] else "Low"

# ---- riskiest different-drug pairs
sel = np.where(vals >= cut["high"])[0]
rows = []
for p in sel:
    i, j = int(iu[0][p]), int(iu[1][p])
    if e.kinds[i] == "generic" and e.kinds[j] == "generic":
        continue
    A, B = e.ing_ids[i], e.ing_ids[j]
    ov = len(A & B) / max(min(len(A), len(B)), 1)
    rows.append((float(vals[p]), i, j, ov))
diff = sorted([r for r in rows if r[3] < 0.5], reverse=True)
print("pairs in top 0.1% (brand involved):", len(rows), "| of which different drugs:", len(diff))
risky = [dict(sim=round(s, 3), a=e.names[i], b=e.names[j], a_ing=e.ings[i], b_ing=e.ings[j], a_kind=e.kinds[i], b_kind=e.kinds[j],
              band=band(s)) for s, i, j, ov in diff[:300]]
json.dump(risky, open(os.path.join(ROOT, "docs", "risky_pairs.json"), "w"), indent=1)

# ---- validation A: misspelling / mis-hearing test (new random names, never used for choosing k or w)
test = V.corruption_set(e.names, 500, seed=2)
accA = V.accuracy(e, test, [K])
# ---- validation B: official ISMP pairs
pairs = V.ismp_pairs(e, os.path.join(ROOT, "data", "ismp_pairs.json"))
ranks = {}
for a, b in pairs:
    for x, y in ((a, b), (b, a)):
        sc = V.rank_methods(e, x, [K])
        for m, s in sc.items():
            s = s.copy(); s[e.index[x]] = -9
            ranks.setdefault(str(m), []).append(int((s > s[e.index[y]]).sum()))
ranks = {m: np.array(v) for m, v in ranks.items()}
rng = np.random.default_rng(0); npairs = len(pairs)
def boot(r, t, B=2000):
    r2 = r.reshape(npairs, 2)
    out = [(r2[rng.integers(0, npairs, npairs)] < t).mean() for _ in range(B)]
    return float((r < t).mean()), float(np.quantile(out, .025)), float(np.quantile(out, .975))
names_map = {"raw n-gram cosine (3000 dims)": "Spelling n-grams, no PCA (3,000 numbers)", "edit distance only": "Edit distance on sound key only",
             str(("pca", K)): "PCA only (256 numbers)", str(("hybrid", K)): "PCA + edit distance (shipped)"}
ismp = {names_map[m]: {"top10": boot(r, 10), "top20": boot(r, 20), "top50": boot(r, 50)} for m, r in ranks.items()}
ismp_sim = np.array([S[e.index[a], e.index[b]] for a, b in pairs])
cover = {"watch": float((ismp_sim >= cut["watch"]).mean()), "high": float((ismp_sim >= cut["high"]).mean()), "very_high": float((ismp_sim >= cut["very_high"]).mean())}
pct = [float((vals < s).mean()) for s in ismp_sim]
corr = {names_map.get(str(m), str(m)): d for m, d in accA.items()}
print("corruption:", {m: {k: round(v, 3) for k, v in d.items()} for m, d in corr.items()})
print("ISMP pairs:", npairs, {m: [round(x, 3) for x in d["top10"]] for m, d in ismp.items()}, "chance top10 =", round(10 / N, 4))
print("ISMP pairs inside top 1% / 0.1% / 0.01% of all pairs:", {k: round(v, 3) for k, v in cover.items()}, "| median percentile", round(float(np.median(pct)), 4))

# ---- PCA facts for the app
ratio = e.pca.explained_variance_ratio_
fn = e.vec.feature_names
def label(g):
    return ("sound " if g[0] == "p" else "spelling ") + g[1:]
load = []
for c in range(8):
    v = e.pca.eigenvectors_[:, c]
    up, dn = np.argsort(-v)[:6], np.argsort(v)[:6]
    load.append({"pc": c + 1, "var": float(ratio[c]), "plus": [label(fn[i]) for i in up], "minus": [label(fn[i]) for i in dn]})
# ---- k-means zones
Zs = e.Z[:, :2]
zone_hi = np.zeros(e.km.k)
for s, i, j, ov in diff:
    zone_hi[e.km.labels_[i]] += 1; zone_hi[e.km.labels_[j]] += 1
zones = []
for c in range(e.km.k):
    m = np.where(e.km.labels_ == c)[0]
    d = ((e.Z[m] - e.km.centres_[c]) ** 2).sum(1)
    zones.append({"id": c, "size": int(len(m)), "examples": [e.names[m[x]] for x in np.argsort(d)[:6]], "risky_pairs": int(zone_hi[c])})
info = dict(n_entities=N, n_brands=e.kinds.count("brand"), n_generics=e.kinds.count("generic"), n_pairs=int(len(vals)), k=K, w=W,
            var_kept=float(ratio[:K].sum()), var_2=float(ratio[:2].sum()), var_cum=[float(x) for x in np.cumsum(ratio)[:400]],
            ratio=[float(x) for x in ratio[:400]], cut=cut, loadings=load, zones=zones, n_ismp=npairs, ismp=ismp, ismp_cover=cover,
            ismp_median_pct=float(np.median(pct)), corruption=corr, risky_total=len(rows), risky_diff=len(diff), chance_top10=10 / N,
            products=int(len(df)), kmeans_inertia=e.km.inertia_)
json.dump(info, open(os.path.join(ROOT, "docs", "validation.json"), "w"), indent=1)
e.save(os.path.join(ROOT, "models", "lassa_v1"))
np.save(os.path.join(ROOT, "docs", "map_xy.npy"), Zs.astype(np.float32))
print("done in", round(time.time() - t0), "s")
