"""Validation: (A) can the engine find a name from a misspelt / mis-heard version, (B) does it find the pairs on the ISMP list."""
import os, sys, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from engine import Engine
from names import brand_stem, sound_key

SIMILAR = {"a": "e", "e": "a", "i": "l", "l": "i", "m": "n", "n": "m", "c": "k", "k": "c", "b": "d", "d": "b", "o": "a", "u": "v", "v": "u", "s": "z", "z": "s", "f": "p", "p": "f", "t": "d", "g": "j", "j": "g"}


def corrupt(s, rng):
    kind = rng.integers(4)
    i = int(rng.integers(1, len(s) - 1))
    if kind == 0:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:] if i + 2 <= len(s) else s[:i - 1] + s[i] + s[i - 1] + s[i + 1:]
    if kind == 1:
        return s[:i] + s[i + 1:]
    if kind == 2:
        return s[:i] + SIMILAR.get(s[i], "x") + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def corruption_set(names, n, seed):
    rng = np.random.default_rng(seed)
    cand = [i for i, nm in enumerate(names) if len(nm) >= 5]
    pick = rng.choice(cand, n, replace=False)
    return [(int(i), corrupt(names[i], rng)) for i in pick]


def rank_methods(e, q, ks):
    stem = q
    x = e.vec.transform([stem])
    out = {}
    cos_raw = np.clip((e.X @ x[0]), 0, 1)
    out["raw n-gram cosine (3000 dims)"] = cos_raw
    ed = e.edit.similarity(sound_key(stem))
    out["edit distance only"] = ed
    Zfull = e.pca.transform(x, n=max(ks))
    for k in ks:
        Zk = e.pca.transform(e.X, n=k) if k not in e._cache else e._cache[k]
        e._cache[k] = Zk
        zq = Zfull[0, :k]
        Zn = Zk / (np.linalg.norm(Zk, axis=1, keepdims=True) + 1e-12)
        c = np.clip(Zn @ (zq / (np.linalg.norm(zq) + 1e-12)), 0, 1)
        out[("pca", k)] = c
        out[("hybrid", k)] = 0.5 * c + 0.5 * ed
    return out


def accuracy(e, qs, ks, topk=(1, 5)):
    res = {}
    for idx, q in qs:
        sc = rank_methods(e, q, ks)
        for m, s in sc.items():
            r = int((s > s[idx]).sum())
            for t in topk:
                res.setdefault(m, {}).setdefault(t, []).append(r < t)
    return {m: {t: float(np.mean(v)) for t, v in d.items()} for m, d in res.items()}


# ---- ISMP pairs
def clean_ismp(nm):
    s = re.sub(r"\(.*?\)", " ", nm.lower())
    m = re.match(r"\s*([a-z]+)", s)
    return m.group(1) if m else ""


def ismp_pairs(e, path):
    raw = json.load(open(path))
    seen, pairs = set(), []
    for a, b in raw:
        ca, cb = clean_ismp(a), clean_ismp(b)
        if ca in e.index and cb in e.index and ca != cb:
            key = tuple(sorted((ca, cb)))
            if key not in seen:
                seen.add(key); pairs.append(key)
    return pairs


def ismp_recall(e, pairs, ks, topn=(5, 10, 20), w=0.5):
    res = {}
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):
            sc = rank_methods(e, x, ks)
            for m, s in sc.items():
                s = s.copy(); s[e.index[x]] = -9
                r = int((s > s[e.index[y]]).sum())
                for t in topn:
                    res.setdefault(m, {}).setdefault(t, []).append(r < t)
    return {m: {t: float(np.mean(v)) for t, v in d.items()} for m, d in res.items()}


# Run through build_model.py, which uses these functions and writes docs/validation.json.
