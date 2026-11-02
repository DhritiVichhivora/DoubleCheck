"""Turning medicine names into numbers: cleaning, a simple sound key, n-gram vectors, edit distance."""
import re
import numpy as np
import pandas as pd

FORMS = {"tablet", "tablets", "capsule", "capsules", "syrup", "injection", "drops", "cream", "gel", "ointment",
         "suspension", "solution", "lotion", "powder", "infusion", "inhaler", "rotacap", "respules", "sachet", "spray"}


def brand_stem(name):
    """'Losar-H 50 Tablet' -> 'losar'  (first alphabetic part of the product name)."""
    s = re.sub(r"\(.*?\)", " ", str(name).lower())
    m = re.match(r"\s*([a-z]+)", s)
    return m.group(1) if m else ""


def parse_ingredients(comp):
    """'Amoxycillin (500mg) + Clavulanic Acid (125mg)' -> ['amoxycillin', 'clavulanic acid']"""
    out = []
    for part in str(comp).split("+"):
        part = re.sub(r"\(.*?\)", " ", part.lower())
        part = re.sub(r"[^a-z ]", " ", part)
        part = " ".join(w for w in part.split() if w not in {"mg", "mcg", "iu", "ml", "gm"})
        if part:
            out.append(part)
    return out


def sound_key(s):
    """Rough 'how it sounds' form: ph->f, ck->k, soft c->s, y->i, doubled letters collapsed ..."""
    s = re.sub(r"[^a-z]", "", s.lower())
    for a, b in (("ph", "f"), ("ck", "k"), ("kh", "k"), ("th", "t"), ("sh", "s"), ("ee", "i"), ("oo", "u"), ("ou", "u")):
        s = s.replace(a, b)
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = s.replace("c", "k").replace("q", "k").replace("x", "ks").replace("z", "s").replace("y", "i").replace("w", "v")
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def ngrams(s, ns=(2, 3)):
    s = "^" + s + "$"
    return [s[i:i + n] for n in ns for i in range(len(s) - n + 1)]


class Vectoriser:
    """Character n-grams of the spelling and of the sound key, weighted by TF-IDF (idf = how rare the n-gram is)."""

    def __init__(self, per_channel=1500):
        self.per_channel = per_channel

    def _grams(self, stem):
        return [("s" + g) for g in ngrams(stem)] + [("p" + g) for g in ngrams(sound_key(stem))]

    def fit(self, stems):
        df = {}
        for st in stems:
            for g in set(self._grams(st)):
                df[g] = df.get(g, 0) + 1
        chosen = []
        for ch in ("s", "p"):
            items = sorted(((c, g) for g, c in df.items() if g[0] == ch), reverse=True)[: self.per_channel]
            chosen += [g for _, g in items]
        self.vocab_ = {g: i for i, g in enumerate(chosen)}
        n = len(stems)
        self.idf_ = np.array([np.log(n / (1 + df[g])) + 1 for g in chosen], dtype=np.float32)
        return self

    def transform(self, stems):
        X = np.zeros((len(stems), len(self.vocab_)), dtype=np.float32)
        for r, st in enumerate(stems):
            for g in self._grams(st):
                j = self.vocab_.get(g)
                if j is not None:
                    X[r, j] += 1
        X *= self.idf_
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        return X

    @property
    def feature_names(self):
        names = [None] * len(self.vocab_)
        for g, i in self.vocab_.items():
            names[i] = g
        return names


# ---------- edit distance (Levenshtein), written out; vectorised over many candidate names at once ----------
class EditIndex:
    def __init__(self, keys):
        self.keys = keys
        self.L = max(len(k) for k in keys)
        self.E = np.zeros((len(keys), self.L), dtype=np.uint8)
        for r, k in enumerate(keys):
            self.E[r, : len(k)] = [ord(c) for c in k]
        self.len = np.array([len(k) for k in keys])

    def distance(self, q):
        """Edit distance from string q to every key."""
        N, L = self.E.shape
        prev = np.tile(np.arange(L + 1), (N, 1))
        for j, ch in enumerate(q, start=1):
            cur = np.empty_like(prev)
            cur[:, 0] = j
            neq = (self.E != ord(ch)).astype(prev.dtype)
            for i in range(1, L + 1):
                cur[:, i] = np.minimum(np.minimum(prev[:, i] + 1, cur[:, i - 1] + 1), prev[:, i - 1] + neq[:, i - 1])
            prev = cur
        return prev[np.arange(N), self.len]

    def similarity(self, q):
        d = self.distance(q)
        return 1.0 - d / np.maximum(np.maximum(self.len, len(q)), 1)


def jaro_winkler(a, b):
    """Jaro-Winkler similarity: rewards names that share letters in order and especially the same beginning."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    rng = max(max(la, lb) // 2 - 1, 0)
    ma, mb = [False] * la, [False] * lb
    m = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - rng), min(lb, i + rng + 1)):
            if not mb[j] and b[j] == ca:
                ma[i] = mb[j] = True
                m += 1
                break
    if m == 0:
        return 0.0
    t, k = 0, 0
    for i in range(la):
        if ma[i]:
            while not mb[k]:
                k += 1
            t += a[i] != b[k]
            k += 1
    jaro = (m / la + m / lb + (m - t / 2) / m) / 3
    p = 0
    for x, y in zip(a, b):
        if x != y or p == 4:
            break
        p += 1
    return jaro + p * 0.1 * (1 - jaro)
