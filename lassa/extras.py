"""Two add-ons built on the same engine: a safer-name suggester (for drug companies) and a shelf planner (for pharmacies).
Nothing here changes the fitted model; it only calls Engine.scores / Engine.overlap and the k-means zone labels."""
import re

import numpy as np

from engine import lev
from names import brand_stem

VOWELS = "aeiou"
ENDINGS = ["ax", "ex", "ix", "ol", "on", "an", "en", "in", "il", "ora", "iva", "ene", "ium", "ova", "ada", "ela",
           "ine", "ide", "ase", "ar", "ir", "oz", "ot", "et", "ura", "ona", "ova", "elo"]


def _ok(c):
    """Rough 'can a person say it' test: has a vowel, no run of four consonants, no letter three times in a row."""
    return bool(re.search("[aeiou]", c)) and not re.search("[^aeiou]{4}", c) and not re.search(r"(.)\1\1", c)


def candidates(stem):
    """Small edits of a proposed name. Returns {candidate: what_changed}."""
    out = {}

    def add(c, why):
        c = re.sub("[^a-z]", "", c)
        if 4 <= len(c) <= min(10, len(stem) + 2) and c != stem and c not in out and _ok(c):
            out[c] = why

    for cut in (1, 2, 3):                                  # swap the ending
        if len(stem) - cut >= 3:
            for en in ENDINGS:
                if stem[:-cut][-1] in VOWELS and en[0] in VOWELS:     # avoids joins like 'oova'
                    continue
                add(stem[:-cut] + en, "new ending '%s'" % en)
    for i, ch in enumerate(stem):                          # change one vowel
        if ch in VOWELS:
            for v in VOWELS:
                if v != ch:
                    add(stem[:i] + v + stem[i + 1:], "vowel %s to %s" % (ch, v))
    for c in "bdfghklmnprstv":                             # change the first letter
        if c != stem[0]:
            add(c + stem[1:], "first letter %s to %s" % (stem[0], c))
    for i in range(1, len(stem)):                          # drop one letter
        add(stem[:i] + stem[i + 1:], "dropped '%s'" % stem[i])
    for i in range(1, len(stem)):                          # add one letter
        for c in "kxzvrm":
            add(stem[:i] + c + stem[i:], "added '%s'" % c)
    return out


def _worst(e, sim, ing, brand_idx, look=40):
    """Closest existing brand that is a different drug: (similarity, name)."""
    order = brand_idx[np.argsort(-sim[brand_idx])[:look]]
    for j in order:
        if ing and e.overlap(ing, int(j)) >= 0.5:
            continue
        return float(sim[j]), e.names[int(j)]
    return 0.0, "-"


def suggest(e, name, ing, n=5, limit=700):
    """The n variations of `name` that are least like any existing different-drug brand.
    Returns (proposed_sim, proposed_closest, [dict(name, why, sim, closest, kept)])."""
    stem = brand_stem(name)
    brand_idx = np.array([i for i, k in enumerate(e.kinds) if k == "brand"])
    _, _, _, s0 = e.scores(stem)
    if stem in e.index:
        s0 = s0.copy(); s0[e.index[stem]] = -1
    base_sim, base_name = _worst(e, s0, ing, brand_idx)
    rows = []
    for cand, why in list(candidates(stem).items())[:limit]:
        if cand in e.index:
            continue
        _, _, _, s = e.scores(cand)
        sim, closest = _worst(e, s, ing, brand_idx)
        rows.append(dict(name=cand, why=why, sim=sim, closest=closest, kept=1 - lev(stem, cand) / max(len(stem), len(cand))))
    rows.sort(key=lambda r: (round(r["sim"], 3), -r["kept"]))
    out, seen_end = [], set()
    for r in rows:                                          # avoid five near-copies of the same idea
        key = r["name"][:3] + r["name"][-2:]
        if key in seen_end:
            continue
        seen_end.add(key); out.append(r)
        if len(out) == n:
            break
    return base_sim, base_name, out


def shelf_plan(e, raw_names, n_shelves, min_sim):
    """Place a pharmacy's medicines on shelves so that close, different-drug names are on different shelves.
    Returns dict(found, missing, pairs, shelves, conflicts)."""
    stems, missing = [], []
    for r in raw_names:
        s = brand_stem(r)
        if not s:
            continue
        if s in e.index:
            if s not in stems:
                stems.append(s)
        elif s not in missing:
            missing.append(s)
    idx = [e.index[s] for s in stems]
    N = len(stems)
    S = np.zeros((N, N))
    for a, s in enumerate(stems):
        S[a] = e.scores(s)[3][idx]
    S = (S + S.T) / 2
    pairs, W = [], np.zeros((N, N))
    for a in range(N):
        for b in range(a + 1, N):
            if S[a, b] >= min_sim and e.overlap(e.ings[idx[a]], idx[b]) < 0.5:
                pairs.append((stems[a], stems[b], float(S[a, b])))
                W[a, b] = W[b, a] = S[a, b]
    pairs.sort(key=lambda p: -p[2])
    zone = [int(e.km.labels_[i]) for i in idx]
    shelves = [[] for _ in range(n_shelves)]
    where = {}
    for a in sorted(range(N), key=lambda a: -W[a].sum()):   # most-conflicted names first
        def cost(k):
            clash = sum(W[a, b] for b in shelves[k])
            same_zone = sum(1 for b in shelves[k] if zone[b] == zone[a])      # k-means zone as a tie-break
            return (round(clash, 6), same_zone, len(shelves[k]))
        k = min(range(n_shelves), key=cost)
        shelves[k].append(a); where[a] = k
    conflicts = [p for p in pairs if where[stems.index(p[0])] == where[stems.index(p[1])]]
    return dict(found=stems, missing=missing, pairs=pairs, conflicts=conflicts,
                shelves=[[stems[a] for a in sh] for sh in shelves], zone=dict(zip(stems, zone)))
