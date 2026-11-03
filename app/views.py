"""Pages of the DoubleCheck dashboard."""
import html
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui
from engine import Engine
from names import brand_stem, sound_key

ROOT = Path(__file__).resolve().parent.parent
esc = html.escape


def md(h):
    st.markdown(h, unsafe_allow_html=True)


@st.cache_resource
def engine():
    return Engine.load(str(ROOT / "models" / "lassa_v1"))


@st.cache_data
def info():
    return json.load(open(ROOT / "docs" / "validation.json"))


@st.cache_data
def risky():
    return json.load(open(ROOT / "docs" / "risky_pairs.json"))


@st.cache_data
def map_xy():
    return np.load(ROOT / "docs" / "map_xy.npy")


def band(s):
    c = info()["cut"]
    return "Very high" if s >= c["very_high"] else "High" if s >= c["high"] else "Watch" if s >= c["watch"] else "Low"


def pct(x, d=0):
    return ("%." + str(d) + "f%%") % (100 * x)


def ing_txt(L, n=3):
    L = list(L)
    return esc(", ".join(L[:n]) + (" +%d" % (len(L) - n) if len(L) > n else "")) if L else "-"


def card(n, title, body):
    return '<div class="nw-card"><div class="nw-title"><span class="nw-n">%d</span>%s</div>%s</div>' % (n, title, body)


def legend_html():
    return ('<div class="nw"><div class="nw-cap">Risk bands compare this pair with all %s name pairs: %s closest 1%% &nbsp; %s closest 0.1%% &nbsp; %s closest 0.01%%.</div></div>'
            % (f"{info()['n_pairs']:,}", ui.chip("Watch"), ui.chip("High"), ui.chip("Very high")))


def verdict_html(b):
    cls = {"Very high": "nw-vbad", "High": "nw-vbad", "Watch": "nw-vmid", "Low": "nw-vfit"}[b]
    return '<span class="nw-vd %s">%s risk of confusion</span>' % (cls, b)


# ------------------------------------------------------------------ 1. overview
def overview(dark):
    v = info()
    md(ui.hero("DoubleCheck", "A look-alike / sound-alike medicine name checker for Indian brands"))
    md(ui.kpis([("%s" % f"{v['n_brands']:,}", "Indian brand names checked"), ("%s" % f"{v['n_pairs']/1e6:.0f} M", "name pairs compared"),
                (pct(v["ismp"]["PCA + edit distance (shipped)"]["top10"][0]), "official confusable pairs found in top 10"),
                (pct(v["corruption"]["PCA + edit distance (shipped)"]["1"]), "misspelt names recovered (top 1)")]))
    md('<div class="nw"><div class="nw-card" style="font-size:1rem"><b>Type any medicine name and, in a second, see which look-alike names could cause a mix-up, and which of them are a different drug.</b></div></div>')
    md(ui.section("Try it in 10 seconds"))
    md('<div class="nw"><div class="nw-grid">'
       '<div class="nw-card"><b>1. Check a Medicine</b><div class="nw-cap">Click <b>Lasix</b>. See its look-alikes and which ones are a different drug.</div></div>'
       '<div class="nw-card"><b>2. New Brand Name Check</b><div class="nw-cap">Keep <b>Lasoprin</b> with aspirin. See why the tool says "review before using".</div></div>'
       '<div class="nw-card"><b>3. Riskiest Pairs</b><div class="nw-cap">Type <b>cef</b> in the filter to see the closest look-alike antibiotics.</div></div>'
       '</div></div>')
    md(ui.section("The problem"))
    md('<div class="nw"><div class="nw-card">Many medicines have names that look or sound almost the same. A rushed pharmacist, a hand-written prescription or a tired nurse can pick the wrong one. '
       'These look-alike sound-alike errors (called LASA errors) are a recognised cause of medication harm. India makes it harder: thousands of brand names, new launches every month.'
       '<div class="nw-cap">What this tool does: type a medicine name and it lists the names most likely to be confused with it, and flags the ones that are a <b>different drug</b>.</div></div></div>')
    md(ui.section("How it works"))
    md(ui.flow([("1", "Clean the name", "Losar-H 50 Tablet becomes losar"), ("2", "Make numbers", "letter groups of the spelling and the sound"),
                ("3", "PCA", "3,000 numbers become 256"), ("4", "Find neighbours", "kNN plus edit distance"), ("5", "Check the drug", "same ingredient is harmless, different is a warning")]))
    md(ui.section("Who would use it"))
    md('<div class="nw"><div class="nw-grid">'
       '<div class="nw-card"><b>Pharmacy or hospital</b><div class="nw-cap">Warn when a typed name is close to a different drug, and plan shelf layout so risky pairs are not side by side.</div></div>'
       '<div class="nw-card"><b>Drug company naming a new brand</b><div class="nw-cap">Check a proposed brand name against existing ones before launch (see "New Brand Name Check").</div></div>'
       '<div class="nw-card"><b>Hospital formulary committee</b><div class="nw-cap">Review which brands are stocked together and which pairs need tall-man lettering or separate storage.</div></div>'
       '</div></div>')
    md(ui.section("What this is, and is not"))
    md('<div class="nw"><div class="nw-card">This is a <b>screening aid</b> that ranks how close two names are. It does not measure how often a mix-up really happens, and it is not a medical device. '
       'The method is deliberately small and open, so every warning can be explained. It is not claimed to beat commercial pharmacy software. See Validation and Limitations for the honest numbers.</div></div>')


# ------------------------------------------------------------------ 2. check a medicine
EXAMPLES = ["Losar", "Lasix", "Zyrtec", "Allegra", "Minocycline"]


def check_page(dark):
    e = engine()
    md(ui.hero("Check a Medicine", "Type a brand name and see which names it could be confused with"))
    if "q" not in st.session_state:
        st.session_state["q"] = "Losar"
    cols = st.columns(len(EXAMPLES))
    for c, ex in zip(cols, EXAMPLES):
        if c.button(ex, width="stretch"):
            st.session_state["q"] = ex
    q = st.text_input("Brand name", key="q", placeholder="e.g. Losar, Zyrtec, Allegra")
    stem = brand_stem(q) or "".join(ch for ch in q.lower() if ch.isalpha())
    if len(stem) < 3:
        md('<div class="nw"><div class="nw-card">Type at least three letters.</div></div>')
        return
    known = stem in e.index
    ing = None
    if known:
        me = e.index[stem]
        ing = e.ings[me]
    else:
        txt = st.text_input("This name is not in the Indian sample. Optional: its active ingredient(s), comma separated", placeholder="e.g. furosemide")
        ing = [x.strip().lower() for x in txt.split(",") if x.strip()] or None
    stem, nb = e.neighbours(q, topn=80, ing=ing)
    diff = [n for n in nb if n["overlap"] is not None and n["overlap"] < 0.5]
    same = [n for n in nb if n["overlap"] is not None and n["overlap"] >= 0.5]
    unk = [n for n in nb if n["overlap"] is None]
    top_band = band(diff[0]["sim"]) if diff else band(unk[0]["sim"]) if unk else "Low"

    md(ui.section("Result"))
    head = '<div class="nw"><div class="nw-card">'
    if known:
        head += "<b>%s</b>: %s<div class='nw-cap'>Ingredients across all %s variants. Example products: %s</div>" % (esc(stem), ing_txt(ing, 5), esc(stem), esc("; ".join(e.prods[me][:3])))
    else:
        head += "<b>%s</b>: not in the Indian sample%s" % (esc(stem), "" if ing else " (add an ingredient above to separate harmless from risky matches)")
    if ing or known:
        head += "<div style='margin-top:10px'>%s</div><div class='nw-cap'>Based on the closest <b>different</b> drug: %s</div>" % (
            verdict_html(top_band), esc(diff[0]["name"]) if diff else "none found among the 80 nearest names")
    head += "</div></div>"
    md(head)

    if ing:
        md(ui.section("Look-alike names that are a different drug"))
        rows = [(esc(n["name"]) + (" <span class='nw-cap'>(generic)</span>" if n["kind"] == "generic" else ""), ui.chip(band(n["sim"])), pct(n["sim"]), ing_txt(n["ingredients"]),
                 esc("; ".join(n["products"][:1])) or "-") for n in diff[:10]]
        md(ui.table(["Name", "Risk", "Similarity", "Active ingredient", "Example product"], rows) if rows else
           '<div class="nw"><div class="nw-card">No different drug among the 80 closest names.</div></div>')
        md(legend_html())
        if same:
            with st.expander("Close names that are the same drug (harmless to mix up) - %d" % len(same)):
                md(ui.table(["Name", "Similarity", "Active ingredient"], [(esc(n["name"]), pct(n["sim"]), ing_txt(n["ingredients"])) for n in same[:10]]))
    else:
        md(ui.section("Closest names"))
        md(ui.table(["Name", "Risk", "Similarity", "Active ingredient"], [(esc(n["name"]), ui.chip(band(n["sim"])), pct(n["sim"]), ing_txt(n["ingredients"])) for n in nb[:10]]))
        md(legend_html())

    md(ui.section("What happened to your name, step by step"))
    x = e.vec.transform([stem])[0]
    top = np.argsort(-x)[:4]
    fn = e.vec.feature_names
    z = e.pca.transform(x[None, :])[0]
    md('<div class="nw"><div class="nw-grid">'
       + card(1, "Clean the name", "<div class='nw-big' style='font-size:1.5rem'>%s</div><div class='nw-cap'>Only the first word is kept, in lower case. Strength and form words (50, tablet) are dropped.</div>" % esc(stem))
       + card(2, "Spell it by sound", "<div class='nw-big' style='font-size:1.5rem'>%s</div><div class='nw-cap'>Simple rules: ph is f, ck is k, soft c is s, y is i, doubled letters collapse.</div>" % esc(sound_key(stem)))
       + card(3, "Turn it into numbers", "<div class='nw-big' style='font-size:1.5rem'>%d of 3,000</div><div class='nw-cap'>letter groups are present. Most distinctive: %s. Rare groups count more (TF-IDF).</div>" % (int((x > 0).sum()), esc(", ".join(fn[i].replace("s", "spelling ", 1) if fn[i][0] == "s" else fn[i].replace("p", "sound ", 1) for i in top))))
       + card(4, "Squeeze with PCA", "<div class='nw-big' style='font-size:1.5rem'>3,000 &rarr; %d</div><div class='nw-cap'>numbers. PC1 = %.2f, PC2 = %.2f. Names with similar numbers are close in this space (cosine).</div>" % (len(z), z[0], z[1]))
       + card(5, "Find and compare", "<div class='nw-cap'>kNN finds the nearest names. Similarity = half PCA cosine + half edit distance on the sound key. The result is compared with all %s name pairs to set the risk band.</div>" % f"{info()['n_pairs']:,}")
       + card(6, "Check the drug", "<div class='nw-cap'>Close names with the <b>same</b> ingredient are harmless. Close names with a <b>different</b> ingredient are the warning.</div>")
       + "</div></div>")
    c = info()["cut"]
    md('<div class="nw"><div class="nw-cap">Risk bands: Watch = closer than 99%% of all name pairs (similarity &ge; %.2f), High = closer than 99.9%% (&ge; %.2f), Very high = closer than 99.99%% (&ge; %.2f).</div></div>'
       % (c["watch"], c["high"], c["very_high"]))


# ------------------------------------------------------------------ 3. new brand name check
def newname_page(dark):
    e = engine()
    md(ui.hero("New Brand Name Check", "Before launching a brand, see which existing names it could be confused with"))
    name = st.text_input("Proposed brand name", value="Lasoprin")
    known = sorted({g for L in e.ings for g in L if " " not in g})
    ing = st.multiselect("Active ingredient(s) of the new drug", known, default=["aspirin"] if "aspirin" in known else [])
    extra = st.text_input("Ingredient not in the list? Type it here")
    ing = [i.lower() for i in ing] + ([extra.strip().lower()] if extra.strip() else [])
    stem = brand_stem(name)
    if len(stem) < 3 or not ing:
        md('<div class="nw"><div class="nw-card">Enter a name of at least three letters and at least one ingredient.</div></div>')
        return
    stem, nb = e.neighbours(name, topn=80, ing=ing, kinds=("brand",))
    diff = [n for n in nb if n["overlap"] is not None and n["overlap"] < 0.5]
    exact = stem in e.index
    b = band(diff[0]["sim"]) if diff else "Low"
    msg = {"Very high": ("Do not use this name", "It is closer to an existing different drug than 99.99% of all name pairs."),
           "High": ("Review before using", "Closer to an existing different drug than 99.9% of all name pairs."),
           "Watch": ("Acceptable, with care", "Closer than 99% of pairs to a different drug. Consider a clearer spelling."),
           "Low": ("Looks clear", "No close different-drug names in this sample.")}[b]
    md(ui.section("Verdict"))
    md('<div class="nw"><div class="nw-card">%s<div style="margin-top:10px;font-weight:700">%s</div><div class="nw-cap">%s%s</div></div></div>'
       % (verdict_html(b), msg[0], msg[1], " <b>This exact name already exists in the sample.</b>" if exact else ""))
    md(ui.section("Closest existing brands with a different drug"))
    md(ui.table(["Existing brand", "Risk", "Similarity", "Its active ingredient", "Example product"],
                [(esc(n["name"]), ui.chip(band(n["sim"])), pct(n["sim"]), ing_txt(n["ingredients"]), esc("; ".join(n["products"][:1]))) for n in diff[:8]]))
    md(legend_html())
    md('<div class="nw"><div class="nw-cap">Only the 5,746 brands in the Indian sample are checked, not the whole market, and not trademark databases. A clear result here is not a legal clearance.</div></div>')


# ------------------------------------------------------------------ 4. PCA
def pca_page(dark):
    e, v = engine(), info()
    fg, line = ui.palette(dark)
    md(ui.hero("PCA: the core", "How 3,000 letter-group numbers become 256, and what the components mean"))
    md(ui.kpis([("3,000 &rarr; 256", "numbers per name"), (pct(v["var_kept"]), "variance kept by 256 components"), (pct(v["var_2"], 1), "variance in the first two"), ("%d" % len(e.names), "names in the space")]))
    md(ui.section("Step by step (as in the lab notebook)"))
    md('<div class="nw"><div class="nw-card"><ol style="margin:0 0 0 18px;padding:0">'
       "<li>Each name is a row of 3,000 numbers (TF-IDF weights of spelling and sound letter groups).</li>"
       "<li>Subtract the column means (centre the data).</li><li>Compute the 3,000 &times; 3,000 covariance matrix.</li>"
       "<li>Find its eigenvalues and eigenvectors with <code>np.linalg.eigh</code> and sort the biggest first.</li>"
       "<li>Project each name on the first 256 eigenvectors: <code>Z = centred &middot; eigenvectors</code>.</li>"
       "<li>kNN and k-means then work in this 256-number space.</li></ol>"
       '<div class="nw-cap">The code is in <code>lassa/lab_models.py</code> and follows the same steps as your lab notebook; <code>lassa/lab_check.py</code> compares it with sklearn.</div></div></div>')
    md(ui.section("Scree plot"))
    ratio = np.array(v["ratio"]); cum = np.array(v["var_cum"])
    k = st.slider("Number of components", 2, 400, v["k"])
    d = pd.DataFrame({"component": np.arange(1, 401), "variance": ratio, "cumulative": cum})
    base = alt.Chart(d).encode(x=alt.X("component:Q", title="component"))
    ch = (base.mark_line(color=ui.GREEN).encode(y=alt.Y("cumulative:Q", title="cumulative variance kept", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])))
          + alt.Chart(pd.DataFrame({"x": [k]})).mark_rule(color=ui.RED).encode(x="x:Q")).properties(height=230)
    st.altair_chart(ch.configure(background="transparent").configure_axis(labelColor=fg, titleColor=fg, gridColor=line, domainColor=line).configure_view(strokeWidth=0), width="stretch", theme=None)
    md('<div class="nw"><div class="nw-cap">With %d components %s of the variance is kept. Name data is spread over many small directions, so the curve rises slowly. Even so, 256 components find names as well as the full 3,000 numbers (see Validation), and search is 12 times lighter.</div></div>' % (k, pct(cum[k - 1])))
    md(ui.section("What the first components mean"))
    rows = [("PC%d" % L["pc"], pct(L["var"], 2), esc(", ".join(L["plus"][:4])), esc(", ".join(L["minus"][:4]))) for L in v["loadings"]]
    md(ui.table(["Component", "Variance", "High when the name has", "Low when the name has"], rows))
    md('<div class="nw"><div class="nw-cap">A letter group written "x$" means the name ends with x; "^x" means it starts with x. The first components mostly separate names by their endings (-in, -ol, -t, -x), which is how many brand names are built.</div></div>')
    md(ui.section("Map of names (first two components)"))
    xy = map_xy()
    zone = e.km.labels_
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xy), 2500, replace=False)
    df = pd.DataFrame({"PC1": xy[idx, 0], "PC2": xy[idx, 1], "name": [e.names[i] for i in idx], "type": [e.kinds[i] for i in idx]})
    sel = st.text_input("Highlight a name on the map", value="losar")
    layers = [alt.Chart(df).mark_circle(size=14, opacity=.45).encode(x="PC1:Q", y="PC2:Q", color=alt.Color("type:N", scale=alt.Scale(domain=["brand", "generic"], range=[ui.GREEN, "#4e79a7"]), title=None), tooltip=["name", "type"])]
    s = brand_stem(sel)
    if s in e.index:
        _, nb = e.neighbours(s, topn=8)
        hl = pd.DataFrame({"PC1": [xy[e.index[s], 0]] + [xy[n["idx"], 0] for n in nb], "PC2": [xy[e.index[s], 1]] + [xy[n["idx"], 1] for n in nb],
                           "name": [s] + [n["name"] for n in nb], "role": ["your name"] + ["neighbour"] * len(nb)})
        layers.append(alt.Chart(hl).mark_circle(size=90, stroke=fg, strokeWidth=1).encode(x="PC1:Q", y="PC2:Q", color=alt.Color("role:N", scale=alt.Scale(domain=["your name", "neighbour"], range=[ui.RED, ui.ORANGE]), title=None), tooltip=["name", "role"]))
    st.altair_chart(alt.layer(*layers).properties(height=340).configure(background="transparent").configure_axis(labelColor=fg, titleColor=fg, gridColor=line, domainColor=line).configure_legend(labelColor=fg).configure_view(strokeWidth=0), width="stretch", theme=None)
    md('<div class="nw"><div class="nw-cap">The first two components hold only %s of the variance, so this picture is a rough sketch: names that look close here may not be close in the full 256-number space, and the other way round. The real comparisons use all 256 numbers.</div></div>' % pct(v["var_2"], 1))
    md(ui.section("k-means: crowded zones of names"))
    zs = sorted(v["zones"], key=lambda z: -z["risky_pairs"] / max(z["size"], 1))[:8]
    md(ui.table(["Zone", "Names", "Risky pairs inside or touching", "Example names"], [("Z%d" % z["id"], z["size"], z["risky_pairs"], esc(", ".join(z["examples"]))) for z in zs]))
    md('<div class="nw"><div class="nw-cap">k-means (40 zones, written from scratch) groups names that sit near each other in the 256-number space. Zones with many risky pairs for their size are crowded name neighbourhoods; a pharmacy could keep these medicines apart.</div></div>')


# ------------------------------------------------------------------ 5. validation
def validation_page(dark):
    v = info()
    ship = "PCA + edit distance (shipped)"
    md(ui.hero("Validation", "How well it finds confusable names, tested on names it was not tuned on"))
    t10 = v["ismp"][ship]["top10"]
    md(ui.kpis([(str(v["n_ismp"]), "official ISMP pairs found in the Indian sample"), (pct(t10[0]), "partner in the top 10 (95%% CI %s to %s)" % (pct(t10[1]), pct(t10[2]))),
                (pct(v["ismp_cover"]["watch"]), "of ISMP pairs fall in the closest 1% of all pairs"), (pct(v["corruption"][ship]["1"]), "misspelt names recovered (top 1)")]))
    md(ui.section("Test A: the official ISMP confused-names list"))
    md('<div class="nw"><div class="nw-card">The Institute for Safe Medication Practices (ISMP) publishes pairs of drug names that have really been confused in practice (June 2024 list). '
       "%d of its pairs have both names in the Indian sample. For each name, the question is: is its partner among the 10 closest names out of %s? Random guessing would succeed %.2f%% of the time.</div></div>" % (v["n_ismp"], f"{v['n_entities']:,}", 100 * v["chance_top10"]))
    rows = []
    for m, d in v["ismp"].items():
        rows.append((("<b>%s</b>" % esc(m)) if m == ship else esc(m), "%s (%s to %s)" % (pct(d["top10"][0]), pct(d["top10"][1]), pct(d["top10"][2])), pct(d["top20"][0]), pct(d["top50"][0])))
    md(ui.table(["Method", "Partner in top 10 (95% CI)", "Top 20", "Top 50"], rows))
    md('<div class="nw"><div class="nw-cap"><b>Honest reading:</b> all methods find about half of the official pairs, and with only %d pairs the differences between them are inside the margin of error. Edit distance alone is as good as, or slightly better than, the PCA version. PCA does not make the search more accurate here; it makes it 12 times smaller (256 numbers instead of 3,000) at the same accuracy and gives the map and the name zones. Some ISMP pairs are probably confusable for reasons other than spelling, which no name-based method can catch.</div></div>' % v["n_ismp"])
    md(ui.section("Test B: recovering a misspelt or mis-heard name"))
    md('<div class="nw"><div class="nw-card">500 random names, each damaged (two letters swapped, one letter dropped, a look-alike letter substituted, or a letter doubled). Is the original name the closest match? This is a fresh random draw, separate from the names used to choose the number of components.</div></div>')
    md(ui.table(["Method", "Original is #1", "Original in top 5"], [(("<b>%s</b>" % esc(m)) if m == ship else esc(m), pct(d["1"], 1), pct(d["5"], 1)) for m, d in v["corruption"].items()]))
    md(ui.section("How the risk bands were set"))
    c = v["cut"]
    md(ui.table(["Band", "Rule", "Similarity at least", "Share of ISMP pairs inside"],
                [(ui.chip("Watch"), "closest 1%% of all %s pairs" % f"{v['n_pairs']:,}", "%.2f" % c["watch"], pct(v["ismp_cover"]["watch"])),
                 (ui.chip("High"), "closest 0.1%", "%.2f" % c["high"], pct(v["ismp_cover"]["high"])),
                 (ui.chip("Very high"), "closest 0.01%", "%.2f" % c["very_high"], pct(v["ismp_cover"]["very_high"]))]))
    md('<div class="nw"><div class="nw-cap">The bands come only from how common close names are among all pairs, not from the ISMP list, so the ISMP numbers are not inflated by tuning. The typical ISMP pair sits at the %s percentile of all pairs.</div></div>' % ("%.2f" % (100 * v["ismp_median_pct"])))
    md(ui.section("What this does not show"))
    md('<div class="nw"><div class="nw-card"><ul style="margin:0 0 0 18px;padding:0"><li>It does not show that using the tool would reduce real dispensing errors. That needs a trial in a pharmacy.</li>'
       "<li>The ISMP list is American. It checks that known-confusable names rank high, not that the Indian risk bands are exactly right.</li>"
       "<li>About half of the ISMP pairs are not found in the top 10, so a quiet result for a name is not proof of safety.</li></ul></div></div>")


# ------------------------------------------------------------------ 6. riskiest pairs
def pairs_page(dark):
    v = info()
    md(ui.hero("Riskiest Pairs", "The closest look-alike names in the Indian sample that are different drugs"))
    md(ui.kpis([(f"{v['risky_diff']:,}", "different-drug pairs in the closest 0.1%"), ("300", "closest shown here")]))
    q = st.text_input("Filter by name or ingredient", placeholder="e.g. cef, insulin")
    R = risky()
    if q:
        ql = q.lower()
        R = [r for r in R if ql in r["a"] or ql in r["b"] or any(ql in x for x in r["a_ing"] + r["b_ing"])]
    md(ui.section("%d pairs" % len(R)))
    md(ui.table(["Name A", "Ingredient A", "Name B", "Ingredient B", "Similarity", "Risk"],
                [(esc(r["a"]), ing_txt(r["a_ing"], 2), esc(r["b"]), ing_txt(r["b_ing"], 2), pct(r["sim"], 1), ui.chip(r["band"])) for r in R[:120]]))
    md('<div class="nw"><div class="nw-cap">Some pairs here come from the source data, not from real confusion. For example, a misspelt entry such as "minicycline" is listed with a different ingredient than "minocycline". Checking the data source is one of the next steps. Pairs are listed only when similarity reaches the High band.</div></div>')


# ------------------------------------------------------------------ 7. limits & about
def about_page(dark):
    v = info()
    md(ui.hero("Limitations & About", "What this project does not do, and where the data comes from"))
    md(ui.section("Limitations"))
    items = [("Names only", "It compares written names. It does not hear pronunciation, read handwriting, or compare packaging, strength or dosage form."),
             ("A sample, not the market", "%s brand names from %s products. India has many more brands, so a clear result here is not a clearance." % (f"{v['n_brands']:,}", f"{v['products']:,}")),
             ("Risk means similarity, not proven harm", "The bands rank how close two names are. They are not error rates. No trial of real pharmacy use was done."),
             ("About half of the official pairs are missed", "Many confusions are not about spelling. A quiet result is not proof of safety."),
             ("Ingredient data may contain errors", "The source looks scraped from a pharmacy website; entries such as a misspelt brand with the wrong ingredient exist. The ingredient check inherits those errors."),
             ("Not a medical device", "It is a student project and a screening aid. Pharmacists and doctors make the decisions.")]
    md('<div class="nw"><div class="nw-grid2">' + "".join('<div class="nw-card"><b>%s</b><div class="nw-cap">%s</div></div>' % (a, esc(b)) for a, b in items) + "</div></div>")
    md(ui.section("Data and credits"))
    md('<div class="nw"><div class="nw-card"><b>Indian medicines:</b> Hugging Face dataset <code>dmedhi/indian-medicines</code> (11,825 products: name, composition, uses, side effects). No licence is stated on the dataset page, so it is used here only for study and not redistributed in the repository.<br>'
       '<b>Validation list:</b> ISMP List of Confused Drug Names (June 2024), &copy; ISMP, used only to check results, not redistributed.<br>'
       '<b>Method:</b> PCA, kNN and k-means are written in NumPy and follow the FOML lab notebooks; Levenshtein edit distance and TF-IDF are also written out. scikit-learn is used only to cross-check the lab code.</div></div>')
    md(ui.section("Course topics used"))
    md(ui.table(["Topic", "Where"], [("PCA (covariance, eigenvectors, scree)", "name vectors to 256 numbers; map and loadings"), ("kNN", "nearest look-alike names"), ("k-means", "zones of crowded names"),
                                      ("Evaluation and bias", "held-out names, official list, confidence intervals, limits")]))
