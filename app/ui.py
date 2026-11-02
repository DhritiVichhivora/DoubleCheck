"""HTML building blocks for the LASA Guard pages. Colours use translucent greys so the same markup works in light and dark."""
import re

GREEN, YELLOW, ORANGE, RED = "#2e9d4f", "#f2b01e", "#ef6c00", "#d32f2f"
BAND_COLOUR = {"Very high": RED, "High": ORANGE, "Watch": "#c98a00", "Low": GREEN}


def css(dark):
    fg, bg, card, line, side = ("#e8eaed", "#0e1117", "rgba(255,255,255,.06)", "rgba(255,255,255,.18)", "#161a22") if dark else \
                               ("#1b1f24", "#ffffff", "rgba(0,0,0,.035)", "rgba(0,0,0,.14)", "#f3f4f7")
    return """<style>
.stApp, [data-testid="stAppViewContainer"] {background:%(bg)s !important; color:%(fg)s !important}
[data-testid="stHeader"] {background:transparent !important}
[data-testid="stHeader"] * {color:%(fg)s !important}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {background:%(side)s !important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {color:%(fg)s !important}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div {background:%(card)s !important; border-color:%(line)s !important}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] * {color:%(fg)s !important}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp span, .stApp li, .stApp div[data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stFileUploaderDropzoneInstructions"] * {color:%(fg)s !important}
.stApp [data-testid="stFileUploaderDropzone"] {background:%(card)s !important; border:1px dashed %(line)s !important}
.stApp [data-testid="stFileUploaderDropzone"] button {background:transparent !important; border:1px solid %(line)s !important; color:%(fg)s !important}
.stApp [data-testid="stFileUploader"] small {color:%(fg)s !important; opacity:.7}
[data-testid="stDecoration"] {display:none !important}
.block-container {max-width:1020px; padding:3.2rem 1rem 2.5rem !important}
.stApp {font-size:14px}
.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"] {font-size:.92rem}
[data-testid="stSidebar"] {min-width:220px !important; max-width:240px !important}
[data-testid="stSidebar"] label p {font-size:.88rem !important}
.nw {color:%(fg)s; font-family:"Source Sans Pro",system-ui,sans-serif}
.nw h2 {font-size:.9rem; margin:1.4rem 0 .6rem; font-weight:700; letter-spacing:.02em; text-transform:uppercase; opacity:.8}
.nw-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:10px}
.nw-grid2 {display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:10px}
.nw-card {border:1px solid %(line)s; background:%(card)s; border-radius:10px; padding:11px 13px; min-width:0; font-size:.88rem}
.nw-card img {width:100%%; display:block; border-radius:10px; background:#808080}
.nw-title {display:flex; align-items:center; gap:10px; font-weight:700; margin-bottom:8px}
.nw-n {flex:none; width:26px; height:26px; border-radius:50%%; background:%(green)s; color:#fff; display:flex; align-items:center; justify-content:center; font-size:.9rem}
.nw-cap {font-size:.8rem; opacity:.75; margin-top:6px; line-height:1.4}
.nw-badge {display:inline-block; padding:3px 10px; border-radius:999px; font-size:.82rem; font-weight:600; margin-top:8px; color:#fff}
.stApp .nw-badge {color:#fff !important}
.nw-ok {background:%(green)s} .nw-warn {background:#c98a00} .nw-bad {background:%(red)s}
.nw-big {font-size:2.4rem; font-weight:800; line-height:1}
.nw-row {display:grid; grid-template-columns:minmax(110px,40%%) 1fr 44px; gap:8px; align-items:center; margin:5px 0; font-size:.82rem}
.nw-track {position:relative; height:14px; background:rgba(128,128,128,.16); border-radius:7px}
.nw-mid {position:absolute; left:50%%; top:-2px; bottom:-2px; width:2px; background:%(line)s}
.nw-bar {position:absolute; top:0; bottom:0; border-radius:7px}
.nw-val {text-align:right; font-variant-numeric:tabular-nums; opacity:.85}
.nw-legend {display:flex; justify-content:space-between; font-size:.8rem; opacity:.7; margin-top:6px}
.nw-strip {position:relative; height:14px; border-radius:7px; margin:34px 6px 62px; background:linear-gradient(90deg,%(green)s 0%%,%(yellow)s 25%%,%(orange)s 50%%,%(red)s 100%%)}
.nw-pin {position:absolute; top:-26px; transform:translateX(-50%%); font-weight:700; font-size:.85rem; white-space:nowrap}
.nw-dot {position:absolute; top:-3px; width:20px; height:20px; border-radius:50%%; background:currentColor; border:3px solid %(bg)s; transform:translateX(-50%%); box-sizing:border-box}
.nw-ref {position:absolute; top:18px; transform:translateX(-50%%); font-size:.76rem; opacity:.8; text-align:center; line-height:1.15; white-space:nowrap}
.nw-ref:before {content:""; position:absolute; left:50%%; top:-6px; margin-left:-1px; height:8px; width:2px; background:currentColor}
.nw details {border:1px solid %(line)s; border-radius:14px; padding:10px 14px; margin-top:10px; background:%(card)s}
.nw summary {cursor:pointer; font-weight:700}
.nw details p, .nw details li {font-size:.84rem; line-height:1.45}
.nw-aucrow {display:grid; grid-template-columns:minmax(120px,34%%) 1fr 44px; gap:8px; align-items:center; margin:6px 0; font-size:.82rem}
.nw-auct {position:relative; height:12px; background:rgba(128,128,128,.16); border-radius:6px}
.nw-aucb {position:absolute; top:0; bottom:0; border-radius:6px; background:%(green)s; opacity:.85}
.nw-aucw {position:absolute; top:5px; height:2px; background:currentColor; opacity:.7}
.nw-hero {text-align:left; padding:.2rem 0 .4rem; border-bottom:1px solid %(line)s; margin-bottom:.4rem}
.nw-hero h1 {font-size:clamp(1.25rem,4vw,1.65rem); font-weight:800; line-height:1.2; margin:0; padding:0}
.nw-hero p {opacity:.7; margin:.3rem 0 .6rem; font-size:.88rem}
.nw-sec {border-left:3px solid %(red)s; padding:1px 0 1px 9px; margin:1.3rem 0 .6rem; font-size:.95rem; font-weight:700; text-transform:none}
.nw-kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:9px}
.nw-kpi {border:1px solid %(line)s; background:%(card)s; border-radius:10px; padding:9px 12px}
.nw-kpi b {display:block; font-size:1.45rem; line-height:1.1; font-weight:800}
.nw-kpi span {display:block; font-size:.74rem; opacity:.75; margin-top:3px; line-height:1.3}
.nw-flow {display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px}
.nw-flow > div {position:relative; border:1px solid %(line)s; background:%(card)s; border-radius:10px; padding:9px; text-align:center; font-size:.8rem}
.nw-flow > div b {display:block; font-size:1.15rem; margin-bottom:2px}
.nw-tbl {width:100%%; border-collapse:collapse; font-size:.8rem}
.nw-tbl th {text-align:left; font-weight:700; opacity:.8; border-bottom:2px solid %(line)s; padding:7px 8px}
.nw-tbl td {border-bottom:1px solid %(line)s; padding:7px 8px; vertical-align:top}
.nw-wrap {overflow-x:auto}
.nw-tag {display:inline-block; font-size:.75rem; font-weight:700; padding:2px 9px; border-radius:999px; border:1px solid %(line)s; margin:0 6px 4px 0; opacity:.9}
.nw-vd {display:inline-block; font-weight:800; font-size:1.05rem; padding:6px 18px; border-radius:999px; color:#fff !important}
.nw-vfit {background:%(green)s} .nw-vmid {background:#c98a00} .nw-vbad {background:%(red)s}
.stApp .nw-vd {color:#fff !important}
.nw-pass {color:%(green)s; font-weight:700} .nw-fail {color:%(red)s; font-weight:700}
</style>""" % dict(fg=fg, bg=bg, card=card, line=line, side=side, green=GREEN, yellow=YELLOW, orange=ORANGE, red=RED)



def _plain(t):
    return re.sub(r"^[^A-Za-z0-9&<(]+", "", t)


def hero(title, sub):
    title = _plain(title)
    return '<div class="nw nw-hero"><h1>%s</h1><p>%s</p></div>' % (title, sub)


def section(text):
    text = _plain(text)
    return '<div class="nw"><div class="nw-sec">%s</div></div>' % text


def kpis(items):
    """items: (big value, label)"""
    return '<div class="nw"><div class="nw-kpis">%s</div></div>' % "".join(
        '<div class="nw-kpi"><b>%s</b><span>%s</span></div>' % (v, l) for v, l in items)


def flow(items):
    """items: (emoji, title, sub)"""
    return '<div class="nw"><div class="nw-flow">%s</div></div>' % "".join(
        '<div><b>%s</b>%s<div style="opacity:.7;font-size:.82rem;margin-top:3px">%s</div></div>' % t for t in items)


def table(headers, rows):
    head = "".join("<th>%s</th>" % h for h in headers)
    body = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % c for c in r) for r in rows)
    return '<div class="nw"><div class="nw-wrap"><table class="nw-tbl"><tr>%s</tr>%s</table></div></div>' % (head, body)


def palette(dark):
    return ("#e8eaed", "rgba(255,255,255,.14)") if dark else ("#1b1f24", "rgba(0,0,0,.12)")


def chip(band):
    return '<span class="nw-badge" style="background:%s;margin-top:0">%s</span>' % (BAND_COLOUR[band], band)


def bar(value, colour=GREEN):
    return '<div class="nw-auct" style="min-width:90px"><div class="nw-aucb" style="left:0;width:%d%%;background:%s"></div></div>' % (round(100 * value), colour)
