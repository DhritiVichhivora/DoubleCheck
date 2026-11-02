"""Parse the ISMP List of Confused Drug Names (June 2024) PDF into name pairs."""
import re, json, sys
import pdfplumber

def col(xc):
    return 0 if xc < 175 else 1 if xc < 300 else 2 if xc < 420 else 3

def parse(path):
    pairs = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = [w for w in page.extract_words() if 120 < w["top"] < 735]
            rows = {}
            for w in words:
                rows.setdefault(round(w["top"] / 4), []).append(w)
            for _, ws in sorted(rows.items()):
                cells = {}
                for w in sorted(ws, key=lambda w: w["x0"]):
                    c = col((w["x0"] + w["x1"]) / 2)
                    cells.setdefault(c, []).append(w["text"])
                cells = {k: " ".join(v) for k, v in cells.items()}
                for a, b in ((0, 1), (2, 3)):
                    if a in cells and b in cells:
                        pairs.append((cells[a], cells[b]))
    return pairs

if __name__ == "__main__":
    p = parse(sys.argv[1])
    json.dump(p, open(sys.argv[2], "w"))
    print(len(p)); print(p[:12])
