"""Download the two inputs and prepare them. Run once:  python lassa/download_data.py
  1. Indian medicines (Hugging Face dataset dmedhi/indian-medicines) -> data/indian_medicines.parquet
  2. ISMP List of Confused Drug Names (PDF) -> parsed into data/ismp_pairs.json (used only to validate)."""
import json, os, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
MED = "https://huggingface.co/datasets/dmedhi/indian-medicines/resolve/main/data/train-00000-of-00001.parquet"
ISMP = "https://online.ecri.org/hubfs/ISMP/Resources/ISMP_ConfusedDrugNames.pdf"
urllib.request.urlretrieve(MED, os.path.join(ROOT, "data", "indian_medicines.parquet"))
pdf = os.path.join(ROOT, "data", "ismp_confused_names.pdf")
urllib.request.urlretrieve(ISMP, pdf)
from parse_ismp import parse
pairs = parse(pdf)
json.dump(pairs, open(os.path.join(ROOT, "data", "ismp_pairs.json"), "w"))
print("medicines saved; ISMP rows parsed:", len(pairs))
