# DoubleCheck: a look-alike / sound-alike medicine name checker for Indian brands

Foundations of Machine Learning mini project, NMIMS MPSTME (Dhriti Vichhivora, E014).
PCA, kNN and k-means are written from scratch in NumPy, in the style of the FOML lab notebooks.

## The problem
Many medicines have names that look or sound almost the same. A hand-written prescription, a rushed pharmacist or a tired nurse
can pick the wrong one. These look-alike sound-alike (LASA) errors are a recognised cause of medication harm. India has thousands of
brand names and new ones launch every month.

**What the tool does:** type a brand name and it lists the names most likely to be confused with it, flags the ones that are a
*different drug* (same ingredient is harmless, different ingredient is the danger), and gives a risk band.
A second page checks a *proposed new brand name* against existing ones before launch.

## Who would use it
- A pharmacy or hospital: warn when a typed name is close to a different drug; plan shelves so risky pairs are not side by side.
- A drug company naming a new brand: check the name before launch.
- A hospital formulary committee: decide which pairs need separate storage or tall-man lettering.

## How it works
1. **Clean the name.** `Losar-H 50 Tablet` becomes `losar`.
2. **Make numbers.** Letter groups (2 and 3 letters) of the spelling and of a simple *sound key* (ph = f, ck = k, soft c = s, y = i ...), weighted by TF-IDF: 3,000 numbers per name.
3. **PCA** (eigen-decomposition of the covariance matrix, lab-notebook steps) squeezes 3,000 numbers into 256.
4. **kNN** finds the nearest names by cosine similarity in PCA space; closeness = half PCA cosine + half Levenshtein edit distance on the sound key.
5. **Ingredient check.** Close names with the same active ingredient are harmless; close names with a different ingredient are the warning.
6. **Risk band** = how close the pair is compared with all 21 million name pairs: Watch (closest 1%), High (0.1%), Very high (0.01%).
7. **k-means** (40 zones) groups names into crowded neighbourhoods.

## Results (honest)
Data: 11,825 Indian medicine products, giving 5,746 brand names and 749 generic names.

| Test | Result |
|---|---|
| Official ISMP confused-drug-name pairs found in the Indian data | 53 pairs |
| Partner name in the 10 closest of 6,495 names | **50%** (95% CI 37% to 62%); random guessing would give 0.15% |
| ISMP pairs inside the closest 1% / 0.1% of all 21 M pairs | 68% / 42% |
| Misspelt or mis-heard name: original recovered as #1 / in top 5 (500 random names) | 92% / 99.6% |
| Hand-written PCA vs scikit-learn | explained variance identical to 1e-8; kNN and k-means match |

**Reading these numbers honestly**
- Edit distance alone (55%, CI 42% to 67%) is as good as the PCA version (50%) on the ISMP pairs; with only 53 pairs the differences are inside the margin of error. PCA does **not** make the search more accurate here. What it does is shrink 3,000 numbers to 256 (12 times smaller) at the same accuracy, and it gives the map, component meanings and name zones.
- About half of the ISMP pairs are not found in the top 10. Some pairs are probably confusable for reasons other than spelling. A quiet result is not proof of safety.
- The risk bands are set only from how common close names are among all pairs, not from the ISMP list, so the ISMP numbers are not inflated by tuning. The number of PCA components (256) was chosen on a separate misspelling test; the 0.5 / 0.5 weights were fixed in advance and not tuned.

## What makes it different
Commercial pharmacy systems and the ISMP / FDA tall-man lists exist. This project does not claim to beat them. It offers:
- an open, small, explainable method where every warning can be traced step by step;
- coverage of **Indian brand names**, which the American ISMP list does not cover (I did not find a free tool for Indian brands, but could not prove none exists);
- a new-brand name check and a list of the riskiest look-alike pairs in the Indian sample.

## Limitations
- Compares written names only: no pronunciation, handwriting, packaging, strength or dosage form.
- A sample (5,746 brands), not the whole Indian market. A clear result is not a clearance and not trademark advice.
- "Risk" means closeness of names, not a measured error rate. No pharmacy trial was done.
- Ingredient data comes from a scraped source and contains errors (for example a misspelt brand with the wrong ingredient), which the ingredient check inherits.
- A student project and a screening aid, not a medical device.

## Run it
```
pip install -r requirements.txt
streamlit run app\app.py
```
To rebuild everything from the raw data:
```
pip install -r requirements-build.txt
python lassa\download_data.py     # downloads the medicine data and the ISMP list
python lassa\build_model.py       # model, validation numbers, riskiest pairs (about 3 minutes)
python lassa\lab_check.py         # compares the hand-written PCA, kNN and k-means with scikit-learn
```

## Folders
- `app/`: Streamlit dashboard (7 pages, light and dark).
- `lassa/`: `lab_models.py` (PCA, kNN, k-means, lab style), `names.py` (cleaning, sound key, n-grams, edit distance), `engine.py`, `validate.py`, `build_model.py`, `lab_check.py`, `parse_ismp.py`, `download_data.py`.
- `models/`: the fitted model (`lassa_v1.npz` and `lassa_v1.json`).
- `docs/`: validation numbers, riskiest pairs, lab cross-check.
- `data/`: raw inputs (not committed: see licences).

## Course topics used
PCA (covariance, eigenvectors, scree, loadings), kNN, k-means, evaluation and bias (held-out names, official list, confidence intervals, limitations).

## Data and credits
- Indian medicines: Hugging Face dataset `dmedhi/indian-medicines` (11,825 products). No licence is stated on the dataset page, so it is used for study only and is not committed to this repository.
- ISMP List of Confused Drug Names (June 2024), (c) ISMP / ECRI. Used only to check results; not redistributed.
