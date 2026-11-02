# DoubleCheck: 15-minute presentation script (3 Nov, 11:15 AM)

Before you start: open a terminal in the project folder, run `streamlit run app\app.py`, and keep the app open in the browser on "Overview".
Say the numbers in your own words; they are written out here so you do not have to remember them.

## 0:00 to 1:30  The problem
- "Many medicines have names that look or sound almost the same. Losar is a blood pressure tablet. Lasix is a diuretic. The names look alike, the drugs are completely different."
- "A tired nurse, a rushed pharmacist or an unclear prescription can pick the wrong one. This is called a look-alike sound-alike error, and it is a recognised cause of medication harm. India has thousands of brand names and new ones launch every month."
- "My project is a checker: type a brand name, and it tells you which names could be confused with it, and which of those are a different drug."

## 1:30 to 3:00  Who uses it
- Pharmacy or hospital: warn when a typed name is close to a different drug; keep risky pairs apart on shelves.
- Drug company naming a new brand: check before launch.
- Hospital formulary committee: decide which pairs need separate storage.
- Say plainly: "This does not replace professional pharmacy software. It is a small, open, explainable screening aid."

## 3:00 to 4:00  Data
- 11,825 Indian medicine products, which give 5,746 brand names and 749 generic names.
- For validation, the official ISMP list of confused drug names (June 2024). 53 of its pairs have both names in my Indian data.
- "I used the official list only to test the model, never to tune the risk bands."

## 4:00 to 7:30  Method (the course part)
Show the "PCA: the core" page.
1. Clean the name: `Losar-H 50 Tablet` becomes `losar`.
2. Turn the name into numbers: letter groups of 2 and 3 letters, of the spelling and of a sound key (ph = f, ck = k, soft c = s). TF-IDF weights. Result: 3,000 numbers per name.
3. **PCA**: centre the data, covariance matrix, eigenvalues and eigenvectors with `np.linalg.eigh`, project on the top 256. Written step by step as in our lab notebook. 3,000 numbers become 256 and keep 63% of the variance. Point at the scree plot and move the slider.
4. **kNN**: nearest names by cosine similarity in PCA space. I add edit distance on the sound key (half and half).
5. **Ingredient check**: close names with the same ingredient are harmless; close names with a different ingredient are the warning.
6. **k-means**: 40 zones of names, written from scratch; crowded zones are where mix-ups are likely.
- "I checked my hand-written PCA, kNN and k-means against scikit-learn: PCA explained variance matches to 1e-8, kNN and k-means match."

## 7:30 to 10:30  Live demo (go in this order)
1. **Check a Medicine**, click `Lasix`: show the ingredient (furosemide), the look-alike list, risk chips, and scroll to "what happened to your name, step by step".
2. Click `Zyrtec`, then `Allegra`: different results, same flow.
3. **New Brand Name Check**: keep "Lasoprin" with aspirin. "A company wants to launch this name. The tool says High risk, review before using, and lists the existing brands it is close to."
4. **PCA page**: type `losar` in the map box; the neighbours light up. Say: "The first two components hold only 1.8% of the variance, so this map is a rough sketch; real comparisons use all 256 numbers."
5. **Riskiest Pairs**: "These are the closest different-drug pairs in the Indian sample. I also noticed a data error here: a misspelt entry listed with the wrong ingredient. That shows the data source needs checking."

If the live app fails: say "the app is the same code that produced these numbers" and move to Validation using the README table.

## 10:30 to 12:30  Honest results
- "On the official ISMP list: for 53 known confusable pairs, the partner name is in the 10 closest of 6,495 names for **50%** of cases (95% interval 37 to 62). Random guessing would be 0.15%."
- "68% of those pairs fall in the closest 1% of all 21 million name pairs."
- "On misspelt or mis-heard names, the original is the first match 92% of the time and in the top 5 99.6% of the time."
- **Say this before they ask:** "Edit distance alone scores 55% on the ISMP pairs, as good as my PCA version at 50%. With 53 pairs the difference is inside the margin of error. So PCA did not improve accuracy. What it did: shrink 3,000 numbers to 256 at the same accuracy, and give the map, component meanings and zones."
- "About half of the official pairs are not found, probably because many confusions are not about spelling."

## 12:30 to 13:30  Limitations
- Compares written names only; not pronunciation, handwriting or packaging.
- A sample of the market (5,746 brands), not all of it.
- "Risk" means closeness of names, not a measured error rate; no pharmacy trial.
- Ingredient data comes from a scraped source with some errors.
- Not a medical device.

## 13:30 to 15:00  Close and questions
"What I built: a transparent name-confusion checker for Indian brands, with PCA, kNN and k-means written from scratch, tested on the official ISMP list, with its limits stated. Next step would be a pilot in a pharmacy and a cleaner ingredient source."

---

## Questions your sir may ask, with straight answers

**How is this different from what pharmacies already use?**
"Commercial systems and the ISMP tall-man lists exist and are better backed. I am not claiming to beat them. Mine is open, small and explainable, it covers Indian brand names, which the American list does not, and it adds a new-brand name check. I did not find a free tool for Indian brands, but I cannot prove none exists."

**Why PCA, if edit distance does as well?**
"On this test it does not improve accuracy, and I report that. PCA compresses 3,000 numbers to 256 at the same accuracy, and gives the map, the meaning of components and the k-means zones. It is also the method the course asks us to demonstrate."

**What is your accuracy?**
"There is no single accuracy because there is no label for 'confusable' in my data. I measure two things: how often a known confusable partner appears in the top 10 (50%), and how often a misspelt name is recovered (92% first match)."

**How did you choose the risk bands?**
"From how common close names are among all 21 million pairs: top 1%, 0.1%, 0.01%. Not from the ISMP list, so the ISMP result is not inflated by tuning."

**How did you choose 256 components?**
"On a separate misspelling test, not on the ISMP pairs. The 0.5 and 0.5 weights were fixed in advance."

**Why only 53 pairs?**
"The ISMP list is American. Only 53 of its pairs have both names in my Indian sample. That is why I show confidence intervals."

**Is the data legal to use?**
"The dataset page states no licence, so I use it for study only and do not redistribute it in the repository. The ISMP list is used only for checking."

**Does it reduce real errors?**
"I do not know. That needs a pilot in a pharmacy. This project only shows that the tool ranks known confusable names highly."

**What does k-means add?**
"It groups names into 40 zones. Zones with many risky pairs are crowded name neighbourhoods; a pharmacy could keep those medicines apart."

**What would you do next?**
"A cleaner ingredient source, pronunciation (Hinglish and regional accents), packaging similarity, and a trial in a pharmacy."
