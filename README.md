# Corporate Credit Risk Modeling with Explainable AI

Predicting corporate default from published accounts — and being able to explain every prediction, line by line.

Built on 78,682 annual filings from 8,971 companies listed on Nasdaq and the NYSE between 1999 and 2018, of which 609 failed (6.8%).

---

## Headline results

Validation is strictly out-of-time, following the protocol recommended by the dataset's authors: train on 1999–2011, tune on 2012–2014, test on 2015–2018. The model never sees the years it is judged on.

| Model | Ranking accuracy (AUC) | Failures caught in the riskiest 20% |
|---|---|---|
| **Gradient boosting (LightGBM)** | **0.861** | **76 %** |
| Random Forest | 0.842 | 73 % |
| Logistic regression | 0.795 | 66 % |
| Clustering with rules | 0.734 | 78 % |
| Neural network | 0.716 | 56 % |

With 119 failures in the test period, the margin of error on each figure is roughly ±3 points — the gap between the first three models is not statistically meaningful.

**In practice:** reviewing the 20% of files ranked riskiest surfaces 76% of the companies that go on to fail within three years. Within that set, about one file in twelve is genuinely heading for default, against one in forty-eight picked at random.

---

## What the model looks at

The strongest separations between companies that failed within three years and the rest (median values, test period):

| Indicator | Healthy | Failing within 3 years |
|---|---|---|
| Altman Z-score | 3.42 | −0.01 |
| Market value / book value | 2.68 | 0.51 |
| Retained earnings / assets | −0.03 | −0.78 |
| Return on assets | +1 % | −21 % |
| Debt / assets | 55 % | 90 % |
| Current ratio | 1.87 | 1.22 |

The Altman Z-score result is an external check worth noting: the formula dates from 1968 and places its distress zone below 1.8. The two populations fall either side of that threshold without any fitting on our part.

Beyond levels, the pipeline builds **trajectory indicators** — how each ratio moves over one, two and three years, its volatility, consecutive earnings declines, erosion of accumulated reserves. These are computed for every company in the panel, healthy and failing alike, and they carry a signal that a single-year snapshot cannot.

---

## Pipeline

Four scripts, run in order. Each ends with a block of automated checks; **if a check fails, no output file is written.**

```bash
pip install -r requirements.txt

python 01_build_dataset.py      # raw data -> v8_base.csv
python 02_features.py           # labelling + 99 indicators -> v8_features.csv
python 03_models.py             # five model families, tuning, calibration
python 04_graphiques.py         # five publication charts
```

### 01 — Building the dataset

The public dataset ships with anonymised column names (`X1`…`X18`). This step applies the correct mapping and repairs two defects in the source file:

- **Column mapping.** Established by two independent routes: the named columns in the paper's companion files, and a blind search over all 4,896 possible triples for accounting identities. Under the correct mapping, EBITDA − D&A = EBIT holds on 97.9% of rows; under a wrong one, 0.3%.
- **Thousands separators.** About 22% of values are stored as `701.854` instead of `701854`. Repairing them lifts the accounting identities from ~57% to ~98%.
- **Units.** Market capitalisation is expressed in millions while accounting items are in thousands.

Verification gate: three accounting identities above 95%, zero impossible negatives on balance-sheet items, row and company counts, date range, and a market-to-book sanity check.

### 02 — Labelling and indicators

The data records *whether* a company failed, not *when*. Applied to a company's whole history, that flag would ask the model "will this firm fail one day?". It is therefore restricted to the final three financial years, so the question becomes **"will it default within three years?"**

Verification gate: no healthy company flagged, no failing company with more than three flagged years, flagged years are the most recent, ratio medians within plausible ranges, trajectory indicators present on healthy companies, no infinite values.

### 03 — Models

Five families compared, tuned on the validation period. Class imbalance handled by synthetic over-sampling of the training set — tested both ways rather than assumed (`--smote` flag): it improves ranking by roughly two points while slightly degrading absolute probabilities.

Probabilities are calibrated by isotonic regression on the validation period, so a stated 10% corresponds to an observed 10%.

Verification gate: periods disjoint and ordered, every model better than chance, calibrated mean close to the observed rate.

### 04 — Charts

Five figures: indicator importance, direction of effect, cumulative gains, calibration, and a single-company decomposition.

---

## Repository contents

```
01_build_dataset.py        step 1 — mapping, repairs, verification
02_features.py             step 2 — labelling, ratios, Altman Z, trajectory
03_models.py               step 3 — five model families, tuning, calibration
04_graphiques.py           step 4 — publication charts
sample_500_companies.csv   500-company extract, correct column names
requirements.txt
```

Intermediate files (`v8_base.csv`, `v8_features.csv`) are not committed: they are rebuilt in seconds by steps 1 and 2.

**Source data.** The full dataset is public: Pellegrino, Lombardo, Adosoglou, Cagnoni, Pardalos & Poggi, *Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks*, Future Internet, MDPI, 2022. Step 1 locates `american_bankruptcy_dataset.csv` automatically.

---

## Limitations

US-listed companies, 1999–2018 — nothing here bears on later events, and risk behaves differently in other markets or for unlisted mid-caps. The date of failure is approximated by the last published financial year, and the three-year horizon rests on that proxy. Sector rates are frequencies observed over the period, not forecasts. Management quality, governance, litigation and customer concentration are absent from the data and are frequently decisive.

This is a methodological exercise, not a production credit model.

---

## License

MIT.
