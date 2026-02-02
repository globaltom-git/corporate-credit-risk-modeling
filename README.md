
# Corporate Credit Risk Modeling with Explainable AI

**Bridging 20+ years of corporate finance, investment control & advisory with modern machine learning for interpretable default prediction.**

![image(1)](https://github.com/user-attachments/assets/7f2b7f16-7118-4968-9def-6b0fcc239fd2)

*SHAP analysis: retained earnings, fiscal year (economic cycle) dominate default risk drivers, before Ebit and macro sectorial classification.* 

## Business Context

This project applies machine learning to predict corporate default risk using a large-scale dataset of **80,000 firm-year observations** from Nasdaq & NYSE-listed companies (2000–2018), with **20+ financial indicators** and detailed industry classifications.

**Relevance to my career**  
- Cross-border due diligence & private equity screening (LJ Advisory, Global Equity, Edgar Brandt)  
- Investment control & project governance (Engie E&P Roemerberg, Petroplus)  
- Risk management in high-stakes environments (Allianz, Cap Gemini)  

The focus is on **explainable models** (SHAP values) to support regulated credit decisions, portfolio monitoring, governance, and strategic advisory — where interpretability is essential for board-level trust and regulatory compliance (ECB/EBA standards).

## Key Results

- **Models** : Clustering, Random Forest, LightGBM, Neural Networks, SMOTE for class imbalance  
- **Metrics** : AUC / Precision-Recall emphasis (imbalanced defaults)  
- **Top SHAP Drivers** (mean absolute values) :
1. Retained Earnings (accumulated undistributed profits) → by far the #1 driver
2. Fiscal year → macro cycle effect is massive
3. EBIT
4. Division (broad macro-sector)
5. Total long-term debts
6. EBITDA
7. Total revenue
8. Net income
9. Current assets
10. Inventory 

**Insight** : retained earnnings and fiscal year outweighs many financial ratios — consistent with real-world due diligence and PE practices.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/globaltom-git/corporate-credit-risk-modeling/blob/master/credit_risk_full_pipeline.ipynb)

## Repository Contents

- `notebooks/credit_risk_full_pipeline.ipynb` → End-to-end: preprocessing, modeling, SHAP analysis  
- `data/american_bankruptcy_dataset_enriched_sample.csv` → Anonymized/synthetic sample (full dataset proprietary)  
- `plots/` → SHAP bar, beeswarm, feature importance visuals  
- `requirements.txt` → Dependencies (shap, lightgbm, pandas, scikit-learn, matplotlib)
- `.gitignore`
- `License`

## How to Run

```bash
git clone https://github.com/globaltom-git/corporate-credit-risk-modeling.git
cd corporate-credit-risk-modeling
pip install -r requirements.txt
jupyter lab notebooks/credit_risk_full_pipeline.ipynb
