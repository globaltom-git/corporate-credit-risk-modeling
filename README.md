
# Corporate Credit Risk Modeling with Explainable AI

**Bridging 20+ years of corporate finance, investment control & advisory with modern machine learning for interpretable default prediction.**

<img width="579" height="435" alt="Capture d&#39;écran 2026-01-14 160215" src="https://github.com/user-attachments/assets/490ad78f-dd23-459d-aeb7-c8440639c05c" />

*SHAP analysis: Fiscal year (economic cycle) and Industry classification (division and majorgroup) dominate default risk drivers, before financial metrics. 

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
  1. fyear, fiscal year (cycle/macro effects) is the dominant driver
  2. division, a macro industrial classification is a second dominant trigger
  3. majorgroup, a micro industrial classification     
  4. long term debts
  5. edbitda
  6. ebit  
  7. net sales 
  8. gross profit  
  9. total assets  
  10. total receivables  

**Insight** : fiscal year and Sector risk outweighs many financial ratios — consistent with real-world due diligence and PE practices.

## Repository Contents

- `notebooks/credit_risk_full_pipeline.ipynb` → End-to-end: preprocessing, modeling, SHAP analysis  
- `data/sample_data.csv` → Anonymized/synthetic sample (full dataset proprietary)  
- `plots/` → SHAP bar, beeswarm, feature importance visuals  
- `requirements.txt` → Dependencies (shap, lightgbm, pandas, scikit-learn, matplotlib)

## How to Run

```bash
git clone https://github.com/[ton-username]/corporate-credit-risk-modeling.git
cd corporate-credit-risk-modeling
pip install -r requirements.txt
jupyter lab notebooks/credit_risk_full_pipeline.ipynb
