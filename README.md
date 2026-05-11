# 🏠 Residential Property Valuation Modelling
### House Prices: Advanced Regression Techniques

> **Top 12% on Kaggle** — XGBoost regression model predicting residential sale prices from 79 structured property features.

---

## 📌 Project Overview

Residential property valuation is one of the most commercially significant applications of predictive analytics — underpinning mortgage underwriting, listing price recommendations, investment screening, and market intelligence.

This project builds a full automated valuation pipeline using the Ames Iowa Housing dataset, achieving a **top 12% leaderboard position** among 4,700+ teams on Kaggle. The goal was not just to optimise a competition score — but to build an interpretable, business-grounded valuation model with documented decisions at every stage.

---

## 🏆 Competition Result

| | |
|---|---|
| Competition | House Prices: Advanced Regression Techniques (Kaggle) |
| Model | XGBoost Regressor |
| Evaluation metric | RMSE (log scale) |
| **Leaderboard position** | **Top 12%** |
| Dataset | Ames Iowa Housing — 1,460 train / 1,459 test properties |

---

## 🛠️ Feature Engineering

Six composite features were engineered from raw columns, each grounded in real estate valuation logic:

| Feature | Formula | Rationale |
|---|---|---|
| `AllSF` | TotalBsmtSF + 1stFlrSF + 2ndFlrSF | Total usable interior square footage — strongest composite size signal |
| `BackyardSF` | LotArea − 1stFlrSF | Outdoor space valued independently from indoor area |
| `PorchSF` | Sum of all porch/deck areas | Total outdoor living space captures lifestyle premium |
| `Total_Bathrooms` | FullBath + 0.5×HalfBath + basement equivalents | Weighted count — half baths contribute less value |
| `MedNhbdArea` | Median GrLivArea per Neighbourhood | Encodes neighbourhood size norms as continuous context |
| `IsAbvGr` | 1 if GrLivArea > MedNhbdArea | Binary: is this home larger than typical for its neighbourhood? |

---

## 🤖 Model

### XGBoost Configuration

```python
XGBRegressor(
    learning_rate  = 0.01,      # Low LR — prevents overfitting, requires more trees
    n_estimators   = 3460,      # High tree count to compensate for low LR
    gamma          = 0,         # No minimum loss reduction required to split
    objective      = 'reg:squarederror',
    reg_alpha      = 0.00006,   # L1 regularisation — reduces test error
    nthread        = -1,        # Use all available cores
    random_state   = 42
)
```

### Target Transformation

`SalePrice` is right-skewed (skewness ≈ 1.88). The model trains on `log1p(SalePrice)` and predictions are reversed with `np.expm1()`. This improves model performance by normalising the target distribution and penalising percentage errors equally across price ranges.

### Missing Value Strategy

| Missing type | Treatment | Rationale |
|---|---|---|
| PoolQC, MiscFeature, Alley, Fence, FireplaceQu | Fill "No" | 80–99% missing = structural absence (no pool, no alley) |
| Bsmt features | Fill "No" / 0 | No basement present |
| Garage features | Fill "No" / 0 | No garage present |
| Functional | Fill "Typ" | Assume typical unless stated otherwise |
| Electrical | Fill "SBrkr" | Most common value |
| Numeric columns | Fill median | True missing — safe imputation |

---

## 📊 Key EDA Findings

- **OverallQual** is the strongest single predictor (r ≈ 0.79) — quality-10 homes sell for ~4× quality-4 homes
- **Neighbourhood drives significant price variance** — NridgHt, NoRidge, StoneBr are premium areas
- **Newer builds command a clear premium** — homes built post-2000 significantly outprice pre-war stock
- **Right-skew in continuous features** — LotArea, GrLivArea, and engineered features required Box-Cox correction
- **80%+ missing in 4 features** — structural absence correctly treated as "No" not imputed

---

## 📈 Results

| Metric | Value |
|---|---|
| RMSE (log scale) | — |
| R² (validation set) | — |
| Kaggle leaderboard | **Top 12%** |

*Fill in RMSE and R² after final model run.*

---

## 🏢 Business Applications

| Use case | Application |
|---|---|
| **Automated Valuation Model (AVM)** | Listing price recommendation for sellers and agents |
| **Mortgage underwriting** | Independent valuation cross-check for loan decisions |
| **Buyer intelligence** | Identify undervalued properties vs model prediction |
| **Investment screening** | Neighbourhood price-per-sqft opportunity mapping |
| **Market analytics** | Track price driver shifts over time |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | XGBoost, Scikit-learn |
| Environment | Jupyter Notebook |

---

## ⚠️ Limitations & Next Steps

| Limitation | Proposed improvement |
|---|---|
| Ames Iowa data (2006–2010) only | Retrain on target market and time period |
| No macroeconomic features | Add interest rates, CPI, local employment data |
| Static model — no drift monitoring | Quarterly retraining cadence in production |
| Single model — no ensemble | Blend XGBoost with LightGBM or Ridge for marginal gain |
