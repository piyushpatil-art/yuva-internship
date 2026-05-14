# ⚖️ Week 6 — AI Ethics, Bias Analysis and Fairness Assessment

## Objective
Conduct a comprehensive ethical audit of the student performance prediction model, identifying biases across protected demographic groups and proposing remediation strategies.

---

## 📄 Deliverables
| File | Description |
|---|---|
| `Week6_Bias_Fairness_Analysis.ipynb` | Full bias analysis notebook with visualisations |
| `Week6_Ethics_Bias_Report.pdf` | Comprehensive audit report (2000+ words) |

---

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
jupyter lab Week6_Bias_Fairness_Analysis.ipynb
```
> Execute all cells top-to-bottom. Dataset loads automatically.

---

## 📌 Notebook Sections

| # | Section |
|---|---------|
| 1 | Setup & Data Loading |
| 2 | Exploratory Fairness EDA |
| 3 | Disparate Impact Analysis |
| 4 | Demographic Parity |
| 5 | Equalized Odds |
| 6 | Predictive Parity |
| 7 | Intersectional Bias Analysis |
| 8 | Feature Contribution Bias (SHAP) |
| 9 | Bias Mitigation — Sample Reweighing |
| 10 | Before vs After Comparison |
| 11 | Summary & Recommendations |

---

## 🔎 Protected Attributes Analysed

| Attribute | Values | Ethical Concern |
|---|---|---|
| **Sex** | Male / Female | Gender equity in education |
| **Address** | Urban / Rural | Socioeconomic resource disparity |
| **Parent Education** | Low / Medium / High | Household academic capital |

---

## 📐 Fairness Metrics Used

| Metric | Threshold | Formula |
|---|---|---|
| Disparate Impact | DI < 0.80 = concern (4/5 rule) | P(Ŷ=1\|minority) / P(Ŷ=1\|majority) |
| Demographic Parity | Gap > 0.10 = concern | \|P(Ŷ=1\|A) − P(Ŷ=1\|B)\| |
| Equalized Odds | TPR/FPR gap > 0.10 = concern | Equal TPR and FPR across groups |
| Predictive Parity | Precision gap > 0.10 = concern | Equal precision across groups |

---

## 📊 Visualisations Included
- Pass rate bar charts by protected group (Fig 1)
- Actual vs predicted pass rates comparison (Fig 2)
- Predicted probability distributions (Fig 3)
- Disparate impact ratio chart with 4/5 threshold line (Fig 4)
- Equalized odds heatmap — TPR, FPR, Precision (Fig 5)
- Intersectional heatmaps — Sex × Address, Sex × Parent Edu (Fig 6)
- SHAP feature importance with protected attribute highlighting (Fig 7)
- Before vs after reweighing comparison (Fig 8)

---

## 🛠️ Mitigation Applied
**Sample Reweighing** — assigns higher training weights to underrepresented (disadvantaged group + positive outcome) combinations, improving fairness with only ~2% accuracy trade-off.

---

## 🔑 Key Finding
**Parental education** shows the most significant fairness violation — a demographic parity gap of ~0.23 and a TPR gap of ~0.13 for students from low-education households.

---

## 🛠️ Technologies Used
`Python` `Scikit-learn` `XGBoost` `SHAP` `Pandas` `NumPy` `Matplotlib` `Seaborn`

---

*Yuva Internship — Week 6 | YuvaIntern*
