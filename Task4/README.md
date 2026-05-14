# 🔍 Week 4 — Explainable AI and Model Interpretability

## Objective
Implement and analyse Explainable AI (XAI) techniques to interpret the predictions of the XGBoost student performance model.

---

## 📄 Deliverables
| File | Description |
|---|---|
| `Week4_Explainable_AI.ipynb` | Full XAI analysis notebook with visualisations |
| `Week4_Explainable_AI_Report.pdf` | Comprehensive report (2000+ words) |

---

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn xgboost shap lime matplotlib seaborn jupyter
jupyter lab Week4_Explainable_AI.ipynb
```
> All dependencies auto-install in the first cell. Execute top-to-bottom.

---

## 📌 Notebook Sections

| # | Section |
|---|---------|
| 1 | Setup & Imports |
| 2 | Data Loading & Model Training |
| 3 | SHAP — Global & Local Explanations |
| 4 | LIME — Instance-Level Explanations |
| 5 | Permutation Feature Importance |
| 6 | Partial Dependence Plots (PDP) |
| 7 | Cross-Technique Comparison |
| 8 | Real-World Application Discussion |
| 9 | Summary & Conclusion |

---

## 🧪 XAI Techniques Implemented

| Technique | Scope | Key Output |
|---|---|---|
| **SHAP TreeExplainer** | Global + Local | Beeswarm, waterfall, dependence plots |
| **LIME** | Local only | Feature weight bar charts per instance |
| **Permutation Importance** | Global | F1 drop per feature (30 repeats) |
| **Partial Dependence Plots** | Global | Marginal effect curves + 2D interaction |

---

## 📊 Visualisations Included
- SHAP beeswarm summary plot
- SHAP bar plot (mean absolute values)
- SHAP waterfall — passing student
- SHAP waterfall — at-risk student
- SHAP dependence plot (avg_grade × failures)
- LIME explanation — at-risk student
- LIME explanation — high-achieving student
- Permutation importance bar chart
- PDP for top 4 features
- 2D interaction PDP (avg_grade × failures)
- Cross-technique comparison chart

---

## 🔑 Key Finding
`avg_grade`, `failures`, and `absences` consistently rank in the **top 3** across all four XAI techniques — providing high-confidence, multi-validated feature importance.

---

## 🛠️ Technologies Used
`Python` `SHAP` `LIME` `XGBoost` `Scikit-learn` `Matplotlib` `Seaborn`

---

*Yuva Internship — Week 4 | YuvaIntern*
