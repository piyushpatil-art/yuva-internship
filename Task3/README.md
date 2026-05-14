# 🤖 Week 3 — Building and Tuning an AI Model

## Objective
Implement, train, tune, and evaluate machine learning models for predicting student pass/fail outcomes and final grade.

---

## 📄 Deliverables
| File | Description |
|---|---|
| `Week3_Model_Building_Tuning.ipynb` | Complete ML pipeline with code and visualisations |
| `Week3_Model_Building_Report.pdf` | Detailed report (1500+ words) |

---

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
jupyter lab Week3_Model_Building_Tuning.ipynb
```
> Execute all cells top-to-bottom. All data is generated automatically.

---

## 📌 Notebook Sections

| # | Section |
|---|---------|
| 1 | Setup & Imports |
| 2 | Data Loading & Preprocessing |
| 3 | Algorithm Selection & Justification |
| 4 | Baseline Model Training (4 algorithms) |
| 5 | Hyperparameter Tuning (GridSearchCV + RandomizedSearchCV) |
| 6 | Final Model Evaluation |
| 7 | Feature Importance & Visualisations |
| 8 | Regression Task (RMSE / R²) |
| 9 | Summary & Conclusion |

---

## 🏆 Models Trained

| Model | Tuning Method | Result |
|---|---|---|
| Logistic Regression | Baseline only | ~83% accuracy |
| Decision Tree | Baseline only | ~84% accuracy |
| **Random Forest** | GridSearchCV | **~90% accuracy** |
| **XGBoost** | RandomizedSearchCV | **~91% accuracy** |

---

## 📊 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC *(Classification)*
- MAE, RMSE, R² *(Regression)*

---

## 📊 Visualisations Included
- Baseline model comparison bar chart
- Tuning score distribution histograms
- Confusion matrices (both models)
- ROC curves
- Feature importance charts (top 15)
- Actual vs Predicted scatter plot (regression)

---

## 🛠️ Technologies Used
`Python` `Scikit-learn` `XGBoost` `Pandas` `NumPy` `Matplotlib` `Seaborn`

---

*Yuva Internship — Week 3 | YuvaIntern*
