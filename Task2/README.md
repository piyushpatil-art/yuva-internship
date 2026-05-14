# 🧹 Week 2 — Data Preprocessing and Feature Engineering

## Objective
Design and document a comprehensive data preprocessing and feature engineering pipeline to prepare the student dataset for machine learning model training.

---

## 📄 Deliverable
| File | Description |
|---|---|
| `Week2_Data_Preprocessing_Feature_Engineering.ipynb` | Fully documented Jupyter Notebook |

---

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter lab Week2_Data_Preprocessing_Feature_Engineering.ipynb
```
> Execute all cells top-to-bottom using `Shift+Enter`. Dataset loads automatically.

---

## 📌 Notebook Sections

| # | Section |
|---|---------|
| 1 | Environment Setup & Library Imports |
| 2 | Dataset Loading & Initial Exploration |
| 3 | Data Cleaning & Missing Value Treatment |
| 4 | Exploratory Data Analysis (EDA) |
| 5 | Outlier Detection & Treatment |
| 6 | Encoding Categorical Variables |
| 7 | Feature Scaling & Normalisation |
| 8 | Feature Engineering |
| 9 | Feature Selection & Correlation Analysis |
| 10 | Final Pipeline & Train-Test Split |
| 11 | Summary & Conclusion |

---

## ⚙️ Feature Engineering (6 New Features)

| Feature | Formula | Rationale |
|---|---|---|
| `avg_grade` | (G1 + G2) / 2 | Smoother performance indicator |
| `grade_trend` | G2 − G1 | Captures academic trajectory |
| `avg_parent_edu` | (Medu + Fedu) / 2 | Socioeconomic signal |
| `alcohol_exposure` | Dalc + Walc | Composite risk factor |
| `support_score` | schoolsup + famsup | Combined support indicator |
| `is_at_risk` | failures>0 AND absences>median | Binary compound risk flag |

---

## 📊 Visualisations Included
- Missing value heatmap
- Grade distribution histograms (G1, G2, G3)
- Categorical feature box plots vs G3
- Scaler comparison chart
- Engineered feature scatter plots
- Correlation heatmap

---

## 🛠️ Technologies Used
`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn`

---

*Yuva Internship — Week 2 | YuvaIntern*
