# 💊 Drug Classification using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A supervised machine learning project that predicts the most suitable drug type for a patient based on clinical features such as age, sex, blood pressure, cholesterol level, and sodium-to-potassium ratio.

---

## 📌 Business Problem

Healthcare professionals need an efficient way to prescribe the right drug for a patient. This project builds a multi-class classification model trained on patient diagnostic data to predict which of five drug types (DrugA, DrugB, DrugC, DrugX, DrugY) is most appropriate for a given patient.

---

## 📂 Dataset

| Feature | Description | Sample Values |
|---|---|---|
| `Age` | Patient age | 23, 47, ... |
| `Sex` | Gender | M, F |
| `BP` | Blood pressure level | HIGH, NORMAL, LOW |
| `Cholesterol` | Cholesterol level | HIGH, NORMAL |
| `Na_to_K` | Sodium-to-potassium ratio in blood | 25.355, 13.093, ... |
| `Drug` | **Target** — Drug type | DrugY, DrugC, DrugA, DrugB, DrugX |

- **Source:** [UCI Machine Learning Repository – Drug200](https://www.kaggle.com/datasets/prathamtripathi/drug-classification)
- **Size:** 200 rows × 6 columns
- **Missing values:** None

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

- **DrugY** is the most frequently prescribed drug (91 out of 200 patients), creating a class imbalance.
- Gender distribution is balanced: 104 Male, 96 Female.
- Blood pressure is evenly spread across HIGH (77), LOW (64), and NORMAL (59).
- `Age` distribution is symmetric (skewness ≈ 0.03).
- `Na_to_K` is moderately right-skewed (skewness ≈ 1.04).

Visualizations include:
- Drug type distribution (count plot)
- Gender, BP, and Cholesterol distribution
- Gender vs Drug Type crosstab bar chart
- BP vs Cholesterol crosstab bar chart
- Na_to_K vs Age scatter plot by gender

---

## ⚙️ Data Preparation

### Feature Engineering

**Age Binning** — continuous age converted to 7 categories:
```
<20 | 20s | 30s | 40s | 50s | 60s | >60s
```

**Na_to_K Binning** — continuous ratio converted to 4 categories:
```
<10 | 10-20 | 20-30 | >30
```

**One-Hot Encoding** applied to all categorical features using `sklearn.preprocessing.OneHotEncoder`.

### Train-Test Split
- **70% Training / 30% Test**
- `random_state=0` for reproducibility

### Class Imbalance Handling
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on training data to balance the drug class distribution and prevent model bias toward DrugY.

---

## 🤖 Models Trained

| Model | Accuracy |
|---|---|
| Logistic Regression | **85.00%** |
| Support Vector Machine (SVM) | **85.00%** |
| Gaussian Naive Bayes | **85.00%** |
| Categorical Naive Bayes | 83.33% |
| K-Nearest Neighbours (KNN) | 61.67% |

> **Final Model Selected:** Logistic Regression (solver: `liblinear`, max_iter: 5000)
> Saved using `pickle` as `Finalmodel.pk1` along with the encoder `Finalencoder.pk1`.

---

## 🧪 Sample Prediction

```python
import pickle
import numpy as np

model = pickle.load(open('Finalmodel.pk1', 'rb'))
enc   = pickle.load(open('Finalencoder.pk1', 'rb'))

# Patient: Male, Normal BP, Normal Cholesterol, Age 40s, Na_to_K >30
test_input = np.array([['M', 'NORMAL', 'NORMAL', '40s', '>30']])
encoded    = enc.transform(test_input)
prediction = model.predict(encoded)

print(prediction)  # Output: ['drugY']
```

---

## 🗂️ Repository Structure

```
drug-classification/
│
├── drug200.csv                  # Dataset
├── Drug_Classification.ipynb    # Main Jupyter Notebook
├── Finalmodel.pk1               # Saved Logistic Regression model
├── Finalencoder.pk1             # Saved OneHotEncoder
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/drug-classification.git
cd drug-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook Drug_Classification.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
jupyter
```

> See `requirements.txt` for version-pinned dependencies.

---

## 🔮 Future Improvements

- [ ] Try Decision Tree and Random Forest classifiers
- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Replace deprecated `sns.distplot` with `sns.histplot`
- [ ] Fix SMOTE being applied before the train-test split (data leakage risk)
- [ ] Build a simple web UI using Streamlit or Flask for live predictions
- [ ] Add cross-validation for more robust evaluation

---

## 👤 Author

**Nivedh J**
BCA Graduate — Mahatma Gandhi University, Palghat, Kerala

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/nivedhj/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/nivedh-j)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
