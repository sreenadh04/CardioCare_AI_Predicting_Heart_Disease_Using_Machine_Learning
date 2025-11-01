# 🩺 Predicting Heart Disease Severity with Machine Learning

### Project Overview

This project implements an end-to-end machine learning workflow to predict the severity of heart disease. The objective is to classify patients into one of five categories (0 for no disease, and 1-4 for increasing severity) based on a set of 13 clinical features.

This repository demonstrates a complete data science process, including:
* In-depth Exploratory Data Analysis (EDA)
* Robust data preprocessing using `scikit-learn` Pipelines to handle significant missing data and mixed data types.
* Comparative analysis of four distinct classification models.
* Honest evaluation of model performance on a highly imbalanced, real-world dataset.

---

### 📊 Dataset

The analysis was performed on a composite heart disease dataset containing 920 patient records and 16 columns.

* **Features:** 13 clinical features, including `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), and `ca` (number of major vessels).
* **Target Variable:** `num`, an ordinal variable representing the severity of heart disease (0, 1, 2, 3, 4).
* **Data Quality Challenges:**
    * **Significant Missing Data:** The dataset contained 1,759 missing values in total, affecting key predictors like `ca`, `thal`, and `slope`.
    * **Severe Class Imbalance:** The target variable is heavily skewed towards class 0 (no disease), with classes 2, 3, and 4 being extremely rare. This presents a major challenge for model training and evaluation.

---

### ⚙️ Methodology

#### 1. Exploratory Data Analysis (EDA)

Initial analysis confirmed the data quality issues and revealed relationships between key features and the target variable. For instance, `thalch` (max heart rate) tended to be lower for patients with heart disease, and specific `cp` (chest pain types) were strongly correlated with the presence of disease. The severe class imbalance was identified as the primary challenge to address.

#### 2. Robust Preprocessing Pipeline

To ensure reproducibility and prevent data leakage, a `ColumnTransformer` and `Pipeline` architecture was created. This process systematically handles all preprocessing steps.

* **Numerical Features (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`, `ca`):**
    1.  **Imputation:** Missing values were filled using `SimpleImputer` with the `mean` strategy.
    2.  **Scaling:** Features were scaled using `StandardScaler` to normalize their range.

* **Categorical Features (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`):**
    1.  **Imputation:** Missing values were filled using `SimpleImputer` with the `most_frequent` strategy.
    2.  **Encoding:** Features were converted into numerical format using `OneHotEncoder` (dropping the first category to avoid multicollinearity).

#### 3. Model Training and Comparison

The dataset was split into training (80%) and testing (20%) sets. Crucially, `stratify=y` was used to ensure the imbalanced class distribution was preserved in both sets.

Four different models were trained using the complete preprocessing pipeline:
1.  **Logistic Regression** (as a baseline)
2.  **Random Forest Classifier**
3.  **Support Vector Machine (SVM)**
4.  **K-Nearest Neighbors (KNN)**

---

### 📈 Results and Evaluation

Model performance was evaluated using the classification report, focusing on accuracy and F1-scores due to the class imbalance.

| Model | Accuracy | Weighted Avg F1-Score | Macro Avg F1-Score |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.58 | 0.56 | 0.34 |
| Random Forest | 0.56 | 0.54 | 0.32 |
| **Support Vector Machine (SVM)** | **0.59** | **0.56** | **0.33** |
| K-Nearest Neighbors (KNN) | 0.58 | 0.55 | 0.32 |

#### Analysis
* The **Support Vector Machine (SVM)** achieved the highest overall accuracy at 59%.
* All models demonstrated modest performance, which is expected given the dataset's challenges.
* The significant gap between the **Weighted Avg F1-Score** (influenced by the majority class 0) and the **Macro Avg F1-Score** (which treats all classes equally) highlights the models' inability to effectively predict the rare minority classes (2, 3, and 4).
* The SVM's confusion matrix confirmed it was reasonably effective at identifying 'No Disease' (class 0) but struggled to differentiate between the various levels of disease severity.

---

### 🎯 Feature Importance

Feature importance was extracted from the Random Forest model to understand which clinical factors were most predictive.

**Top 10 Most Important Features:**
1.  **`ca`** (number of major vessels)
2.  **`thal_3`** (Thalassemia: reversible defect)
3.  **`thalch`** (max heart rate achieved)
4.  **`oldpeak`** (ST depression)
5.  **`cp_3`** (Chest Pain: asymptomatic)
6.  **`age`**
7.  **`chol`** (cholesterol)
8.  **`trestbps`** (resting blood pressure)
9.  **`exang_1`** (exercise-induced angina)
10. **`sex_1`** (Male)

This aligns with medical intuition, confirming that factors like the number of blocked vessels, thalassemia test results, and max heart rate are critical indicators.

---

### 🏁 Conclusion and Next Steps

This project successfully built a complete and robust machine learning pipeline to tackle a challenging, imbalanced medical dataset with significant missing values. The analysis demonstrated that while a simple model can achieve ~59% accuracy, predictive power is severely limited by the data's quality and imbalance.

**Future work to improve performance would include:**
* **Addressing Imbalance:** Implementing advanced over-sampling techniques like **SMOTE** or **ADASYN** on the training data to create more examples of the minority classes.
* **Hyperparameter Tuning:** Using `GridSearchCV` or `RandomizedSearchCV` to optimize the parameters for the top-performing SVM model.
* **Problem Re-framing:** Simplifying the problem from a 5-class classification to a binary (disease vs. no disease) or 3-class (no disease, low-severity, high-severity) problem, which may yield a more practical and accurate model.
