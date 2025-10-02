# Anemia Detection Classifier: Multi-Model ML Pipeline

---

## Overview

Anemia detection is vital for early intervention in global health, affecting over 1.6 billion people worldwide. This project implements a binary classification pipeline to predict anemia presence (yes/no) from blood parameters using six classical machine learning models. Trained on a compact dataset of 1,421 samples, the models achieve up to 100% accuracy, with Gradient Boosting selected for inference—enabling quick, interpretable diagnostics for clinical screening tools.

**Project Context**: Developed during the **Smart Internz Virtual Training Internship** (July 14 - August 20, 2024), focusing on practical ML applications in healthcare.

---

## Dataset

- **Source**: Public Anemia Blood Test Dataset (inspired by Kaggle's anemia classification sets)
- **Size**: 1,421 samples × 6 features
- **Features**:
  - **Categorical**: Gender (0: Female, 1: Male)
  - **Numerical**: Hemoglobin (g/dL), MCH (pg), MCHC (g/dL), MCV (fL)
- **Target**: Result (0: No Anemia, 1: Anemia)
- **Characteristics**: Balanced classes (~45% anemia cases); no missing values; clean numerical data ready for modeling

---

## Methodology

### 1. Data Preprocessing
- Loaded via Pandas; verified integrity (no nulls, consistent dtypes)
- Feature matrix (X): All columns except 'Result'; Target (y): 'Result'
- Split: 80/20 train-test (stratified to preserve class balance)

### 2. Model Training
Evaluated six algorithms with default hyperparameters for baseline performance:
- **Logistic Regression**: Linear probabilistic classifier
- **Decision Tree**: Tree-based partitioning
- **Random Forest**: Ensemble of 100 decision trees
- **Naive Bayes (Gaussian)**: Probabilistic naive assumption on features
- **Support Vector Machine (SVC)**: Kernel-based margin maximization (linear kernel)
- **Gradient Boosting**: Boosted trees with iterative error correction

All models trained via Scikit-learn; evaluated on test set using accuracy and full classification reports.

### 3. Inference Example
- Sample Input: Female (0), Hemoglobin=11.6, MCH=22.3, MCHC=30.9, MCV=74.5
- Gradient Boosting Prediction: 1 (Anemia present)

---

## Results

| Model                | Test Accuracy | Key Notes (Precision/Recall/F1 for Class 1) |
|----------------------|---------------|--------------------------------------------|
| **Logistic Regression** | 96.77%      | 0.97 / 0.96 / 0.96                        |
| **Decision Tree**    | 100.00%     | 1.00 / 1.00 / 1.00                        |
| **Random Forest**    | 100.00%     | 1.00 / 1.00 / 1.00                        |
| **Naive Bayes**      | 97.98%      | 0.97 / 0.99 / 0.98                        |
| **SVM**              | 93.95%      | 0.91 / 0.99 / 0.95                        |
| **Gradient Boosting**| 100.00%     | 1.00 / 1.00 / 1.00                        |

*Note*: Perfect scores suggest potential overfitting on this small dataset; real-world validation recommended. All models exhibit strong recall for anemia cases (>95%).

---

## Key Highlights

- **High Fidelity**: Ensemble methods (RF, GB) deliver flawless classification, ideal for high-stakes medical triage
- **Simplicity & Speed**: Models train in seconds on CPU; no complex preprocessing needed
- **Practical Utility**: Single-sample prediction via GB enables point-of-care apps—e.g., input blood metrics for instant risk assessment

---

## Future Enhancements

- Address overfitting with cross-validation, regularization, or SMOTE augmentation
- Integrate explainability (e.g., SHAP values) for feature importance in clinical reports
- Expand to multi-class (e.g., iron-deficiency vs. thalassemia) using larger datasets
- Deploy as a Streamlit/Flask web app for user-friendly blood test uploads

---

## Tech Stack

- **Language**: Python 3.12+
- **Data/ML**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn (for EDA plots)
- **Environment**: Jupyter Notebook

---

## Impact

This classifier democratizes anemia screening by leveraging accessible blood metrics, potentially aiding resource-limited settings where lab access is scarce. With 100% test accuracy on key models, it supports proactive healthcare—reducing undetected cases and enabling timely interventions. Repository includes full notebook for replication; extendable for broader hematology AI applications.thcare—reducing undetected cases and enabling timely interventions. Repository includes full notebook for replication; extendable for broader hematology AI applications.
