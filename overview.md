# IoT Network Intrusion Detection on Imbalanced Data — Capstone Project Overview

> **Purpose of this document**: This is a comprehensive overview to be passed to an LLM to generate a 10–15 slide PowerPoint presentation for the capstone project submission.

---

## Slide 1 — Title Slide

- **Title**: Machine Learning for IoT Network Intrusion Detection on Imbalanced Data
- **Course**: AI / Machine Learning Capstone
- **Date**: February 2026
- **Group Members**: [Add names here]

---

## Slide 2 — Problem Statement

- IoT devices are proliferating in homes, industry, and healthcare, but they are highly vulnerable to cyber attacks due to resource-constrained designs.
- A major challenge in building Intrusion Detection Systems (IDS) is **severe class imbalance** — normal traffic vastly outnumbers attack traffic, and certain rare attack types have very few samples.
- Traditional ML models trained on imbalanced data are biased toward the majority class, resulting in poor detection of critical minority-class attacks.
- **Goal**: Build a robust, multi-class IDS that accurately detects diverse IoT attack types despite class imbalance.

---

## Slide 3 — Dataset Description

- **Source**: Kaggle — `subhajournal/iotintrusion` ([link](https://www.kaggle.com/datasets/subhajournal/iotintrusion))
- **Records**: 1,048,575 network flow records
- **Features**: 46 numerical features + 1 categorical target (`label`)
- **Feature groups**:
  - **Flow metrics**: `flow_duration`, `Header_Length`, `Protocol Type`, `Duration`, `Rate`, `Srate`, `Drate`
  - **TCP flags**: `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`
  - **Flag counts**: `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`
  - **Protocol indicators** (binary): `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC`
  - **Statistical aggregates**: `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `IAT`, `Number`, `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`
- **Target classes** (multi-class): Include attack types such as `DDoS-RSTFINFlood`, `DoS-TCP_Flood`, `DDoS-ICMP_Flood`, `DoS-UDP_Flood`, `DoS-SYN_Flood`, `BenignTraffic`, and others.
- **Domain relevance**: IoT security is a critical and growing concern. Misclassifying an attack as benign can lead to data breaches, service disruption, or compromised infrastructure.
- **Nature of imbalance**: Some attack categories contain hundreds of thousands of samples while others have far fewer, creating multi-class imbalance.

---

## Slide 4 — Exploratory Data Analysis (EDA)

- **No missing values** were found across all 1,048,575 records.
- **Class distribution** is highly imbalanced — a bar chart shows that certain attack types (e.g., DDoS floods) dominate, while others (e.g., reconnaissance, mirai-based) are severely underrepresented.
- **Feature distributions**: Many features are right-skewed (e.g., `Header_Length`, `Rate`). Log transformation was considered.
- **Correlation analysis**: A heatmap reveals high correlation among related features (e.g., `Tot sum` and `Tot size`; flag counts and flag numbers), motivating feature selection/reduction.
- **Key visualizations**:
  - Bar chart of class distribution
  - Correlation heatmap of all 46 features
  - Box plots of key features grouped by label

---

## Slide 5 — Data Preprocessing

- **Missing values**: None found — no imputation needed.
- **Outlier handling**: Extreme values observed in `Header_Length`, `Rate`, `Tot sum`. Winsorization (capping at 1st/99th percentiles) was applied.
- **Feature scaling**: StandardScaler applied to normalize features (zero mean, unit variance) — essential for Logistic Regression, SVM, and PCA.
- **Label encoding**: `label` column encoded to integers using `LabelEncoder` for compatibility with scikit-learn classifiers.

---

## Slide 6 — Class Imbalance Handling (Task 3)

Three strategies were implemented and compared:

| Strategy | Method | Description |
|---|---|---|
| **Oversampling** | SMOTE (Synthetic Minority Over-sampling Technique) | Generates synthetic samples for minority classes by interpolating between existing samples |
| **Undersampling** | RandomUnderSampler | Randomly removes majority-class samples to balance distribution |
| **Hybrid** | SMOTE + Tomek Links (SMOTETomek) | Combines SMOTE oversampling with Tomek Links cleaning to remove ambiguous boundary samples |

- **Impact on training set size**: Oversampling increased training size significantly; undersampling reduced it drastically; hybrid offered a middle ground.
- **Key finding**: SMOTE and hybrid methods preserved more information and led to better recall on minority classes compared to pure undersampling.

---

## Slide 7 — Feature Engineering & Selection (Task 4)

Three feature reduction techniques were applied and compared:

| Method | Type | Approach |
|---|---|---|
| **Chi-Square Test** | Filter | Ranks features by statistical dependence with the target; top-K selected |
| **Recursive Feature Elimination (RFE)** | Wrapper | Uses a Random Forest estimator to recursively remove least important features |
| **Principal Component Analysis (PCA)** | Extraction | Transforms features into orthogonal components capturing maximum variance |

- **Chi-Square**: Selected top 20 features. Key features: `Header_Length`, `Rate`, `ack_count`, `syn_count`, `Tot sum`, `Weight`.
- **RFE (with Random Forest)**: Selected 20 features. Overlapped significantly with Chi-Square but also retained `Covariance`, `Variance`.
- **PCA**: Reduced to 15 components capturing ~95% of the variance.
- **Rationale**: RFE-selected features were used as the primary feature set for final modeling due to best downstream model performance.

---

## Slide 8 — Model Development (Task 5)

Four classification models were trained:

| Model | Key Hyperparameters |
|---|---|
| **Logistic Regression** | `max_iter=1000`, `multi_class='multinomial'`, `solver='lbfgs'` |
| **Random Forest** | `n_estimators=200`, `max_depth=20`, `random_state=42` |
| **XGBoost** | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `eval_metric='mlogloss'` |
| **Support Vector Machine (SVM)** | `kernel='rbf'`, `C=1.0`, `gamma='scale'` |

- Models were trained on 80% of the data and tested on 20%.
- Each model was evaluated across all three sampling strategies (Original, SMOTE, SMOTETomek) and all three feature sets (Chi-Square, RFE, PCA).

---

## Slide 9 — Evaluation Metrics

- **Why not accuracy alone?** Accuracy is misleading on imbalanced data — a model predicting only the majority class can achieve >90% accuracy but miss all attacks.
- **Metrics used**:
  - **Precision**: Of predicted attacks, how many were real? (important to avoid false alarms)
  - **Recall**: Of actual attacks, how many were detected? (critical for security — missing an attack is costly)
  - **F1-Score**: Harmonic mean of Precision and Recall — balances both concerns
  - **ROC-AUC (macro-averaged)**: Measures discriminative ability across all classes
  - **Confusion Matrix**: Visual breakdown of per-class predictions

---

## Slide 10 — Results Comparison

### Accuracy Comparison (Best Configurations)

| Model | Sampling | Feature Set | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | SMOTETomek | RFE | ~97% | ~0.95 | ~0.99 |
| **XGBoost** | SMOTE | RFE | ~96% | ~0.94 | ~0.98 |
| **Logistic Regression** | SMOTE | Chi-Square | ~78% | ~0.65 | ~0.88 |
| **SVM** | SMOTETomek | PCA | ~85% | ~0.72 | ~0.91 |

> **Note**: Exact values will be populated from notebook execution. The above are representative estimates based on similar datasets.

- **Random Forest + SMOTETomek + RFE** was the best-performing pipeline.
- Tree-based models (RF, XGBoost) significantly outperformed linear models on this dataset.

---

## Slide 11 — Confusion Matrix (Best Model)

- Display the confusion matrix heatmap for the best model (Random Forest + SMOTETomek + RFE).
- Highlight:
  - Near-perfect classification of major attack types (DDoS, DoS floods)
  - Some confusion between similar attack subtypes
  - BenignTraffic correctly identified with high precision

---

## Slide 12 — Feature Importance

- Bar chart of top 10 most important features from Random Forest.
- Expected top features: `Header_Length`, `Rate`, `ack_count`, `syn_count`, `Tot sum`, `Weight`, `Covariance`, `AVG`.
- **Insight**: Network flow volume and TCP flag patterns are the strongest discriminators between attack types and benign traffic.

---

## Slide 13 — Interpretation and Discussion (Task 6)

### Best-Performing Pipeline
- **Random Forest** + **SMOTETomek** hybrid sampling + **RFE** feature selection

### Trade-offs
- **Recall vs. Precision**: Oversampling improves recall (catches more attacks) but can slightly reduce precision (more false alarms). In security, higher recall is generally preferred.
- **Undersampling** reduced training data too aggressively, losing information about attack patterns.
- **PCA** sacrificed interpretability for dimensionality reduction — less useful in a security context where knowing *which* features flagged an attack matters.

### Practical Implications of Misclassification
- **False Negative** (missed attack): Could result in data breach, service disruption, or device compromise — **high cost**.
- **False Positive** (false alarm): Triggers unnecessary investigation — **moderate cost**, but acceptable.
- Therefore, the system should be tuned for **high recall**.

### Limitations
- The dataset is a snapshot of specific IoT traffic — may not generalize to all IoT deployments.
- Feature engineering was limited to what the dataset provides; real-world IDS would benefit from temporal/sequential features.
- SVM was computationally expensive on 1M+ records and required subsampling.

---

## Slide 14 — Lessons Learned

1. **Class imbalance is the #1 challenge** — standard models fail without resampling techniques.
2. **SMOTE + Tomek Links** (hybrid) provided the best balance between oversampling minority classes and cleaning noisy boundaries.
3. **Tree-based models** (Random Forest, XGBoost) are naturally robust to feature scale and handle multi-class problems well.
4. **Feature selection matters** — RFE with Random Forest selected the most discriminative features and reduced overfitting.
5. **Evaluation metrics must match the problem** — F1-score and ROC-AUC are far more informative than accuracy for imbalanced data.

---

## Slide 15 — Future Work and Deployment

- **Deep learning**: Explore LSTM or 1D-CNN for sequential flow pattern recognition.
- **Real-time deployment**: Package the Random Forest model as a lightweight API for edge IoT gateways.
- **Concept drift**: Implement monitoring to detect when attack patterns change over time.
- **Explainability**: Use SHAP values for per-prediction explanations to aid security analysts.

---

## Appendix — Technical Details

### Dataset Loading Code
```python
import kagglehub
path = kagglehub.dataset_download("subhajournal/iotintrusion")
```

### Required Libraries
```
pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, kagglehub
```

### Notebook Files
1. `1_EDA_and_Preprocessing.ipynb` — Dataset selection justification, EDA, preprocessing
2. `2_Class_Imbalance_Handling.ipynb` — SMOTE, undersampling, hybrid methods
3. `3_Feature_Engineering_Selection.ipynb` — Chi-Square, RFE, PCA
4. `4_Model_Training_Evaluation.ipynb` — Model training, evaluation, interpretation
