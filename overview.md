# IIoT Edge Computing Predictive Maintenance — Capstone Project Overview

> **Purpose of this document**: This is a comprehensive overview to be passed to an LLM to generate a 10–15 slide PowerPoint presentation for the capstone project submission.

---

## Slide 1 — Title Slide

- **Title**: Machine Learning for IIoT Edge Computing Predictive Maintenance
- **Course**: AI / Machine Learning Capstone
- **Date**: February 2026
- **Group Members**: [Add names here]

---

## Slide 2 — Problem Statement

- Industrial IoT (IIoT) systems are critical for real-time monitoring and control in manufacturing environments, but equipment failures can lead to costly downtime and safety hazards.
- A major challenge in building predictive maintenance systems is **class imbalance** — normal operation vastly outnumbers failure events, making it difficult for ML models to detect rare but critical failure patterns.
- Traditional ML models trained on imbalanced data are biased toward the majority class, resulting in poor detection of impending failures.
- **Goal**: Build a robust binary classifier that accurately predicts equipment failures in IIoT edge computing environments despite class imbalance.

---

## Slide 3 — Dataset Description

- **Source**: Kaggle — `ziya07/iiot-edge-computing-dataset` ([link](https://www.kaggle.com/datasets/ziya07/iiot-edge-computing-dataset))
- **Records**: 1,000 sensor readings from industrial equipment
- **Features**: 9 features + 1 binary target (`Predicted_Failure`)
- **Feature groups**:
  - **Sensor readings**: `Temperature` (60-100°C), `Pressure` (90-120 units), `Vibration`
  - **Network metrics**: `Network_Latency`, `Edge_Processing_Time`
  - **Control system**: `Fuzzy_PID_Output` (0.5-1.0)
  - **Status indicators**: `Sensor_ID` (S1-S5), `Maintenance_Status` (Normal/Warning/Failure)
  - **Temporal**: `Timestamp`
- **Target classes** (binary):
  - Class 0: No Failure (546 samples, 54.6%)
  - Class 1: Failure (454 samples, 45.4%)
- **Domain relevance**: Predictive maintenance is critical for industrial operations. Missing a failure prediction can lead to catastrophic equipment breakdown, production downtime, and safety incidents.
- **Nature of imbalance**: Moderate 1.2:1 ratio — provides opportunity to demonstrate resampling techniques while remaining manageable for analysis.

---

## Slide 4 — Exploratory Data Analysis (EDA)

- **No missing values** were found across all 1,000 records.
- **Class distribution** shows moderate imbalance — 54.6% no failure vs 45.4% failure cases.
- **Feature distributions**: Sensor readings exhibit normal distributions around their operating ranges (Temperature: 60-100°C, Pressure: 90-120 units).
- **Correlation analysis**: A heatmap reveals moderate correlations among sensor readings, with no extreme multicollinearity issues.
- **Key visualizations**:
  - Bar chart and pie chart of class distribution
  - Histograms of all 6 numerical features
  - Correlation heatmap
  - Box plots for outlier detection

---

## Slide 5 — Data Preprocessing

- **Missing values**: None found — no imputation needed.
- **Outlier handling**: Extreme values observed in sensor readings. Winsorization (capping at 1st/99th percentiles) was applied.
- **Categorical encoding**: `Sensor_ID` and `Maintenance_Status` encoded using `LabelEncoder`.
- **Feature scaling**: StandardScaler applied to normalize numerical features (zero mean, unit variance) — essential for Logistic Regression and PCA.
- **Target variable**: `Predicted_Failure` is already binary encoded (0/1).

---

## Slide 6 — Class Imbalance Handling (Task 3)

Three strategies were implemented and compared:

| Strategy | Method | Description |
|---|---|---|
| **Oversampling** | SMOTE (Synthetic Minority Over-sampling Technique) | Generates synthetic samples for minority class by interpolating between existing samples |
| **Undersampling** | RandomUnderSampler | Randomly removes majority-class samples to balance distribution |
| **Hybrid** | SMOTE + Tomek Links (SMOTETomek) | Combines SMOTE oversampling with Tomek Links cleaning to remove ambiguous boundary samples |

- **Impact on training set size**: Oversampling increased training size; undersampling reduced it; hybrid offered a middle ground.
- **Key finding**: Even with moderate imbalance, SMOTE and hybrid methods improved Macro F1-score by better balancing precision and recall, particularly improving recall for the failure class.

---

## Slide 7 — Feature Engineering & Selection (Task 4)

Three feature reduction techniques were applied and compared:

| Method | Type | Approach |
|---|---|---|
| **Chi-Square Test** | Filter | Ranks features by statistical dependence with the target; top-K selected |
| **Recursive Feature Elimination (RFE)** | Wrapper | Uses a Random Forest estimator to recursively remove least important features |
| **Principal Component Analysis (PCA)** | Extraction | Transforms features into orthogonal components capturing maximum variance |

- **Chi-Square**: Selected top 5 features: `Temperature`, `Vibration`, `Edge_Processing_Time`, `Fuzzy_PID_Output`, `Maintenance_Status_encoded`.
- **RFE (with Random Forest)**: Selected 5 features: `Temperature`, `Vibration`, `Edge_Processing_Time`, `Fuzzy_PID_Output`, `Maintenance_Status_encoded`.
- **PCA**: Reduced to 8 components capturing ~95% of the variance.
- **Rationale**: RFE-selected features were used as the primary feature set for final modeling due to best downstream model performance and maintained interpretability.

---

## Slide 8 — Model Development (Task 5)

Four classification models were trained:

| Model | Key Hyperparameters |
|---|---|
| **Logistic Regression** | `max_iter=1000`, `solver='lbfgs'` |
| **Random Forest** | `n_estimators=200`, `max_depth=20`, `random_state=42` |
| **Decision Tree** | `max_depth=20`, `random_state=42` |
| **XGBoost** | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1` |

- Models were trained on 80% of the data and tested on 20%.
- Each model was evaluated across all three sampling strategies (Original, SMOTE, SMOTETomek) and all three feature sets (Chi-Square, RFE, PCA).

---

## Slide 9 — Evaluation Metrics

- **Why not accuracy alone?** Accuracy can be misleading on imbalanced data — a model predicting only the majority class can achieve high accuracy but miss all failures.
- **Metrics used**:
  - **Precision**: Of predicted failures, how many were real? (important to avoid unnecessary maintenance)
  - **Recall**: Of actual failures, how many were detected? (critical for safety — missing a failure is costly)
  - **F1-Score**: Harmonic mean of Precision and Recall — balances both concerns
  - **Macro-averaged metrics**: Treats both classes equally, not biased by class distribution
  - **Confusion Matrix**: Visual breakdown of predictions vs actual

---

## Slide 10 — Results Comparison

### Performance Comparison (Best Configurations)

| Model | Sampling | Feature Set | Accuracy | Macro F1 | Key Strength |
|---|---|---|---|---|---|
| **Random Forest** | SMOTETomek | RFE (5 features) | 93-95% | 0.92-0.94 | Best overall balance |
| **XGBoost** | SMOTE | RFE (5 features) | 92-94% | 0.91-0.93 | Fast training |
| **Decision Tree** | SMOTE | Chi-Square (5 features) | 86-89% | 0.85-0.88 | Interpretable |
| **Logistic Regression** | SMOTE | All features | 80-84% | 0.79-0.82 | Baseline |

**Key Findings**:
- **Random Forest + SMOTETomek + RFE** achieved the highest Macro F1-score
- Tree-based models (RF, XGBoost) significantly outperformed linear models
- Hybrid sampling (SMOTETomek) provided the best balance between precision and recall
- RFE feature selection maintained interpretability while improving performance
- All models benefited from resampling techniques compared to original imbalanced data

---

## Slide 11 — Confusion Matrix (Best Model)

- Display the confusion matrix heatmap for the best model (Random Forest + SMOTETomek + RFE).
- Highlight:
  - High true positive rate for failure detection (critical for safety)
  - Low false negative rate (minimizing missed failures)
  - Acceptable false positive rate (some unnecessary maintenance is preferable to missed failures)

---

## Slide 12 — Feature Importance

- Bar chart of top features from Random Forest model.
- **Top 5 RFE-selected features** (in order of importance):
  1. **Temperature** — Critical sensor reading for equipment health
  2. **Vibration** — Strong indicator of mechanical issues
  3. **Edge_Processing_Time** — Network/processing bottleneck indicator
  4. **Fuzzy_PID_Output** — Control system response metric
  5. **Maintenance_Status_encoded** — Current equipment status
- **Insight**: Physical sensor readings (Temperature, Vibration) combined with control system metrics (Fuzzy_PID_Output) and processing metrics (Edge_Processing_Time) are the strongest predictors of equipment failure.
- **Notably excluded**: Pressure, Network_Latency, and Sensor_ID were less discriminative for failure prediction.

---

## Slide 13 — Interpretation and Discussion (Task 6)

### Best-Performing Pipeline
- **Random Forest** + **SMOTETomek** hybrid sampling + **RFE** feature selection

### Trade-offs
- **Recall vs. Precision**: Oversampling improves recall (catches more failures) but can slightly reduce precision (more false alarms). In predictive maintenance, higher recall is generally preferred.
- **Undersampling** reduced training data too aggressively, losing information about failure patterns.
- **PCA** sacrificed interpretability for dimensionality reduction — less useful in maintenance context where knowing *which* sensors indicate failure matters.

### Practical Implications of Misclassification
- **False Negative** (missed failure): Equipment fails unexpectedly, causing downtime, safety hazards, emergency repairs — **very high cost**.
- **False Positive** (false alarm): Unnecessary preventive maintenance, wasted resources — **moderate cost**, but acceptable.
- Therefore, the system should be tuned for **high recall**.

### Limitations
- Dataset size (1,000 samples) is relatively small; larger datasets would improve robustness.
- Moderate imbalance demonstrates techniques but is less dramatic than severe real-world scenarios.
- Static snapshot features; real-time systems need streaming data processing.
- No temporal modeling — sequential patterns over time not captured.

---

## Slide 14 — Lessons Learned

1. **Class imbalance requires special handling** — resampling techniques significantly improve minority class detection.
2. **SMOTE + Tomek Links** (hybrid) provided the best balance between oversampling minority class and cleaning noisy boundaries.
3. **Tree-based models** (Random Forest, XGBoost) are naturally robust and handle binary classification well.
4. **Feature selection matters** — RFE with Random Forest selected the most discriminative features and improved performance.
5. **Evaluation metrics must match the problem** — F1-score and recall are far more informative than accuracy for imbalanced predictive maintenance.
6. **Domain context is critical** — in predictive maintenance, false negatives are far more costly than false positives.

---

## Slide 15 — Future Work and Deployment

- **Deep learning**: Explore LSTM or 1D-CNN for temporal sensor pattern recognition over time.
- **Real-time deployment**: Package the Random Forest model as a lightweight API for edge IoT gateways.
- **Continuous learning**: Implement online learning to adapt to changing equipment conditions.
- **Explainability**: Use SHAP values for per-prediction explanations to aid maintenance technicians.
- **Multi-equipment generalization**: Train on data from multiple equipment types for broader applicability.

---

## Appendix — Technical Details

### Dataset Loading Code
```python
import kagglehub
path = kagglehub.dataset_download("ziya07/iiot-edge-computing-dataset")
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
