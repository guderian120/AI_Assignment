# IoT Network Intrusion Detection on Imbalanced Data

## Technical Report — Machine Learning Capstone Project

---

## 1. Introduction

The proliferation of Internet of Things (IoT) devices in smart homes, healthcare, and industrial systems has introduced significant cybersecurity vulnerabilities. These resource-constrained devices are prime targets for network-based attacks such as Distributed Denial-of-Service (DDoS), Denial-of-Service (DoS), and reconnaissance probes. Building effective Intrusion Detection Systems (IDS) for IoT environments is challenged by **severe class imbalance** — normal traffic vastly outnumbers attack traffic, and rare attack types are critically underrepresented.

This project develops an end-to-end machine learning pipeline to classify IoT network traffic into benign and multiple attack categories, addressing class imbalance through resampling strategies, applying feature selection techniques, and evaluating multiple classification models using metrics appropriate for imbalanced data.

---

## 2. Dataset Description

- **Source**: [Kaggle — subhajournal/iotintrusion](https://www.kaggle.com/datasets/subhajournal/iotintrusion)
- **Records**: 1,048,575 network flow records
- **Features**: 46 numerical features + 1 categorical target (`label`)
- **Target Classes**: Multi-class — includes `BenignTraffic`, `DDoS-RSTFINFlood`, `DoS-TCP_Flood`, `DDoS-ICMP_Flood`, `DoS-UDP_Flood`, `DoS-SYN_Flood`, and others

**Feature categories**:

| Category | Features |
|---|---|
| Flow metrics | `flow_duration`, `Header_Length`, `Protocol Type`, `Duration`, `Rate`, `Srate`, `Drate` |
| TCP flags | `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number` |
| Flag counts | `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count` |
| Protocol indicators | `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC` |
| Statistical aggregates | `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `IAT`, `Number`, `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight` |

**Domain Relevance**: IoT security is a critical concern — misclassifying an attack as benign can lead to data breaches, service disruption, or compromised infrastructure.

**Class Imbalance**: The dataset exhibits significant multi-class imbalance. Certain attack types contain hundreds of thousands of samples while others have far fewer, making naive classifiers biased toward majority classes.

---

## 3. Methodology

### 3.1 Exploratory Data Analysis & Preprocessing

**Missing Values**: No missing values were found across all 1,048,575 records.

**Class Distribution**:

![Class Distribution — Bar chart and pie chart showing severe imbalance across attack types](class_distribution.png)

**Feature Distributions**: Many features exhibit right-skewed distributions, particularly `Header_Length`, `Rate`, and `Tot sum`.

![Feature Distributions — Histograms of key numerical features](feature_distributions.png)

**Correlation Analysis**: A correlation heatmap revealed high correlation (|r| > 0.9) among related features such as `Tot sum`/`Tot size` and flag counts/flag numbers, motivating feature selection.

![Correlation Heatmap — Triangular heatmap of all 46 features](correlation_heatmap.png)

**Outlier Handling**: Extreme values were detected in `Header_Length`, `Rate`, and `Tot sum`. Winsorization (capping at 1st/99th percentiles) was applied.

![Outlier Detection — Box plots of key features](outlier_boxplots.png)

**Preprocessing Steps**:
- Outliers capped via Winsorization
- Labels encoded using `LabelEncoder`
- Features scaled using `StandardScaler` (zero mean, unit variance)

---

### 3.2 Class Imbalance Handling

Three resampling strategies were implemented and compared:

| Strategy | Method | Description |
|---|---|---|
| **Oversampling** | SMOTE | Generates synthetic minority samples via nearest-neighbor interpolation |
| **Undersampling** | RandomUnderSampler | Randomly removes majority-class samples |
| **Hybrid** | SMOTETomek | SMOTE + Tomek Links boundary cleaning |

![Sampling Strategy Comparison — Accuracy and Macro F1 across strategies](sampling_comparison.png)

**Key Finding**: SMOTE and SMOTETomek significantly improved Macro F1-score for minority classes compared to the baseline. Pure undersampling discarded too much data and reduced overall performance.

---

### 3.3 Feature Engineering & Selection

Three feature reduction techniques were applied:

| Method | Type | Approach |
|---|---|---|
| **Chi-Square** | Filter | Ranks features by statistical dependence with target |
| **RFE** | Wrapper | Recursive elimination using Random Forest importance |
| **PCA** | Extraction | Orthogonal transformation retaining 95% variance |

![Chi-Square Feature Scores — Top 20 features ranked by Chi-Square score](chi2_feature_scores.png)

![RFE Feature Ranking — All features ranked (green = selected)](rfe_feature_ranking.png)

![PCA Cumulative Variance — Elbow plot showing components needed for 95% variance](pca_variance.png)

**Comparison**:

![Feature Selection Comparison — Accuracy and F1 across methods](feature_selection_comparison.png)

**Selected Feature Set**: RFE-selected features (20 features) were chosen as the primary set due to best downstream model performance and maintained interpretability.

---

### 3.4 Model Development

Four classification models were trained across all sampling strategies:

| Model | Key Configuration |
|---|---|
| Logistic Regression | `max_iter=1000`, `multi_class='multinomial'` |
| Random Forest | `n_estimators=200`, `max_depth=20` |
| Decision Tree | `max_depth=20` |
| XGBoost | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1` |

Each model was evaluated on: **Accuracy**, **Precision (macro)**, **Recall (macro)**, and **F1-Score (macro)**.

---

## 4. Results

### 4.1 Model Comparison

![Model Comparison — Grouped bar charts of F1 and Accuracy by model and sampling](model_comparison.png)

### 4.2 Best Model — Detailed Evaluation

The best-performing pipeline was identified by highest Macro F1-Score.

**Confusion Matrix**:

![Confusion Matrix — Best model's per-class prediction breakdown](best_model_confusion_matrix.png)

**Feature Importance**:

![Feature Importance — Top features from the best tree-based model](feature_importance.png)

**Precision vs. Recall Trade-off**:

![Precision vs Recall — Scatter plot by class (bubble size = support)](precision_recall_tradeoff.png)

---

## 5. Discussion and Conclusion

### Best-Performing Pipeline
The combination of **tree-based model + SMOTETomek hybrid sampling + RFE feature selection** yielded the best Macro F1-Score, demonstrating strong detection across both majority and minority attack classes.

### Trade-offs: Recall vs. Precision
- **High recall** is critical — missing an attack (false negative) can result in data breaches or device compromise.
- SMOTE/SMOTETomek improve recall for minority classes at a slight cost to precision on majority classes.
- In security contexts, this trade-off is acceptable: it is better to investigate a false alarm than to miss a real attack.

### Practical Implications of Misclassification

| Error Type | Impact | Cost Level |
|---|---|---|
| False Negative (missed attack) | Data breach, service disruption | **Very High** |
| False Positive (false alarm) | Unnecessary investigation | Moderate |
| Wrong attack type | Incorrect response protocol | Moderate |

### Limitations
1. **Dataset specificity** — trained on one IoT dataset; may not generalize to other deployments.
2. **Static features** — uses precomputed flow statistics; real-time IDS needs streaming extraction.
3. **No temporal modeling** — sequential traffic patterns are not captured by tabular classifiers.
4. **Computational cost** — SMOTE on 1M+ records is memory-intensive.
5. **Concept drift** — attack patterns evolve; model needs periodic retraining.

### Future Work
- **Deep Learning**: LSTM or 1D-CNN for sequential flow pattern recognition.
- **Ensemble Stacking**: Combine RF, XGBoost, and neural network via meta-learner.
- **Edge Deployment**: Export model as ONNX for lightweight inference on IoT gateways.
- **Explainability**: SHAP values for per-prediction explanations.
- **Continuous Learning**: Online learning pipeline with periodic retraining on new traffic.

---

## References

1. Subha Journal. "IoT Intrusion Detection Dataset." Kaggle, 2024. https://www.kaggle.com/datasets/subhajournal/iotintrusion
2. Chawla, N. V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." JAIR, 2002.
3. Batista, G. E., et al. "A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data." ACM SIGKDD, 2004.
4. Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python." JMLR, 2011.

---

> **Note**: All visualizations above are generated by running the Jupyter Notebooks in order (1 → 2 → 3 → 4). Run the notebooks first to produce the PNG files referenced in this report.
