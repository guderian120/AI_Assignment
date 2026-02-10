# 🏭 IIoT Edge Computing Predictive Maintenance

A machine learning capstone project that builds an end-to-end predictive maintenance system for Industrial IoT (IIoT) edge computing environments, tackling the challenge of **class imbalance** in failure prediction.

## 📋 Project Overview

Industrial equipment failures can lead to costly downtime and safety hazards. This project uses the [IIoT Edge Computing Dataset](https://www.kaggle.com/datasets/ziya07/iiot-edge-computing-dataset) (1,000 sensor readings, 9 features) to:

1. **Explore & preprocess** sensor data with visualizations
2. **Handle class imbalance** using SMOTE, undersampling, and hybrid methods
3. **Select optimal features** via Chi-Square, RFE, and PCA
4. **Train & evaluate** multiple ML models (Random Forest, XGBoost, Logistic Regression, Decision Tree)
5. **Interpret results** with confusion matrices, feature importance, and precision-recall analysis

## 📁 Repository Structure

```
AI_Assignment/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── Technical_Report.md                    # Concise technical report with visualizations
├── 1_EDA_and_Preprocessing.ipynb          # Tasks 1-2: Dataset selection, EDA, preprocessing
├── 2_Class_Imbalance_Handling.ipynb       # Task 3: SMOTE, undersampling, SMOTETomek
├── 3_Feature_Engineering_Selection.ipynb  # Task 4: Chi-Square, RFE, PCA
├── 4_Model_Training_Evaluation.ipynb      # Tasks 5-6: Model training, evaluation, discussion
├── inspect_dataset.py                     # Utility script to inspect dataset structure
└── dataset_info.txt                       # Dataset schema reference
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Kaggle account (for dataset download via `kagglehub`)

### Installation

```bash
# Clone the repository
git clone https://github.com/guderian120/AI_Assignment.git
cd AI_Assignment

# Install dependencies
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
```

### Running the Notebooks

> ⚠️ **Run notebooks in order** — each saves intermediate files used by the next.

```bash
jupyter notebook
```

Then open and run sequentially:

1. **`1_EDA_and_Preprocessing.ipynb`** → Outputs: `processed_iiot_edge.csv`
2. **`2_Class_Imbalance_Handling.ipynb`** → Outputs: `train_smote.csv`, `train_smotetomek.csv`, `test_set.csv`
3. **`3_Feature_Engineering_Selection.ipynb`** → Outputs: `selected_features.json`
4. **`4_Model_Training_Evaluation.ipynb`** → Outputs: Final results and visualizations

The dataset is **automatically downloaded** from Kaggle via `kagglehub` when you run Notebook 1.

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — ziya07/iiot-edge-computing-dataset](https://www.kaggle.com/datasets/ziya07/iiot-edge-computing-dataset) |
| **Records** | 1,000 sensor readings |
| **Features** | 9 features (6 numerical + 3 categorical) |
| **Target** | `Predicted_Failure` — binary (0=No Failure, 1=Failure) |
| **Imbalance** | Moderate — 1.2:1 ratio (54.6% vs 45.4%) |

## 🔬 Methodology

### Pipeline

```
Raw Data → EDA & Preprocessing → Class Imbalance Handling → Feature Selection → Model Training → Evaluation
```

### Class Imbalance Strategies
| Strategy | Method |
|---|---|
| Oversampling | SMOTE |
| Undersampling | RandomUnderSampler |
| Hybrid | SMOTETomek (SMOTE + Tomek Links) |

### Feature Selection Methods
| Method | Type | Features Selected |
|---|---|---|
| Chi-Square | Filter | Top 5 by statistical dependence |
| RFE | Wrapper | Top 5 by Random Forest importance |
| PCA | Extraction | Components retaining 95% variance |

### Models Evaluated
- Logistic Regression
- Random Forest
- Decision Tree
- XGBoost

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (macro-averaged)
- Confusion Matrix
- Feature Importance
- Precision-Recall trade-off analysis

## 📈 Key Results

- **Best pipeline**: Tree-based model + SMOTE/SMOTETomek + RFE features
- **SMOTE/Hybrid sampling** improves recall for failure class, critical for avoiding missed failures
- **RFE feature selection** outperforms Chi-Square and PCA for downstream classification
- **Tree-based models** (Random Forest, XGBoost) outperform linear models on this dataset

## 📝 Deliverables

1. **Technical Report**: [`Technical_Report.md`](Technical_Report.md) — structured as a mini research paper
2. **Code**: 4 Jupyter Notebooks with documented, reproducible code
3. **Presentation**: Can be generated from the technical report and notebook visualizations

## ⚙️ Tech Stack

- **Languages**: Python 3
- **ML Libraries**: scikit-learn, imbalanced-learn, XGBoost
- **Data**: pandas, NumPy
- **Visualization**: matplotlib, seaborn
- **Dataset Access**: kagglehub

## 💡 Dataset Selection Rationale

This dataset was selected to provide a manageable scope for the capstone project while still demonstrating all required machine learning techniques:
- **Moderate imbalance** (1.2:1) allows demonstration of resampling techniques
- **Real-world relevance** in industrial predictive maintenance
- **Appropriate size** (1,000 samples) for thorough analysis within assignment timeframe
- **Clear domain context** with interpretable features

## 📄 License

This project is for academic purposes as part of a Machine Learning Capstone course.
