# Dataset Selection Disclaimer

## Context

This capstone project originally utilized a large-scale IoT Intrusion Detection dataset with over 1 million records and 46 features. While this dataset provided excellent opportunities for demonstrating machine learning techniques on severely imbalanced data, its size proved impractical for the scope and timeline of this assignment.

## Dataset Change Rationale

We have switched to the **IIoT Edge Computing Dataset** for the following reasons:

### 1. **Manageable Scope**
- **Original dataset**: 1,048,575 records, 46 features
- **New dataset**: 1,000 records, 9 features
- The smaller dataset allows for faster iteration, experimentation, and thorough analysis within the assignment timeframe

### 2. **Still Demonstrates All Required Techniques**
The new dataset fully satisfies all capstone requirements:
- ✅ Real-world imbalanced dataset (1.2:1 ratio)
- ✅ Domain relevance (industrial predictive maintenance)
- ✅ Opportunity to apply resampling techniques (SMOTE, undersampling, hybrid)
- ✅ Feature selection methods (Chi-Square, RFE, PCA)
- ✅ Multiple ML models (Logistic Regression, Random Forest, Decision Tree, XGBoost)
- ✅ Appropriate evaluation metrics for imbalanced data

### 3. **Educational Value**
- The moderate class imbalance (54.6% vs 45.4%) is more representative of many real-world scenarios
- Smaller feature space (9 vs 46) allows for more interpretable analysis
- Faster training times enable more extensive hyperparameter tuning and experimentation

### 4. **Real-World Relevance**
- **Domain**: Industrial IoT edge computing for predictive maintenance
- **Impact**: Predicting equipment failures is critical for preventing downtime, reducing costs, and ensuring safety
- **Features**: Interpretable sensor readings (temperature, pressure, vibration) and network metrics

## Conclusion

While the original dataset would have provided more dramatic demonstrations of class imbalance handling, the IIoT Edge Computing Dataset offers a more appropriate balance between:
- Demonstrating all required machine learning techniques
- Maintaining real-world relevance and interpretability
- Completing thorough analysis within assignment constraints

This dataset change was made to ensure the highest quality deliverables while meeting all capstone project requirements.

---

**Dataset Sources:**
- Original: [Kaggle — subhajournal/iotintrusion](https://www.kaggle.com/datasets/subhajournal/iotintrusion)
- Current: [Kaggle — ziya07/iiot-edge-computing-dataset](https://www.kaggle.com/datasets/ziya07/iiot-edge-computing-dataset)
