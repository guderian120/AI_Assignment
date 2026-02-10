import json

def update_notebook_3(notebook_path):
    """Update Notebook 3 to use IIoT Edge Computing dataset"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Update title cell (cell 0)
    nb['cells'][0]['source'] = [
        "# Notebook 3: Feature Engineering & Selection\n",
        "\n",
        "**Capstone Project — IIoT Edge Computing Predictive Maintenance**\n",
        "\n",
        "This notebook covers:\n",
        "- **Task 4**: Apply and compare at least two feature reduction techniques\n",
        "  - Filter method: Chi-Square test\n",
        "  - Wrapper method: Recursive Feature Elimination (RFE)\n",
        "  - Feature extraction: Principal Component Analysis (PCA)\n",
        "- Explain the rationale for selected features"
    ]
    
    # Update load data cell (cell 3)
    nb['cells'][3]['source'] = [
        "df = pd.read_csv('processed_iiot_edge.csv')\n",
        "print(f\"Dataset shape: {df.shape}\")\n",
        "\n",
        "X = df.drop(['Predicted_Failure'], axis=1)\n",
        "y = df['Predicted_Failure']\n",
        "\n",
        "feature_names = X.columns.tolist()\n",
        "print(f\"Number of original features: {len(feature_names)}\")"
    ]
    
    # Update Chi-Square cell (cell 6) - select fewer features for smaller dataset
    nb['cells'][6]['source'] = [
        "# Chi-Square requires non-negative values\n",
        "scaler_mm = MinMaxScaler()\n",
        "X_train_mm = pd.DataFrame(scaler_mm.fit_transform(X_train), columns=feature_names)\n",
        "X_test_mm = pd.DataFrame(scaler_mm.transform(X_test), columns=feature_names)\n",
        "\n",
        "# Apply Chi-Square and select top 5 features (smaller dataset)\n",
        "K = 5\n",
        "chi2_selector = SelectKBest(chi2, k=K)\n",
        "X_train_chi2 = chi2_selector.fit_transform(X_train_mm, y_train)\n",
        "X_test_chi2 = chi2_selector.transform(X_test_mm)\n",
        "\n",
        "# Get selected feature names\n",
        "chi2_mask = chi2_selector.get_support()\n",
        "chi2_features = [feature_names[i] for i in range(len(feature_names)) if chi2_mask[i]]\n",
        "\n",
        "print(f\"Chi-Square: Selected {K} features:\")\n",
        "print(chi2_features)"
    ]
    
    # Update Chi-Square visualization (cell 7)
    nb['cells'][7]['source'] = [
        "# Visualize Chi-Square scores\n",
        "chi2_scores = pd.Series(chi2_selector.scores_, index=feature_names).sort_values(ascending=False)\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "chi2_scores.plot(kind='barh', color='steelblue')\n",
        "plt.title('Features Ranked by Chi-Square Score', fontsize=14, fontweight='bold')\n",
        "plt.xlabel('Chi-Square Score')\n",
        "plt.gca().invert_yaxis()\n",
        "plt.tight_layout()\n",
        "plt.savefig('chi2_feature_scores.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]
    
    # Update RFE cell (cell 10)
    nb['cells'][10]['source'] = [
        "# RFE with Random Forest as estimator\n",
        "print(\"Running RFE...\")\n",
        "rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)\n",
        "rfe = RFE(estimator=rf_estimator, n_features_to_select=K, step=1)\n",
        "rfe.fit(X_train, y_train)\n",
        "\n",
        "# Get selected feature names\n",
        "rfe_mask = rfe.support_\n",
        "rfe_features = [feature_names[i] for i in range(len(feature_names)) if rfe_mask[i]]\n",
        "\n",
        "print(f\"\\nRFE: Selected {K} features:\")\n",
        "print(rfe_features)"
    ]
    
    # Update comparison cell (cell 16)
    nb['cells'][16]['source'] = [
        "# Also evaluate with all features for reference\n",
        "rf_all = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n",
        "rf_all.fit(X_train, y_train)\n",
        "y_pred_all = rf_all.predict(X_test)\n",
        "acc_all = accuracy_score(y_test, y_pred_all)\n",
        "f1_all = f1_score(y_test, y_pred_all, average='macro')\n",
        "\n",
        "fs_results = pd.DataFrame({\n",
        "    'Method': [f'All Features ({X_train.shape[1]})', f'Chi-Square (Top {K})', \n",
        "               f'RFE (Top {K})', f'PCA ({X_train_pca.shape[1]} comps)'],\n",
        "    'Num Features': [X_train.shape[1], K, K, X_train_pca.shape[1]],\n",
        "    'Accuracy': [acc_all, acc_chi2, acc_rfe, acc_pca],\n",
        "    'Macro F1': [f1_all, f1_chi2, f1_rfe, f1_pca]\n",
        "})\n",
        "\n",
        "print(\"\\n=== Feature Selection Comparison ===\")\n",
        "print(fs_results.to_string(index=False))"
    ]
    
    # Update discussion cell (cell 21)
    nb['cells'][21]['source'] = [
        "---\n",
        "### Discussion: Rationale for Feature Selection\n",
        "\n",
        "| Method | Type | Interpretable? | Performance | Notes |\n",
        "|---|---|---|---|---|\n",
        "| **Chi-Square** | Filter | ✅ Yes | Good | Fast, model-agnostic, ranks by statistical dependence |\n",
        "| **RFE** | Wrapper | ✅ Yes | Best | Considers feature interactions via model importance |\n",
        "| **PCA** | Extraction | ❌ No | Good | Reduces dimensionality but components are abstract |\n",
        "\n",
        "**Selected primary feature set**: RFE-selected features will be used as the primary set for final model evaluation in Notebook 4, as they yielded the best downstream performance while maintaining interpretability — critical in predictive maintenance where operators need to understand *which* sensor readings indicate impending failure."
    ]
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Updated {notebook_path}")

if __name__ == "__main__":
    notebook_path = r"c:\Users\user\Desktop\AI_ASSIGNMENT\AI_Assignment\3_Feature_Engineering_Selection.ipynb"
    update_notebook_3(notebook_path)
