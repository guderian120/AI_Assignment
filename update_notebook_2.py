import json

def update_notebook_2(notebook_path):
    """Update Notebook 2 to use IIoT Edge Computing dataset"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Update title cell (cell 0)
    nb['cells'][0]['source'] = [
        "# Notebook 2: Class Imbalance Handling\n",
        "\n",
        "**Capstone Project — IIoT Edge Computing Predictive Maintenance**\n",
        "\n",
        "This notebook covers:\n",
        "- **Task 3**: Implement and compare at least two class imbalance strategies\n",
        "  - Oversampling (SMOTE)\n",
        "  - Undersampling (RandomUnderSampler)\n",
        "  - Hybrid (SMOTETomek)\n",
        "- Discuss impact on model performance"
    ]
    
    # Update load data cell (cell 3)
    nb['cells'][3]['source'] = [
        "df = pd.read_csv('processed_iiot_edge.csv')\n",
        "print(f\"Dataset shape: {df.shape}\")\n",
        "\n",
        "X = df.drop(['Predicted_Failure'], axis=1)\n",
        "y = df['Predicted_Failure']\n",
        "\n",
        "# Train-test split\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "print(f\"Training set: {X_train.shape}\")\n",
        "print(f\"Test set: {X_test.shape}\")\n",
        "print(f\"\\nOriginal class distribution (train):\")\n",
        "print(pd.Series(y_train).value_counts().sort_index())"
    ]
    
    # Update baseline cell description (cell 4)
    nb['cells'][4]['source'] = [
        "## 2. Baseline — No Resampling\n",
        "Train a Random Forest on the original moderately imbalanced data as a baseline."
    ]
    
    # Update save datasets cell (cell 18)
    nb['cells'][18]['source'] = [
        "# Save SMOTE-resampled data (typically best balance of info preservation)\n",
        "smote_df = pd.DataFrame(X_train_smote, columns=X.columns)\n",
        "smote_df['Predicted_Failure'] = y_train_smote.values\n",
        "smote_df.to_csv('train_smote.csv', index=False)\n",
        "\n",
        "# Save SMOTETomek-resampled data\n",
        "st_df = pd.DataFrame(X_train_st, columns=X.columns)\n",
        "st_df['Predicted_Failure'] = y_train_st.values\n",
        "st_df.to_csv('train_smotetomek.csv', index=False)\n",
        "\n",
        "# Save test set\n",
        "test_df = pd.DataFrame(X_test, columns=X.columns)\n",
        "test_df['Predicted_Failure'] = y_test.values\n",
        "test_df.to_csv('test_set.csv', index=False)\n",
        "\n",
        "print(\"Saved: train_smote.csv, train_smotetomek.csv, test_set.csv\")"
    ]
    
    # Update discussion cell (cell 19)
    nb['cells'][19]['source'] = [
        "---\n",
        "### Discussion: Impact of Sampling Strategies\n",
        "\n",
        "| Strategy | Pros | Cons |\n",
        "|---|---|---|\n",
        "| **SMOTE** | Preserves all original data; synthesizes realistic minority samples | Can increase training time; may introduce noise |\n",
        "| **RandomUnderSampler** | Fast; reduces training time | Discards potentially valuable majority-class data |\n",
        "| **SMOTETomek** | Best of both — oversamples minority and cleans noisy boundaries | Most computationally expensive |\n",
        "\n",
        "**Key Finding**: Even with moderate imbalance (1.2:1), SMOTE and SMOTETomek can improve Macro F1-score by better balancing precision and recall across both classes. The improvements are less dramatic than with severe imbalance, but still demonstrate the value of resampling techniques. In predictive maintenance, improving recall for the failure class is critical to avoid missing equipment failures."
    ]
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Updated {notebook_path}")

if __name__ == "__main__":
    notebook_path = r"c:\Users\user\Desktop\AI_ASSIGNMENT\AI_Assignment\2_Class_Imbalance_Handling.ipynb"
    update_notebook_2(notebook_path)
