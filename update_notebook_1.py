import json
import sys

def update_notebook_1(notebook_path):
    """Update Notebook 1 to use IIoT Edge Computing dataset"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Update title cell
    nb['cells'][0]['source'] = [
        "# Notebook 1: Dataset Selection, EDA & Preprocessing\n",
        "\n",
        "**Capstone Project — IIoT Edge Computing Predictive Maintenance**\n",
        "\n",
        "This notebook covers:\n",
        "- **Task 1**: Dataset Selection & Justification\n",
        "- **Task 2**: Data Understanding & Preprocessing"
    ]
    
    # Update dataset description cell
    nb['cells'][1]['source'] = [
        "## Task 1: Dataset Selection & Justification\n",
        "\n",
        "### Dataset\n",
        "- **Name**: IIoT Edge Computing Dataset\n",
        "- **Source**: [Kaggle — ziya07/iiot-edge-computing-dataset](https://www.kaggle.com/datasets/ziya07/iiot-edge-computing-dataset)\n",
        "- **Records**: 1,000 sensor readings\n",
        "- **Features**: 9 features (6 numerical + 3 categorical) + 1 binary target (`Predicted_Failure`)\n",
        "\n",
        "### Domain Relevance\n",
        "Industrial IoT (IIoT) edge computing systems are critical for real-time monitoring and predictive maintenance in manufacturing and industrial environments. This dataset captures sensor readings from industrial equipment including temperature, pressure, vibration, network latency, and edge processing metrics. Predicting equipment failures before they occur is essential for preventing costly downtime, reducing maintenance costs, and ensuring operational safety.\n",
        "\n",
        "### Nature & Severity of Class Imbalance\n",
        "The dataset exhibits moderate class imbalance with a 1.2:1 ratio between non-failure (546 samples, 54.6%) and failure cases (454 samples, 45.4%). While less severe than many real-world datasets, this imbalance still provides an opportunity to demonstrate resampling techniques (SMOTE, undersampling, hybrid methods) and their impact on model performance. In predictive maintenance, missing a failure (false negative) can be extremely costly, making recall optimization critical."
    ]
    
    # Update dataset loading cell (cell index 4)
    nb['cells'][4]['source'] = [
        "# Download and load dataset\n",
        "path = kagglehub.dataset_download(\"ziya07/iiot-edge-computing-dataset\")\n",
        "csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]\n",
        "df = pd.read_csv(os.path.join(path, csv_file))\n",
        "\n",
        "print(f\"Dataset Shape: {df.shape}\")\n",
        "print(f\"Number of features: {df.shape[1] - 1}\")\n",
        "print(f\"Number of records: {df.shape[0]:,}\")\n",
        "print(f\"\\nFeature columns: {df.columns.tolist()}\")\n",
        "df.head()"
    ]
    
    # Update class distribution cell (cell index 8)
    nb['cells'][8]['source'] = [
        "# Class distribution\n",
        "class_dist = df['Predicted_Failure'].value_counts()\n",
        "print(\"Class Distribution:\")\n",
        "print(class_dist)\n",
        "print(f\"\\nClass 0 (No Failure): {class_dist[0]} samples ({class_dist[0]/len(df)*100:.1f}%)\")\n",
        "print(f\"Class 1 (Failure): {class_dist[1]} samples ({class_dist[1]/len(df)*100:.1f}%)\")\n",
        "print(f\"\\nImbalance ratio (majority/minority): {class_dist.max() / class_dist.min():.2f}\")"
    ]
    
    # Update class visualization cell (cell index 9)
    nb['cells'][9]['source'] = [
        "# Visualize class distribution\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n",
        "\n",
        "# Bar chart\n",
        "ax1 = axes[0]\n",
        "class_labels = ['No Failure', 'Failure']\n",
        "class_counts = [class_dist[0], class_dist[1]]\n",
        "bars = ax1.bar(class_labels, class_counts, color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5)\n",
        "ax1.set_title('Distribution of Failure Prediction Classes', fontsize=14, fontweight='bold')\n",
        "ax1.set_ylabel('Count')\n",
        "ax1.set_xlabel('Class')\n",
        "for bar in bars:\n",
        "    height = bar.get_height()\n",
        "    ax1.text(bar.get_x() + bar.get_width()/2., height,\n",
        "            f'{int(height)}\\n({height/len(df)*100:.1f}%)',\n",
        "            ha='center', va='bottom', fontsize=11)\n",
        "\n",
        "# Pie chart for proportions\n",
        "ax2 = axes[1]\n",
        "ax2.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90,\n",
        "        colors=['#2ecc71', '#e74c3c'], explode=(0.05, 0.05))\n",
        "ax2.set_title('Class Proportion', fontsize=14, fontweight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]
    
    # Update feature distributions cell (cell index 11)
    nb['cells'][11]['source'] = [
        "# Distribution of key continuous features\n",
        "key_features = ['Temperature', 'Pressure', 'Vibration', 'Network_Latency', 'Edge_Processing_Time', 'Fuzzy_PID_Output']\n",
        "\n",
        "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
        "axes = axes.flatten()\n",
        "for i, feat in enumerate(key_features):\n",
        "    df[feat].hist(bins=30, ax=axes[i], color='steelblue', edgecolor='white')\n",
        "    axes[i].set_title(feat, fontsize=12, fontweight='bold')\n",
        "    axes[i].set_ylabel('Frequency')\n",
        "    axes[i].set_xlabel(feat)\n",
        "\n",
        "plt.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]
    
    # Update outlier detection cell (cell index 15)
    nb['cells'][15]['source'] = [
        "# Box plots for continuous features to visualize outliers\n",
        "outlier_features = ['Temperature', 'Pressure', 'Vibration', 'Network_Latency', 'Edge_Processing_Time', 'Fuzzy_PID_Output']\n",
        "\n",
        "fig, axes = plt.subplots(1, len(outlier_features), figsize=(20, 5))\n",
        "for i, feat in enumerate(outlier_features):\n",
        "    sns.boxplot(y=df[feat], ax=axes[i], color='lightcoral')\n",
        "    axes[i].set_title(feat, fontsize=10)\n",
        "\n",
        "plt.suptitle('Outlier Detection — Box Plots', fontsize=14, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig('outlier_boxplots.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]
    
    # Update label encoding cell (cell index 18)
    nb['cells'][18]['source'] = [
        "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
        "\n",
        "# Encode categorical features\n",
        "le_sensor = LabelEncoder()\n",
        "le_maintenance = LabelEncoder()\n",
        "\n",
        "df['Sensor_ID_encoded'] = le_sensor.fit_transform(df['Sensor_ID'])\n",
        "df['Maintenance_Status_encoded'] = le_maintenance.fit_transform(df['Maintenance_Status'])\n",
        "\n",
        "# Print mappings\n",
        "print(\"Sensor_ID Encoding:\")\n",
        "for name, code in zip(le_sensor.classes_, le_sensor.transform(le_sensor.classes_)):\n",
        "    print(f\"  {code}: {name}\")\n",
        "\n",
        "print(\"\\nMaintenance_Status Encoding:\")\n",
        "for name, code in zip(le_maintenance.classes_, le_maintenance.transform(le_maintenance.classes_)):\n",
        "    print(f\"  {code}: {name}\")\n",
        "\n",
        "print(f\"\\nTarget variable 'Predicted_Failure' is already binary encoded (0/1)\")"
    ]
    
    # Update feature scaling cell (cell index 19)
    nb['cells'][19]['source'] = [
        "# Separate features and target\n",
        "# Drop original categorical columns and timestamp, keep encoded versions\n",
        "X = df.drop(['Timestamp', 'Sensor_ID', 'Maintenance_Status', 'Predicted_Failure'], axis=1)\n",
        "y = df['Predicted_Failure']\n",
        "\n",
        "# Scale numerical features\n",
        "scaler = StandardScaler()\n",
        "X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)\n",
        "\n",
        "print(f\"Features shape: {X_scaled.shape}\")\n",
        "print(f\"Target shape: {y.shape}\")\n",
        "print(f\"Feature columns: {X_scaled.columns.tolist()}\")\n",
        "X_scaled.head()"
    ]
    
    # Update save processed data cell (cell index 21)
    nb['cells'][21]['source'] = [
        "# Save processed data for subsequent notebooks\n",
        "processed_df = X_scaled.copy()\n",
        "processed_df['Predicted_Failure'] = y.values\n",
        "\n",
        "processed_df.to_csv('processed_iiot_edge.csv', index=False)\n",
        "print(\"Processed data saved to 'processed_iiot_edge.csv'\")\n",
        "print(f\"Final shape: {processed_df.shape}\")"
    ]
    
    # Update summary cell (cell index 22)
    nb['cells'][22]['source'] = [
        "---\n",
        "### Summary of EDA & Preprocessing\n",
        "- **No missing values** found.\n",
        "- **Moderate class imbalance** (1.2:1 ratio) between failure and non-failure cases.\n",
        "- **Categorical features** (Sensor_ID, Maintenance_Status) encoded using LabelEncoder.\n",
        "- **Outliers** handled via Winsorization at 1st/99th percentiles.\n",
        "- **Features scaled** using StandardScaler.\n",
        "- Data saved for use in Notebooks 2, 3, and 4."
    ]
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Updated {notebook_path}")

if __name__ == "__main__":
    notebook_path = r"c:\Users\user\Desktop\AI_ASSIGNMENT\AI_Assignment\1_EDA_and_Preprocessing.ipynb"
    update_notebook_1(notebook_path)
