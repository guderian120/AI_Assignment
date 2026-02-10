import json

def update_notebook_4(notebook_path):
    """Update Notebook 4 to use IIoT Edge Computing dataset"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find and update the title cell
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and 'Notebook 4' in ''.join(cell['source']):
            nb['cells'][i]['source'] = [
                "# Notebook 4: Model Training & Evaluation\n",
                "\n",
                "**Capstone Project — IIoT Edge Computing Predictive Maintenance**\n",
                "\n",
                "This notebook covers:\n",
                "- **Task 5**: Train at least two ML models and evaluate using appropriate metrics\n",
                "- **Task 6**: Interpretation and discussion\n",
                "  - Identify best-performing pipeline\n",
                "  - Discuss precision vs recall trade-offs\n",
                "  - Practical implications of misclassification\n",
                "  - Limitations and future work"
            ]
            break
    
    # Update data loading cells to use new dataset
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'processed_iot_intrusion.csv' in ''.join(cell['source']):
            nb['cells'][i]['source'] = [line.replace('processed_iot_intrusion.csv', 'processed_iiot_edge.csv')
                                        .replace("['label', 'label_encoded']", "['Predicted_Failure']")
                                        .replace("'label_encoded'", "'Predicted_Failure'")
                                        for line in cell['source']]
        
        # Update train/test loading
        if cell['cell_type'] == 'code' and ('train_smote.csv' in ''.join(cell['source']) or 
                                             'train_smotetomek.csv' in ''.join(cell['source'])):
            nb['cells'][i]['source'] = [line.replace("'label_encoded'", "'Predicted_Failure'")
                                        for line in cell['source']]
    
    # Update discussion cells about domain context
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and 'Practical Implications' in ''.join(cell['source']):
            nb['cells'][i]['source'] = [
                "### Practical Implications of Misclassification\n",
                "\n",
                "In predictive maintenance for industrial IoT systems:\n",
                "\n",
                "| Error Type | Impact | Cost Level |\n",
                "|---|---|---|\n",
                "| False Negative (missed failure) | Equipment fails unexpectedly, causing downtime, safety hazards, and expensive emergency repairs | **Very High** |\n",
                "| False Positive (false alarm) | Unnecessary preventive maintenance, wasted resources | Moderate |\n",
                "\n",
                "**Conclusion**: High recall is critical — missing a failure prediction can lead to catastrophic equipment failure, production downtime, and safety incidents. A false alarm is preferable to a missed failure."
            ]
            break
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Updated {notebook_path}")

if __name__ == "__main__":
    notebook_path = r"c:\Users\user\Desktop\AI_ASSIGNMENT\AI_Assignment\4_Model_Training_Evaluation.ipynb"
    update_notebook_4(notebook_path)
