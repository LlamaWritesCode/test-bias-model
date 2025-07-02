Loading loan_access_dataset.csv...
Cleaning and preprocessing training data...
Training the Tuned Random Forest Classifier model...
Model and preprocessing tools saved.

--- Evaluating Model on Internal Test Split ---
Accuracy on internal test split: 0.636
Classification Report on internal test split:

               precision    recall  f1-score   support

           0       0.66      0.75      0.70      1137

           1       0.60      0.48      0.53       863

   accuracy                            0.64      2000

   macro avg       0.63      0.62      0.62      2000

weighted avg       0.63      0.64      0.63      2000


🎯 Generating bias visualizations (using predictions on full loan_access_dataset for comprehensive audit)...

--- Fairness Auditing for Tuned Random Forest ---
- bias_visualization_selection_rate.png saved.
- confusion_matrix.png saved.
Creating Feature Importance plot (SHAP was not available)...
- feature_importance.png saved.
- demographic_analysis.png saved.

--- Generating submission.csv for test.csv ---
✅ submission.csv generated successfully for 'test.csv' with 0/1 predicted values.

--- AI Bias Bounty Hackathon Deliverables Completed ---
- **loan_model.py**: This script contains all data processing, model training, and fairness auditing steps.
- **submission.csv**: Model's output on the provided test dataset ('test.csv').
- **Visual Evidence of Bias**: Charts have been generated in the current directory:
  - `bias_visualization_selection_rate.png` (Predicted Approval rate by Race)
  - `confusion_matrix.png` (Overall Confusion Matrix)
  - `feature_importance.png` (Feature Importance Chart)
  - `demographic_analysis.png` (Predicted Approval Rates by various demographics)

**Model Performance Summary (on Full Loan_Access_Dataset Predictions):**
Total predictions: 10000
Overall predicted approval rate: 0.366
Actual approval rate (full dataset): 0.431
Overall Accuracy (full dataset): 0.815
Demographic Parity Difference (Race): 0.143
Note: DPD closer to 0 indicates better fairness
