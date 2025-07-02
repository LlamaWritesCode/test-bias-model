# loan_model.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Define the filenames for saved model components
MODEL_FILE = "mitigated_model.pkl"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODERS_FILE = "label_encoders.pkl"

# --- 1. Data Cleaning and Preprocessing ---
def clean_and_preprocess(df_raw, categorical_cols, train_mode=True, scaler=None, label_encoders=None):
    """
    Performs data cleaning and preprocessing steps.
    - Converts 'Loan_Approved' to numerical (1 for Approved, 0 for Denied) if in training mode.
    - Handles categorical features using LabelEncoder.
    - Scales numerical features using StandardScaler.
    - Drops the 'ID' column.
    """
    df = df_raw.copy()
    original_ids = df['ID']

    if 'Loan_Approved' in df.columns: # Only present in training data
        df['Loan_Approved'] = df['Loan_Approved'].map({'Approved': 1, 'Denied': 0})
        target_series = df['Loan_Approved']
        df = df.drop(columns=['ID', 'Loan_Approved'])
    else: # For test data, no target column
        target_series = None
        df = df.drop(columns=['ID'])

    # Initialize or use provided encoders/scaler
    if train_mode:
        label_encoders = {}
        scaler = StandardScaler()
    elif label_encoders is None or scaler is None:
        raise ValueError("Scaler and LabelEncoders must be provided in non-training mode.")

    # Encode categorical variables
    for col in categorical_cols:
        if col in df.columns:
            if train_mode:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
            else: # Apply to test data
                # Handle unseen labels in test data
                unique_train_labels = label_encoders[col].classes_
                df[col] = df[col].apply(lambda x: x if x in unique_train_labels else unique_train_labels[0])
                df[col] = label_encoders[col].transform(df[col])
        elif not train_mode: # Column might be missing in test set if it's all one category in train
            df[col] = 0 # Or some other appropriate default for missing categorical feature in test

    # Scale numerical features
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if train_mode:
        X_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
    else:
        X_scaled = pd.DataFrame(scaler.transform(df[numeric_cols]), columns=numeric_cols, index=df.index)

    return X_scaled, target_series, original_ids, scaler, label_encoders


# --- 2. Feature Engineering ---
# For this dataset, feature engineering primarily involves label encoding of categorical
# variables and scaling numerical features, which are handled in the clean_and_preprocess function.
# More complex feature engineering (e.g., polynomial features, interaction terms) could be added here.


# --- 3. Model Training ---
def train_model(X_train, y_train):
    """
    Trains a Tuned Random Forest Classifier model.
    """
    # Tuned Random Forest parameters from previous successful run (accuracy 0.6375)
    model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)
    return model


# --- 4. Fairness Auditing and Bias Detection ---
def audit_fairness(df_raw, y_true_full, y_pred_full_dataset, demographic_features, mitigator, X_scaled_global_for_features, model_name="Model"):
    """
    Performs fairness auditing by calculating and visualizing:
    - Predicted Loan Approval Rates by demographic group.
    - Overall Confusion Matrix.
    - Feature Importance.
    """
    print(f"\n--- Fairness Auditing for {model_name} ---")

    # 1. Demographic Parity & Selection Rates (Manual Calculation)
    if 'Race' in df_raw.columns:
        race_group_data = df_raw.copy()
        race_group_data['Predicted_Loan_Approved'] = y_pred_full_dataset

        selection_rates = race_group_data.groupby('Race')['Predicted_Loan_Approved'].mean().reset_index()
        selection_rates.rename(columns={'Predicted_Loan_Approved': 'Selection_Rate'}, inplace=True)

        max_selection_rate = selection_rates['Selection_Rate'].max()
        min_selection_rate = selection_rates['Selection_Rate'].min()
        dpd = max_selection_rate - min_selection_rate

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        # Corrected line: Use 'color' instead of 'palette' for single color
        sns.barplot(x='Race', y='Selection_Rate', data=selection_rates, ax=ax1, color='skyblue')
        ax1.set_title(f"Predicted Loan Approval Rate by Race\n(Demographic Parity Difference = {dpd:.3f})")
        ax1.set_ylabel("Approval Rate")
        ax1.set_xlabel("Race Group")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig1.savefig("bias_visualization_selection_rate.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("- bias_visualization_selection_rate.png saved.")
    else:
        dpd = np.nan # If Race column is not present

    # 2. Confusion Matrix (Overall)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true_full, y_pred_full_dataset)
    ConfusionMatrixDisplay(cm, display_labels=['Denied', 'Approved']).plot(ax=ax2, cmap='Blues')
    ax2.set_title("Confusion Matrix - Loan Approval Predictions (Full Dataset)")
    plt.tight_layout()
    fig2.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("- confusion_matrix.png saved.")

    # 3. Feature Importance (Alternative to SHAP)
    print("Creating Feature Importance plot (SHAP was not available)...")
    if hasattr(mitigator, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_scaled_global_for_features.columns, # Corrected variable name here
            'importance': mitigator.feature_importances_
        }).sort_values('importance', ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title("Feature Importance - Loan Approval Model")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("- feature_importance.png saved.")
    else:
        print("⚠️  Model does not have 'feature_importances_' attribute. Cannot generate feature importance plot.")


    # 4. Additional bias analysis by multiple demographics (Predicted Approval Rate)
    existing_demographics = [col for col in demographic_features if col in df_raw.columns]

    if existing_demographics:
        fig, axes = plt.subplots(len(existing_demographics), 1, figsize=(12, 4*len(existing_demographics)))
        if len(existing_demographics) == 1:
            axes = [axes]

        for i, demo in enumerate(existing_demographics):
            temp_df_for_plot = df_raw.copy()
            temp_df_for_plot['Predicted_Loan_Approved'] = y_pred_full_dataset

            demo_rates = []
            demo_groups = temp_df_for_plot[demo].unique()
            if demo == 'Age_Group': # Specific order for Age_Group
                age_group_order = ['Under 25', '25-60', 'Over 60']
                demo_groups = sorted(demo_groups, key=lambda x: age_group_order.index(x) if x in age_group_order else len(age_group_order))


            for group in demo_groups:
                mask = temp_df_for_plot[demo] == group
                if mask.sum() > 0:
                    rate = temp_df_for_plot.loc[mask, 'Predicted_Loan_Approved'].mean()
                    demo_rates.append(rate)
                else:
                    demo_rates.append(0)

            axes[i].bar(demo_groups, demo_rates, color='lightcoral', alpha=0.7)
            axes[i].set_title(f'Predicted Loan Approval Rate by {demo}')
            axes[i].set_ylabel('Approval Rate')
            axes[i].set_ylim(0, 1)

            for j, rate in enumerate(demo_rates):
                axes[i].text(j, rate + 0.01, f'{rate:.3f}', ha='center', va='bottom')

            if demo in ['Race', 'Employment_Type', 'Education_Level', 'Citizenship_Status', 'Zip_Code_Group']:
                 plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig("demographic_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("- demographic_analysis.png saved.")
    return dpd # Return DPD for summary


# --- Main Execution ---
if __name__ == "__main__":
    # Define categorical columns
    categorical_cols = ['Gender', 'Race', 'Age_Group', 'Employment_Type', 'Education_Level',
                        'Citizenship_Status', 'Language_Proficiency', 'Disability_Status',
                        'Criminal_Record', 'Zip_Code_Group']

    # Load the main dataset
    print("Loading loan_access_dataset.csv...")
    df_raw_train = pd.read_csv('loan_access_dataset.csv')

    # Data Cleaning and Preprocessing for Training Data
    print("Cleaning and preprocessing training data...")
    X_scaled_full, y_full, original_ids_full, scaler, label_encoders = clean_and_preprocess(
        df_raw_train, categorical_cols, train_mode=True
    )
    
    # Store X_scaled_full for use in audit_fairness function (e.g., for feature importance column names)
    X_scaled_global_for_features = X_scaled_full.copy()


    # Split the training data for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

    # Train the model
    print("Training the Tuned Random Forest Classifier model...")
    mitigator = train_model(X_train, y_train)

    # Save the model and preprocessing tools
    joblib.dump(mitigator, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoders, LABEL_ENCODERS_FILE)
    print("Model and preprocessing tools saved.")

    # Evaluate the model on the internal test set
    print("\n--- Evaluating Model on Internal Test Split ---")
    y_pred_test_internal = mitigator.predict(X_test)
    accuracy_test_internal = accuracy_score(y_test, y_pred_test_internal)
    class_report_test_internal = classification_report(y_test, y_pred_test_internal)
    print(f"Accuracy on internal test split: {accuracy_test_internal:.3f}")
    print("Classification Report on internal test split:\n", class_report_test_internal)


    # Fairness Auditing and Bias Detection (using predictions on the full dataset for comprehensive audit)
    print("\n🎯 Generating bias visualizations (using predictions on full loan_access_dataset for comprehensive audit)...")
    y_pred_full_dataset = mitigator.predict(X_scaled_full)
    demographic_features = ['Gender', 'Age', 'Age_Group', 'Race', 'Disability_Status']
    
    # Pass X_scaled_global_for_features to audit_fairness for feature importance
    dpd_summary_value = audit_fairness(df_raw_train, y_full, y_pred_full_dataset, demographic_features, mitigator, X_scaled_global_for_features, model_name="Tuned Random Forest")


    # --- Generate submission.csv for the provided test.csv dataset ---
    print("\n--- Generating submission.csv for test.csv ---")
    test_df_raw = pd.read_csv('test.csv')

    # Preprocess test_df using the SAME scaler and label_encoders fitted on training data
    X_test_submission_scaled, _, _, _, _ = clean_and_preprocess(
        test_df_raw, categorical_cols, train_mode=False, scaler=scaler, label_encoders=label_encoders
    )

    # Ensure columns match training data features for prediction (important for consistency)
    missing_cols_in_test = set(X_scaled_full.columns) - set(X_test_submission_scaled.columns)
    for c in missing_cols_in_test:
        X_test_submission_scaled[c] = 0
    X_test_submission_scaled = X_test_submission_scaled[X_scaled_full.columns]

    # Make predictions on the scaled test data
    predictions_on_test_data = mitigator.predict(X_test_submission_scaled)

    submission_df = pd.DataFrame({
        'ID': test_df_raw['ID'],
        'LoanApproved': predictions_on_test_data # Now correctly 0 or 1
    })
    # Removed the mapping to 'Approved'/'Denied' to keep values as 0 or 1
    submission_df.to_csv('submission.csv', index=False)
    print("✅ submission.csv generated successfully for 'test.csv' with 0/1 predicted values.")

    # Final summary for the user
    print("\n--- AI Bias Bounty Hackathon Deliverables Completed ---")
    print("- **loan_model.py**: This script contains all data processing, model training, and fairness auditing steps.")
    print("- **submission.csv**: Model's output on the provided test dataset ('test.csv').")
    print("- **Visual Evidence of Bias**: Charts have been generated in the current directory:")
    print("  - `bias_visualization_selection_rate.png` (Predicted Approval rate by Race)")
    print("  - `confusion_matrix.png` (Overall Confusion Matrix)")
    print("  - `feature_importance.png` (Feature Importance Chart)")
    print("  - `demographic_analysis.png` (Predicted Approval Rates by various demographics)")
    print("\n**Model Performance Summary (on Full Loan_Access_Dataset Predictions):**")
    print(f"Total predictions: {len(y_pred_full_dataset)}")
    print(f"Overall predicted approval rate: {y_pred_full_dataset.mean():.3f}")
    print(f"Actual approval rate (full dataset): {y_full.mean():.3f}")
    
    # Recalculate and print Overall Accuracy on full dataset for clarity
    overall_accuracy_full_dataset = accuracy_score(y_full, y_pred_full_dataset)
    print(f"Overall Accuracy (full dataset): {overall_accuracy_full_dataset:.3f}")
    
    if not np.isnan(dpd_summary_value): # Check if DPD was calculated
        print(f"Demographic Parity Difference (Race): {dpd_summary_value:.3f}")
        print("Note: DPD closer to 0 indicates better fairness")