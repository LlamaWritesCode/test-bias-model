# generate_bias_visuals.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
import shap
import joblib
import os

# Check if model files exist, if not create a simple model
model_files_exist = all(os.path.exists(f) for f in ["mitigated_model.pkl", "scaler.pkl", "label_encoders.pkl"])

# Load dataset
df = pd.read_csv("loan_access_dataset.csv")
df['Loan_Approved'] = df['Loan_Approved'].map({'Approved': 1, 'Denied': 0})

categorical_cols = ['Gender', 'Race', 'Age_Group', 'Employment_Type', 'Education_Level',
                    'Citizenship_Status', 'Language_Proficiency', 'Disability_Status',
                    'Criminal_Record', 'Zip_Code_Group']

if not model_files_exist:
    print("⚠️  Model files not found. Creating and training a new model...")
    
    # Prepare data
    df_encoded = df.copy()
    label_encoders = {}
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Prepare features and target
    X = df_encoded.drop(columns=['ID', 'Loan_Approved'])
    y = df_encoded['Loan_Approved']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Train a simple model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    mitigator = RandomForestClassifier(n_estimators=100, random_state=42)
    mitigator.fit(X_train, y_train)
    
    # Save the model and preprocessing tools
    joblib.dump(mitigator, "mitigated_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")
    
    print("✅ Model trained and saved!")
    
else:
    print("📁 Loading existing model files...")
    # Load model and preprocessing tools
    mitigator = joblib.load("mitigated_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    
    # Prepare data
    df_encoded = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df_encoded[col] = label_encoders[col].transform(df[col])
    
    X = df_encoded.drop(columns=['ID', 'Loan_Approved'])
    y = df_encoded['Loan_Approved']
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

# Make predictions
y_pred = mitigator.predict(X_scaled)

print("🎯 Generating bias visualizations...")

# 1. Demographic Parity & Selection Rates
if 'Race' in df.columns:
    race_group = df['Race']
    metric_frame = MetricFrame(metrics={"selection_rate": selection_rate},
                               y_true=y,
                               y_pred=y_pred,
                               sensitive_features=race_group)
    dpd = demographic_parity_difference(y, y_pred, sensitive_features=race_group)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    metric_frame.by_group['selection_rate'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title(f"Selection Rate by Race\n(Demographic Parity Difference = {dpd:.3f})")
    ax1.set_ylabel("Approval Rate")
    ax1.set_xlabel("Race Group")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1.savefig("bias_visualization_selection_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

# 2. Confusion Matrix
fig2, ax2 = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Denied', 'Approved']).plot(ax=ax2, cmap='Blues')
ax2.set_title("Confusion Matrix - Loan Approval Predictions")
plt.tight_layout()
fig2.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. SHAP Feature Importance
try:
    print("🧠 Calculating SHAP values (this may take a moment)...")
    # Use a smaller sample for SHAP to improve performance
    sample_size = min(100, len(X_scaled))
    X_sample = X_scaled.sample(n=sample_size, random_state=42)
    
    explainer = shap.Explainer(mitigator.predict_proba, X_scaled)
    shap_values = explainer(X_sample)
    
    # Create SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values[:, :, 1], show=False)  # Show positive class
    plt.title("SHAP Feature Importance - Loan Approval Model")
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
except Exception as e:
    print(f"⚠️  SHAP visualization failed: {e}")
    print("Creating alternative feature importance plot...")
    
    # Alternative: Use model's built-in feature importance
    if hasattr(mitigator, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_scaled.columns,
            'importance': mitigator.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title("Feature Importance - Loan Approval Model")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

# 4. Additional bias analysis by multiple demographics
demographics_to_analyze = ['Gender', 'Age', 'Age_Group', 'Race', 'Disability_Status']
existing_demographics = [col for col in demographics_to_analyze if col in df.columns]

if existing_demographics:
    fig, axes = plt.subplots(len(existing_demographics), 1, figsize=(12, 4*len(existing_demographics)))
    if len(existing_demographics) == 1:
        axes = [axes]
    
    for i, demo in enumerate(existing_demographics):
        # Calculate selection rates by demographic group
        demo_rates = []
        demo_groups = df[demo].unique()
        
        for group in demo_groups:
            mask = df[demo] == group
            if mask.sum() > 0:  # Only if there are samples in this group
                rate = y_pred[mask].mean()
                demo_rates.append(rate)
            else:
                demo_rates.append(0)
        
        axes[i].bar(demo_groups, demo_rates, color='lightcoral', alpha=0.7)
        axes[i].set_title(f'Loan Approval Rate by {demo}')
        axes[i].set_ylabel('Approval Rate')
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for j, rate in enumerate(demo_rates):
            axes[i].text(j, rate + 0.01, f'{rate:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        if len(str(demo_groups[0])) > 10:
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("demographic_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

print("✅ All visualizations saved:")
print("- bias_visualization_selection_rate.png")
print("- confusion_matrix.png")
print("- shap_feature_importance.png (or feature_importance.png)")
print("- demographic_analysis.png")

# Print summary statistics
print(f"\n📊 Model Performance Summary:")
print(f"Total predictions: {len(y_pred)}")
print(f"Overall approval rate: {y_pred.mean():.3f}")
print(f"Actual approval rate: {y.mean():.3f}")
print(f"Accuracy: {(y_pred == y).mean():.3f}")

if 'Race' in df.columns:
    print(f"Demographic Parity Difference (Race): {dpd:.3f}")
    print("Note: DPD closer to 0 indicates better fairness")