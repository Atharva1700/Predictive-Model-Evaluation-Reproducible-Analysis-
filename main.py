"""
Predictive Model Evaluation
Regression + Classification models with reproducible pipeline and uncertainty quantification.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_auc_score
import json

# === REPRODUCIBLE SETUP ===
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# === GENERATE SYNTHETIC DATA ===
n_samples = 500
X_raw = np.random.randn(n_samples, 3) * 10
y_regression = 5 + 2*X_raw[:, 0] - 1.5*X_raw[:, 1] + 0.3*X_raw[:, 2] + np.random.randn(n_samples) * 2
y_classification = (y_regression > np.median(y_regression)).astype(int)

df = pd.DataFrame(X_raw, columns=['feature_1', 'feature_2', 'feature_3'])
df.to_csv('raw_data.csv', index=False)

print("=== DATA ===")
print(f"Samples: {n_samples}, Features: 3")
print(f"Saved raw data to raw_data.csv")

# === TRAIN/TEST SPLIT (REPRODUCIBLE) ===
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_raw, y_regression, y_classification,
    test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

# === REGRESSION MODEL ===
print("\n=== REGRESSION MODEL ===")
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_reg_train)
y_reg_pred = reg_model.predict(X_test_scaled)
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)

print(f"RMSE: {rmse:.4f}")
print(f"Coefficients: {dict(zip(['f1', 'f2', 'f3'], reg_model.coef_.round(3)))}")

# Residuals = uncertainty
residuals = y_reg_test - y_reg_pred
std_residuals = np.std(residuals)
print(f"Residual Std Dev (±1σ uncertainty): {std_residuals:.4f}")

# === CLASSIFICATION MODEL ===
print("\n=== CLASSIFICATION MODEL ===")
clf_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
clf_model.fit(X_train_scaled, y_clf_train)
y_clf_pred = clf_model.predict(X_test_scaled)
y_clf_proba = clf_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_clf_test, y_clf_pred)
auc = roc_auc_score(y_clf_test, y_clf_proba)
tn, fp, fn, tp = confusion_matrix(y_clf_test, y_clf_pred).ravel()

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {auc:.4f}")
print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# False positive rate = uncertainty/risk
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f"False Positive Rate: {fpr:.4f} (model risk)")

# === UNCERTAINTY QUANTIFICATION ===
print("\n=== UNCERTAINTY ===")
uncertainty = {
    "regression": {
        "rmse": float(rmse),
        "residual_std": float(std_residuals),
        "interpretation": f"Predictions ±{std_residuals:.2f} (68% confidence)"
    },
    "classification": {
        "accuracy": float(accuracy),
        "false_positive_rate": float(fpr),
        "interpretation": f"Correct {accuracy*100:.1f}% of time; {fpr*100:.1f}% false alarm risk"
    }
}

with open('uncertainty.json', 'w') as f:
    json.dump(uncertainty, f, indent=2)

print("\nUncertainty saved to uncertainty.json")

# === REPRODUCIBILITY CHECK ===
# Re-run and verify identical output
print("\n=== REPRODUCIBILITY CHECK ===")
reg_model_2 = LinearRegression()
reg_model_2.fit(X_train_scaled, y_reg_train)
y_reg_pred_2 = reg_model_2.predict(X_test_scaled)
rmse_2 = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_2))
print(f"RMSE (rerun): {rmse_2:.4f}")
print(f"Match original: {np.isclose(rmse, rmse_2)}")
