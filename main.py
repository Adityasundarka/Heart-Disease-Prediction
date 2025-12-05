
# Heart Disease Prediction on heart.csv
# using Logistic Regression and Random Forest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from joblib import dump


# 1. Load dataset (heart.csv)
CSV_FILE = "heart.csv"     
TARGET_COL = "target"      

print("Loading dataset...")
df = pd.read_csv(CSV_FILE)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(list(df.columns))

print("\nMissing values in each column (before handling):")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("\nMissing values in each column (after handling):")
print(df.isnull().sum())

print(f"\nTarget column value counts – '{TARGET_COL}':")
print(df[TARGET_COL].value_counts())

# 2. Split features and target
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# 3. Optional: Correlation Heatmap

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 4. Helper: train & evaluate a model
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 40)
    print(f"Training & Evaluating: {name}")
    print("=" * 40)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\nAccuracy of {name}: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ROC Curve (requires predict_proba)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        print(f"\n{name} does not support probability outputs for ROC curve.")

    return acc, model


# 5. Logistic Regression model
log_reg_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

log_acc, log_reg_model = evaluate_model(
    "Logistic Regression", log_reg_model,
    X_train, y_train, X_test, y_test
)

# 6. Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=10
)

rf_acc, rf_model = evaluate_model(
    "Random Forest", rf_model,
    X_train, y_train, X_test, y_test
)

# 7. Save the trained Random Forest model
dump(rf_model, "heart_rf_model.pkl")
print("\nSaved Random Forest model as heart_rf_model.pkl")

# 8. Final summary
print("\n================ FINAL SUMMARY ================")
print(f"Logistic Regression Accuracy: {log_acc:.2f}%")
print(f"Random Forest Accuracy      : {rf_acc:.2f}%")

best_model_name = "Logistic Regression" if log_acc >= rf_acc else "Random Forest"
print(f"\nBest model on this dataset: {best_model_name}")
print("✅ Done.")