# Python dependencies, please execute:
# pip install numpy pandas matplotlib scikit-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Load data from csv file
training = pd.read_csv("./datasets/train.csv")

# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================


def engineer_features(df):
    """Apply feature engineering to a dataframe."""
    df = df.copy()

    # Number of cabins assigned (0 if none)
    df["cabin_multiple"] = df.Cabin.apply(
        lambda x: 0 if pd.isna(x) else len(x.split(" "))
    )

    # First letter of cabin (deck indicator)
    df["cabin_adv"] = df.Cabin.apply(lambda x: str(x)[0])

    # Whether ticket number is purely numeric
    df["numeric_ticket"] = df.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)

    # Title extracted from passenger name
    df["name_title"] = df.Name.apply(lambda x: x.split(",")[1].split(".")[0].strip())

    return df


training = engineer_features(training)

# ==============================================================================
# 4. TRAIN / TEST SPLIT  (80 / 20)
# ==============================================================================

feature_cols = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "cabin_adv",
    "cabin_multiple",
    "numeric_ticket",
    "name_title",
]

X_raw = training[feature_cols]
y = training["Survived"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=1, stratify=y
)

print(f"\nTrain size: {X_train_raw.shape[0]} | Test size: {X_test_raw.shape[0]}")

# ==============================================================================
# 5. DATA PREPROCESSING
# ==============================================================================

# The preprocessing statistics are computed only on the training set to avoid
# data leakage from the test set.
X_train_prepared = X_train_raw.copy()
X_test_prepared = X_test_raw.copy()

age_median = X_train_prepared["Age"].median()
fare_median = X_train_prepared["Fare"].median()
embarked_mode = X_train_prepared["Embarked"].mode()[0]

X_train_prepared["Age"] = X_train_prepared["Age"].fillna(age_median)
X_test_prepared["Age"] = X_test_prepared["Age"].fillna(age_median)

X_train_prepared["Fare"] = X_train_prepared["Fare"].fillna(fare_median)
X_test_prepared["Fare"] = X_test_prepared["Fare"].fillna(fare_median)

X_train_prepared["Embarked"] = X_train_prepared["Embarked"].fillna(embarked_mode)
X_test_prepared["Embarked"] = X_test_prepared["Embarked"].fillna(embarked_mode)

# Log-normalise fare to reduce skewness.
X_train_prepared["norm_fare"] = np.log(X_train_prepared["Fare"] + 1)
X_test_prepared["norm_fare"] = np.log(X_test_prepared["Fare"] + 1)

X_train_prepared = X_train_prepared.drop(columns=["Fare"])
X_test_prepared = X_test_prepared.drop(columns=["Fare"])

# Convert Pclass to string so it becomes a categorical dummy.
X_train_prepared["Pclass"] = X_train_prepared["Pclass"].astype(str)
X_test_prepared["Pclass"] = X_test_prepared["Pclass"].astype(str)

# One-hot encode categorical features after the split, then align test columns
# with the training columns.
X_train = pd.get_dummies(X_train_prepared)
X_test = pd.get_dummies(X_test_prepared)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Scale continuous features using statistics learned only from the training set.
scaler = StandardScaler()
continuous_cols = ["Age", "SibSp", "Parch", "norm_fare"]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

# ==============================================================================
# 6. BASELINE MODEL COMPARISON  (4-fold cross-validation)
# ==============================================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": tree.DecisionTreeClassifier(random_state=1),
    "SVM": SVC(random_state=1),
}

print("\n=== Baseline CV Accuracy (4-fold) ===")
baseline_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=4, scoring="accuracy"
    )
    baseline_results[name] = cv_scores.mean()
    print(f"{name:25s}  mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")

# ==============================================================================
# 7. HYPERPARAMETER TUNING  (GridSearchCV, 4-fold)
# ==============================================================================


def report_best(clf, name):
    """Print GridSearchCV best score and params."""
    print(f"\n[{name}]")
    print(f"  Best CV score : {clf.best_score_:.4f}")
    print(f"  Best params   : {clf.best_params_}")


# --- Logistic Regression ---
lr_param_grid = {
    "C": np.logspace(-4, 4, 20),
    "penalty": ["l1", "l2"],
}
clf_lr = GridSearchCV(
    LogisticRegression(max_iter=2000, solver="saga"),
    param_grid=lr_param_grid,
    cv=4,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
)
clf_lr.fit(X_train_scaled, y_train)
report_best(clf_lr, "Logistic Regression")

# --- SVM ---
svm_param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
}
clf_svm = GridSearchCV(
    SVC(probability=True, random_state=1),
    param_grid=svm_param_grid,
    cv=4,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
)
clf_svm.fit(X_train_scaled, y_train)
report_best(clf_svm, "SVM")

# --- Decision Tree ---
dt_param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 5, 10],
}
clf_dt = GridSearchCV(
    tree.DecisionTreeClassifier(random_state=1),
    param_grid=dt_param_grid,
    cv=4,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
)
clf_dt.fit(X_train_scaled, y_train)
report_best(clf_dt, "Decision Tree")

# ==============================================================================
# 8. FINAL EVALUATION ON HELD-OUT TEST SET
# ==============================================================================

tuned_models = {
    "Logistic Regression": clf_lr.best_estimator_,
    "Decision Tree": clf_dt.best_estimator_,
    "SVM": clf_svm.best_estimator_,
}

print("\n=== Test-set performance (held-out 20%) ===")
test_results = {}
for name, model in tuned_models.items():
    y_pred = model.predict(X_test_scaled)
    test_results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
    }
    print(f"\n{name}")
    print(f"Accuracy : {test_results[name]['Accuracy']:.4f}")
    print(f"Precision: {test_results[name]['Precision']:.4f}")
    print(f"Recall   : {test_results[name]['Recall']:.4f}")
    print(f"F1-score : {test_results[name]['F1-score']:.4f}")
    print(
        classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"])
    )

# ==============================================================================
# 9. SOFT-VOTING ENSEMBLE
# ==============================================================================

voting_clf = VotingClassifier(
    estimators=[
        ("lr", clf_lr.best_estimator_),
        ("dt", clf_dt.best_estimator_),
        ("svm", clf_svm.best_estimator_),
    ],
    voting="soft",
)
voting_clf.fit(X_train_scaled, y_train)
y_pred_vc = voting_clf.predict(X_test_scaled)
voting_results = {
    "Accuracy": accuracy_score(y_test, y_pred_vc),
    "Precision": precision_score(y_test, y_pred_vc),
    "Recall": recall_score(y_test, y_pred_vc),
    "F1-score": f1_score(y_test, y_pred_vc),
}
print("\nVoting Classifier (soft)")
print(f"Accuracy : {voting_results['Accuracy']:.4f}")
print(f"Precision: {voting_results['Precision']:.4f}")
print(f"Recall   : {voting_results['Recall']:.4f}")
print(f"F1-score : {voting_results['F1-score']:.4f}")
print(
    classification_report(
        y_test, y_pred_vc, target_names=["Not Survived", "Survived"]
    )
)

# ==============================================================================
# 10. SUMMARY TABLE
# ==============================================================================

summary_rows = []

for model_name, metrics in test_results.items():
    summary_rows.append(
        {
            "Model": model_name,
            "Test Accuracy": metrics["Accuracy"],
            "Test Precision": metrics["Precision"],
            "Test Recall": metrics["Recall"],
            "Test F1-score": metrics["F1-score"],
        }
    )

summary_rows.append(
    {
        "Model": "Voting Ensemble",
        "Test Accuracy": voting_results["Accuracy"],
        "Test Precision": voting_results["Precision"],
        "Test Recall": voting_results["Recall"],
        "Test F1-score": voting_results["F1-score"],
    }
)

summary = pd.DataFrame(summary_rows)
summary.to_csv(FIGURES_DIR / "final_model_results.csv", index=False)
print("\n=== Final Summary ===")
print(summary.sort_values("Test F1-score", ascending=False).to_string(index=False))
print(f"\nFinal results table saved to {FIGURES_DIR / 'final_model_results.csv'}")

# ==============================================================================
# 11. CONFUSION MATRIX FOR BEST MODEL
# ==============================================================================

best_model_name = summary.sort_values("Test F1-score", ascending=False).iloc[0]["Model"]

if best_model_name == "Voting Ensemble":
    best_model = voting_clf
else:
    best_model = tuned_models[best_model_name]

ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test_scaled,
    y_test,
    display_labels=["Not Survived", "Survived"],
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrix_best_model.png")
plt.close()

print(
    f"\nConfusion matrix for the best model ({best_model_name}) saved to "
    f"{FIGURES_DIR / 'confusion_matrix_best_model.png'}"
)
