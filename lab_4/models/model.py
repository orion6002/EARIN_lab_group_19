# Python dependencies, please execute:
# pip install numpy pandas matplotlib scikit-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report


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
# 4. DATA PREPROCESSING
# ==============================================================================

# Impute missing values with median from full training set
age_median = training.Age.median()
fare_median = training.Fare.median()

training.Age = training.Age.fillna(age_median)
training.Fare = training.Fare.fillna(fare_median)
training.dropna(subset=["Embarked"], inplace=True)

# Log-normalise fare to reduce skewness
training["norm_fare"] = np.log(training.Fare + 1)

# Convert Pclass to string so it becomes a categorical dummy
training.Pclass = training.Pclass.astype(str)

# One-hot encode categorical features
feature_cols = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "norm_fare",
    "Embarked",
    "cabin_adv",
    "cabin_multiple",
    "numeric_ticket",
    "name_title",
]
data_dummies = pd.get_dummies(training[feature_cols])

X = data_dummies
y = training["Survived"]

# ==============================================================================
# 5. TRAIN / TEST SPLIT  (80 / 20)
# ==============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# Scale continuous features
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
    "KNN": KNeighborsClassifier(),
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
    "l1_ratio": [0.0, 1.0],
    "C": np.logspace(-4, 4, 20),
}
clf_lr = GridSearchCV(
    LogisticRegression(max_iter=2000, solver="saga"),
    param_grid=lr_param_grid,
    cv=4,
    n_jobs=-1,
    verbose=0,
)
clf_lr.fit(X_train_scaled, y_train)
report_best(clf_lr, "Logistic Regression")

# --- KNN ---
knn_param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree"],
    "p": [1, 2],
}
clf_knn = GridSearchCV(
    KNeighborsClassifier(), param_grid=knn_param_grid, cv=4, n_jobs=-1, verbose=0
)
clf_knn.fit(X_train_scaled, y_train)
report_best(clf_knn, "KNN")

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
    "KNN": clf_knn.best_estimator_,
}

print("\n=== Test-set accuracy (held-out 20%) ===")
test_results = {}
for name, model in tuned_models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    test_results[name] = acc
    print(f"{name:25s}  accuracy={acc:.4f}")
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
        ("knn", clf_knn.best_estimator_),
    ],
    voting="soft",
)
voting_clf.fit(X_train_scaled, y_train)
y_pred_vc = voting_clf.predict(X_test_scaled)
vc_acc = accuracy_score(y_test, y_pred_vc)
print(f"\nVoting Classifier (soft) test accuracy: {vc_acc:.4f}")

# ==============================================================================
# 10. SUMMARY TABLE
# ==============================================================================

summary = pd.DataFrame(
    {
        "Model": list(tuned_models.keys()) + ["Voting Ensemble"],
        "Test Accuracy": [test_results[m] for m in tuned_models] + [vc_acc],
    }
)
print("\n=== Final Summary ===")
print(summary.sort_values("Test Accuracy", ascending=False).to_string(index=False))
