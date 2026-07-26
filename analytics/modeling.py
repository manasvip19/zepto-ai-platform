import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("LOADING CLEANED DATASET")
print("=" * 60)

df = pd.read_csv("analytics/data/titanic_cleaned.csv")

print(df.shape)

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df["survived"]

print("\nCLASS DISTRIBUTION")

print(y.value_counts())

print("\nPercentage")

print(
    y.value_counts(normalize=True)
    * 100
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

numeric_features = [
    "age",
    "fare",
    "sibsp",
    "parch",
    "pclass"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])
print("\nTRAINING LOGISTIC REGRESSION")

logistic_pipeline.fit(
    X_train,
    y_train
)

pred = logistic_pipeline.predict(X_test)

prob = logistic_pipeline.predict_proba(X_test)[:, 1]

print("\nLOGISTIC REGRESSION RESULTS")

print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1 Score :", f1_score(y_test, pred))
print("AUC      :", roc_auc_score(y_test, prob))

cm = confusion_matrix(
    y_test,
    pred
)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Logistic Regression")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "analytics/outputs/logistic_confusion_matrix.png"
)

plt.show()

decision_tree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=42
    ))
])

print("\n" + "=" * 60)
print("TRAINING DECISION TREE")
print("=" * 60)

decision_tree_pipeline.fit(X_train, y_train)

dt_pred = decision_tree_pipeline.predict(X_test)

dt_prob = decision_tree_pipeline.predict_proba(X_test)[:, 1]

print("\nDECISION TREE RESULTS")

print("Accuracy :", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred))
print("Recall   :", recall_score(y_test, dt_pred))
print("F1 Score :", f1_score(y_test, dt_pred))
print("AUC      :", roc_auc_score(y_test, dt_prob))

cm = confusion_matrix(y_test, dt_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Decision Tree")

plt.tight_layout()

plt.savefig(
    "analytics/outputs/decision_tree_cm.png"
)

plt.show()

cm = confusion_matrix(y_test, dt_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Decision Tree")

plt.tight_layout()

plt.savefig(
    "analytics/outputs/decision_tree_cm.png"
)

plt.show()

random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_estimators=200
    ))
])
print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

random_forest_pipeline.fit(X_train, y_train)

rf_pred = random_forest_pipeline.predict(X_test)

rf_prob = random_forest_pipeline.predict_proba(X_test)[:,1]
print("\nRANDOM FOREST RESULTS")

print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))
print("AUC      :", roc_auc_score(y_test, rf_prob))
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title("Random Forest")

plt.tight_layout()

plt.savefig(
    "analytics/outputs/random_forest_cm.png"
)

plt.show()
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred)
    ],
    "Precision": [
        precision_score(y_test, pred),
        precision_score(y_test, dt_pred),
        precision_score(y_test, rf_pred)
    ],
    "Recall": [
        recall_score(y_test, pred),
        recall_score(y_test, dt_pred),
        recall_score(y_test, rf_pred)
    ],
    "F1 Score": [
        f1_score(y_test, pred),
        f1_score(y_test, dt_pred),
        f1_score(y_test, rf_pred)
    ],
    "AUC": [
        roc_auc_score(y_test, prob),
        roc_auc_score(y_test, dt_prob),
        roc_auc_score(y_test, rf_prob)
    ]
})

print("\nMODEL COMPARISON")
print(comparison)
comparison.to_csv("analytics/outputs/model_comparison.csv", index=False)
print("\n" + "=" * 60)
print("ROC CURVE COMPARISON")
print("=" * 60)

lr_fpr, lr_tpr, _ = roc_curve(y_test, prob)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

plt.figure(figsize=(8,6))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC={roc_auc_score(y_test, prob):.3f})"
)

plt.plot(
    dt_fpr,
    dt_tpr,
    label=f"Decision Tree (AUC={roc_auc_score(y_test, dt_prob):.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC={roc_auc_score(y_test, rf_prob):.3f})"
)

plt.plot([0,1],[0,1],"k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()

plt.tight_layout()

plt.savefig("analytics/outputs/roc_curves.png")

plt.show()

baseline_rf = RandomForestClassifier(
    random_state=42
)

baseline_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", baseline_rf)
])

baseline_pipeline.fit(X_train, y_train)

baseline_pred = baseline_pipeline.predict(X_test)
balanced_rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

balanced_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", balanced_rf)
])

balanced_pipeline.fit(X_train, y_train)

balanced_pred = balanced_pipeline.predict(X_test)
smote_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])

smote_pipeline.fit(X_train, y_train)

smote_pred = smote_pipeline.predict(X_test)
imbalance_results = pd.DataFrame({
    "Method":[
        "Baseline",
        "Class Weight",
        "SMOTE"
    ],
    "Precision":[
        precision_score(y_test, baseline_pred),
        precision_score(y_test, balanced_pred),
        precision_score(y_test, smote_pred)
    ],
    "Recall":[
        recall_score(y_test, baseline_pred),
        recall_score(y_test, balanced_pred),
        recall_score(y_test, smote_pred)
    ],
    "F1":[
        f1_score(y_test, baseline_pred),
        f1_score(y_test, balanced_pred),
        f1_score(y_test, smote_pred)
    ]
})

print("\nIMBALANCE COMPARISON")
print(imbalance_results)

imbalance_results.to_csv(
    "analytics/outputs/imbalance_comparison.csv",
    index=False
)
print("\n" + "=" * 60)
print("GRID SEARCH - RANDOM FOREST")
print("=" * 60)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        oob_score=True,
        bootstrap=True
    ))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Parameters")
print(grid.best_params_)

best_pipeline = grid.best_estimator_

best_rf = best_pipeline.named_steps["classifier"]

print("\nOOB Score")
print(best_rf.oob_score_)

print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

regression_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X_reg = df[regression_features]
y_reg = df["fare"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

numeric = ["age", "sibsp", "parch", "pclass"]
categorical = ["sex", "embarked"]

reg_preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    )
])

reg_pipeline = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("regressor", LinearRegression())
])

reg_pipeline.fit(X_train_reg, y_train_reg)

pred_reg = reg_pipeline.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, pred_reg))
r2 = r2_score(y_test_reg, pred_reg)

n = len(y_test_reg)
p = X_train_reg.shape[1]

adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")
print(f"Adjusted R²: {adj_r2:.3f}")
residuals = y_test_reg - pred_reg

plt.figure(figsize=(8,5))

plt.scatter(pred_reg, residuals)

plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig("analytics/outputs/residual_plot.png")

plt.show()

print("\nSaving best pipeline...")

joblib.dump(
    best_pipeline,
    "analytics/models/best_pipeline.joblib"
)

loaded = joblib.load(
    "analytics/models/best_pipeline.joblib"
)

sample = X.iloc[[0]]

prediction = loaded.predict(sample)

print("Prediction on raw sample:", prediction)