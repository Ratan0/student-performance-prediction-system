import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


DATA_FILE = "data/students_clean.csv"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"


BASE_FEATURES = [
    "attendance",
    "previous_percentage",
    "unit_test_score",
    "midterm_score",
    "assignment_percentage",
    "homework_completion",
    "quiz_average",
    "study_hours",
    "class_participation",
    "academic_average",
    "engagement_score",
    "low_attendance",
    "low_academic_score",
    "overall_score"
]


TARGET = "at_risk"


def load_data():

    df = pd.read_csv(DATA_FILE)

    return df


def evaluate_model(model, X_test, y_test, name):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 60)

    print(name)

    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(cm)

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    }


def main():

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()

    X = df[BASE_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training rows:", len(X_train))
    print("Testing rows :", len(X_test))

    # ------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------

    logistic_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    logistic_pipeline.fit(
        X_train,
        y_train
    )

    logistic_results = evaluate_model(
        logistic_pipeline,
        X_test,
        y_test,
        "Logistic Regression"
    )

    # ------------------------------------------------
    # Random Forest
    # ------------------------------------------------

    rf_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    rf_pipeline.fit(
        X_train,
        y_train
    )

    rf_results = evaluate_model(
        rf_pipeline,
        X_test,
        y_test,
        "Random Forest"
    )

    # ------------------------------------------------
    # Cross Validation
    # ------------------------------------------------

    print("\nCross Validation")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        rf_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1"
    )

    print("F1 scores:", cv_scores)

    print(
        "Mean CV F1:",
        round(cv_scores.mean(), 4)
    )

    # ------------------------------------------------
    # Save best model
    # ------------------------------------------------

    results = pd.DataFrame([
        logistic_results,
        rf_results
    ])

    print("\nModel comparison:")
    print(results)

    best_model_name = results.loc[
        results["f1"].idxmax(),
        "model"
    ]

    if best_model_name == "Random Forest":

        best_model = rf_pipeline

    else:

        best_model = logistic_pipeline

    model_path = os.path.join(
        MODEL_DIR,
        "student_risk_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        "\nBest model:",
        best_model_name
    )

    print(
        "Saved model:",
        model_path
    )

    # ------------------------------------------------
    # Feature importance
    # ------------------------------------------------

    rf_model = rf_pipeline.named_steps["model"]

    importance = pd.DataFrame({
        "feature": BASE_FEATURES,
        "importance": rf_model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\nFeature Importance:")
    print(importance)

    importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    # ------------------------------------------------
    # Plot feature importance
    # ------------------------------------------------

    plt.figure(figsize=(10, 7))

    plt.barh(
        importance["feature"],
        importance["importance"]
    )

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.title(
        "Factors Affecting Student Risk"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        )
    )

    plt.close()

    # ------------------------------------------------
    # Save metrics
    # ------------------------------------------------

    results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_metrics.csv"
        ),
        index=False
    )


if __name__ == "__main__":
    main()