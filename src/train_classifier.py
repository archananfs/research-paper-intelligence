
import pandas as pd
import joblib
import json

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "arxiv_classification_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["abstract", "primary_category"])
df["abstract"] = df["abstract"].astype(str).str.strip()

df = df[df["abstract"] != ""]

X = df["abstract"]
y = df["primary_category"]

print("Dataset shape:", df.shape)
print("\nCategory distribution:")
print(y.value_counts())


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    )
}


# ==========================================
# 5. TRAIN AND EVALUATE MODELS
# ==========================================

results = {}
trained_models = {}

for model_name, classifier in models.items():

    print("\n" + "=" * 50)
    print("Training:", model_name)
    print("=" * 50)

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            )
        ),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(report)

    results[model_name] = accuracy
    trained_models[model_name] = pipeline

    # Save classification report
    report_dict = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    with open(
        REPORT_DIR / f"{model_name.lower().replace(' ', '_')}_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report_dict, file, indent=4)


# ==========================================
# 6. COMPARE MODEL PERFORMANCE
# ==========================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")

best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)


# ==========================================
# 7. SAVE BEST MODEL
# ==========================================

model_path = MODEL_DIR / "best_research_classifier.pkl"

joblib.dump(best_model, model_path)

print("\nBest model saved to:", model_path)


# ==========================================
# 8. GENERATE CONFUSION MATRIX
# ==========================================

best_predictions = best_model.predict(X_test)


# ==========================================
# ERROR ANALYSIS - DETAILED
# ==========================================

error_analysis = pd.DataFrame({
    "paper_id": df.loc[X_test.index, "paper_id"].values,
    "title": df.loc[X_test.index, "title"].values,
    "abstract": X_test.values,
    "actual_category": y_test.values,
    "predicted_category": best_predictions
})

errors = error_analysis[
    error_analysis["actual_category"]
    != error_analysis["predicted_category"]
].copy()

# Save all errors
errors.to_csv(
    REPORT_DIR / "misclassified_papers.csv",
    index=False,
    encoding="utf-8"
)

print("\n--- ERROR ANALYSIS ---")
print("Total incorrect predictions:", len(errors))

# Most common error pairs
error_pairs = (
    errors.groupby(
        ["actual_category", "predicted_category"]
    )
    .size()
    .sort_values(ascending=False)
)

print("\nMost common classification mistakes:")
print(error_pairs.head(15))

# Inspect examples of the most common mistake
print("\n--- EXAMPLE MISCLASSIFIED PAPERS ---")

for (actual, predicted), count in error_pairs.head(3).items():

    print(f"\nActual: {actual} | Predicted: {predicted}")
    print("Number of errors:", count)

    examples = errors[
        (errors["actual_category"] == actual)
        & (errors["predicted_category"] == predicted)
    ].head(3)

    for _, row in examples.iterrows():
        print("\nTitle:", row["title"])
        print("Abstract:", row["abstract"][:400], "...")

print("\nSaved detailed errors to:")
print(REPORT_DIR / "misclassified_papers.csv")

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(9, 7))

disp.plot(
    cmap="Blues",
    xticks_rotation=45,
    ax=ax,
    colorbar=False
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()

confusion_path = REPORT_DIR / "confusion_matrix.png"

plt.savefig(confusion_path, dpi=300)
plt.close()

print("Confusion matrix saved to:", confusion_path)


# ==========================================
# 9. SAVE MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Accuracy"]
)

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\nModel comparison:")
print(comparison.to_string(index=False))


# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)

print("Best model:", best_model_name)
print("Best accuracy:", round(results[best_model_name], 4))
print("Saved model:", model_path)
print("Reports directory:", REPORT_DIR)
