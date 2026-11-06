
from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# ========================================
# 1. CHEMINS
# ========================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ========================================
# 2. CHARGEMENT DES DONNEES
# ========================================

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

X_train = train["text"].fillna("").astype(str)
y_train = train["label"].astype(int)

X_test = test["text"].fillna("").astype(str)
y_test = test["label"].astype(int)

print("Train :", len(X_train))
print("Test :", len(X_test))

# ========================================
# 3. TF-IDF
# Parametres retenus lors de la recherche LR
# ========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    sublinear_tf=True,
    ngram_range=(1, 1),
    min_df=5,
    max_features=50000,
    max_df=1.0
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nDimensions TF-IDF :", X_train_tfidf.shape)

# ========================================
# 4. LOGISTIC REGRESSION
# Notre modele de reference
# ========================================

print("\n========== LOGISTIC REGRESSION ==========")

lr = LogisticRegression(
    C=4.0,
    max_iter=1000,
    solver="lbfgs",
    class_weight=None
)

lr.fit(X_train_tfidf, y_train)

lr_pred = lr.predict(X_test_tfidf)

print(classification_report(
    y_test,
    lr_pred,
    labels=[-1, 0, 1],
    target_names=["Negatif", "Neutre", "Positif"],
    zero_division=0
))

# ========================================
# 5. OPTIMISATION LINEAR SVM
# ========================================

print("\n========== OPTIMISATION SVM ==========")

svm = LinearSVC(random_state=42)

parameters = {
    "C": [0.1, 0.5, 1, 2, 4, 8],
    "class_weight": [None, "balanced"]
}

search = RandomizedSearchCV(
    estimator=svm,
    param_distributions=parameters,
    n_iter=12,
    scoring="f1_macro",
    cv=3,
    random_state=42,
    n_jobs=1,
    verbose=1
)

search.fit(X_train_tfidf, y_train)

print("\nMeilleurs parametres SVM :")
print(search.best_params_)

print("\nMeilleur F1 macro CV :")
print(round(search.best_score_, 4))

# ========================================
# 6. EVALUATION SVM NON CALIBRE
# ========================================

best_svm = search.best_estimator_

svm_pred = best_svm.predict(X_test_tfidf)

print("\n========== SVM NON CALIBRE ==========")

print(classification_report(
    y_test,
    svm_pred,
    labels=[-1, 0, 1],
    target_names=["Negatif", "Neutre", "Positif"],
    zero_division=0
))

# ========================================
# 7. CALIBRATION DES PROBABILITES
# ========================================

print("\n========== CALIBRATION SVM ==========")

calibrated_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            ngram_range=(1, 1),
            min_df=5,
            max_features=50000,
            max_df=1.0
        )
    ),
    (
        "clf",
        CalibratedClassifierCV(
            estimator=LinearSVC(
                C=search.best_params_["C"],
                class_weight=search.best_params_["class_weight"],
                random_state=42
            ),
            method="sigmoid",
            cv=3
        )
    )
])

calibrated_pipeline.fit(X_train, y_train)

calibrated_pred = calibrated_pipeline.predict(X_test)

print("\nRapport SVM calibre :")

print(classification_report(
    y_test,
    calibrated_pred,
    labels=[-1, 0, 1],
    target_names=["Negatif", "Neutre", "Positif"],
    zero_division=0
))

# ========================================
# 8. COMPARAISON FINALE
# ========================================

results = []

for name, predictions in [
    ("Logistic Regression", lr_pred),
    ("Linear SVM", svm_pred),
    ("Linear SVM calibre", calibrated_pred)
]:
    results.append({
        "Modele": name,
        "Accuracy (%)": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),
        "F1 macro": round(
            f1_score(y_test, predictions, average="macro"), 4
        )
    })

print("\n========== COMPARAISON FINALE ==========")

print(pd.DataFrame(results).to_string(index=False))

# ========================================
# 9. SAUVEGARDE SVM CALIBRE
# ========================================

joblib.dump(
    calibrated_pipeline,
    MODELS_DIR / "sentiment_svm_pipeline.joblib"
)

print("\nSVM calibre sauvegarde avec succes !")

# ========================================
# 10. VERIFICATION DES PROBABILITES
# ========================================

examples = [
    "I love this video, it is amazing",
    "This video is terrible",
    "This video is okay"
]

predictions = calibrated_pipeline.predict(examples)
probabilities = calibrated_pipeline.predict_proba(examples)

print("\n========== EXEMPLES ==========")

for text, prediction, probability in zip(
    examples, predictions, probabilities
):
    confidence = probability.max() * 100

    print("\nCommentaire :", text)
    print("Sentiment :", prediction)
    print("Confiance estimee :", round(confidence, 2), "%")
    print("Probabilites :", probability)

# ========================================
# 11. TEMPS D'INFERENCE
# ========================================

batch = X_test.iloc[:50].tolist()

calibrated_pipeline.predict(batch)

start = time.perf_counter()

for _ in range(20):
    calibrated_pipeline.predict(batch)
    calibrated_pipeline.predict_proba(batch)

end = time.perf_counter()

average_time = ((end - start) / 20) * 1000

print("\nTemps moyen pour 50 commentaires :")
print(round(average_time, 2), "ms")

print("\nExperience terminee !")
