
from pathlib import Path
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [-1, 0, 1]

# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

print("\n========== CHARGEMENT DES DONNEES ==========")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["text"].fillna("").astype(str)
y_train = train_df["label"].astype(int)

X_test = test_df["text"].fillna("").astype(str)
y_test = test_df["label"].astype(int)

print("Train :", len(X_train))
print("Test :", len(X_test))

# ==========================================
# 3. PIPELINE TF-IDF + LOGISTIC REGRESSION
# ==========================================

print("\n========== OPTIMISATION TF-IDF ==========")

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            max_features=50000
        )
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        )
    )
])

# ==========================================
# 4. HYPERPARAMETRES A TESTER
# ==========================================

parameters = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [2, 3, 5],
    "tfidf__max_df": [0.9, 1.0],
    "tfidf__max_features": [30000, 50000],
    "tfidf__sublinear_tf": [True, False],
    "clf__C": [0.5, 1.0, 2.0, 4.0],
    "clf__class_weight": [None, "balanced"]
}

# ==========================================
# 5. RANDOMIZED SEARCH
# ==========================================

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameters,
    n_iter=8,
    cv=3,
    scoring="f1_macro",
    random_state=42,
    n_jobs=1,
    verbose=1,
    refit=True
)

print("\nDebut de la recherche des meilleurs parametres...")

search.fit(X_train, y_train)

best_pipeline = search.best_estimator_

print("\nMeilleurs parametres :")
print(search.best_params_)

print("\nMeilleur F1 macro en validation croisee :")
print(round(search.best_score_, 4))

# ==========================================
# 6. EVALUATION LOGISTIC REGRESSION
# ==========================================

print("\n========== EVALUATION LOGISTIC REGRESSION ==========")

y_pred = best_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\nAccuracy :", round(accuracy * 100, 2), "%")
print("F1 macro :", round(f1_macro, 4))

print("\nRapport de classification :")

report = classification_report(
    y_test,
    y_pred,
    labels=LABELS,
    target_names=[
        "Negatif",
        "Neutre",
        "Positif"
    ],
    zero_division=0
)

print(report)

# ==========================================
# 7. COMPARAISON AVEC SVM ET NAIVE BAYES
# ==========================================

print("\n========== COMPARAISON DES MODELES ==========")

vectorizer = best_pipeline.named_steps["tfidf"]

X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    "Logistic Regression": best_pipeline.named_steps["clf"],

    "Linear SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=0.5
    )
}

results = []

for name, model in models.items():

    if name != "Logistic Regression":
        print(f"\nEntrainement : {name}")
        model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, predictions)

    f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    results.append({
        "Modele": name,
        "Accuracy (%)": round(acc * 100, 2),
        "F1 macro": round(f1, 4)
    })

results_df = pd.DataFrame(results)

print("\nResultats comparatifs :")
print(results_df.to_string(index=False))

# ==========================================
# 8. SAUVEGARDE DU MODELE ET DU VECTORIZER
# ==========================================

print("\n========== SAUVEGARDE ==========")

joblib.dump(
    best_pipeline,
    MODELS_DIR / "sentiment_pipeline.joblib"
)

joblib.dump(
    vectorizer,
    MODELS_DIR / "tfidf_vectorizer.joblib"
)

joblib.dump(
    best_pipeline.named_steps["clf"],
    MODELS_DIR / "sentiment_model.joblib"
)

print("Pipeline sauvegarde.")
print("Vectorizer sauvegarde.")
print("Modele sauvegarde.")

# ==========================================
# 9. MATRICE DE CONFUSION
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=LABELS
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negatif", "Neutre", "Positif"],
    yticklabels=["Negatif", "Neutre", "Positif"]
)

plt.xlabel("Sentiment predit")
plt.ylabel("Sentiment reel")
plt.title("Matrice de confusion")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

# ==========================================
# 10. TEMPS D'INFERENCE POUR 50 COMMENTAIRES
# ==========================================

print("\n========== TEMPS D'INFERENCE ==========")

batch = X_test.iloc[:50].tolist()

# Premier appel de chauffe
best_pipeline.predict(batch)

start = time.perf_counter()

for _ in range(20):
    best_pipeline.predict(batch)

end = time.perf_counter()

average_time = ((end - start) / 20) * 1000

print(f"Temps moyen pour 50 commentaires : {average_time:.2f} ms")

# ==========================================
# 11. BILAN
# ==========================================

print("\n========== BILAN FINAL ==========")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"F1 macro : {f1_macro:.4f}")
print(f"Temps inference (50 textes) : {average_time:.2f} ms")

print("\nEntrainement termine avec succes !")
