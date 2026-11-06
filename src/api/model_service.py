
from pathlib import Path
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "sentiment_svm_pipeline.joblib"
)

LABEL_NAMES = {
    -1: "negative",
    0: "neutral",
    1: "positive"
}


class ModelService:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modele introuvable : {MODEL_PATH}"
            )

        print("Chargement du modele SVM calibre...")

        self.model = joblib.load(MODEL_PATH)

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "Le modele ne fournit pas de probabilites."
            )

        self.classes = [
            int(label) for label in self.model.classes_
        ]

        if set(self.classes) != {-1, 0, 1}:
            raise ValueError(
                "Les classes attendues sont -1, 0 et 1."
            )

        print("Modele charge avec succes !")

    def predict(self, comments):

        predictions = self.model.predict(comments)

        probabilities = self.model.predict_proba(comments)

        results = []

        for comment, label, probability in zip(
            comments,
            predictions,
            probabilities
        ):

            label = int(label)

            probability_dict = {
                LABEL_NAMES[class_id]: float(prob)
                for class_id, prob in zip(
                    self.classes,
                    probability
                )
            }

            results.append({
                "comment": comment,
                "label": label,
                "sentiment": LABEL_NAMES[label],
                "confidence": float(max(probability)),
                "probabilities": probability_dict
            })

        return results
