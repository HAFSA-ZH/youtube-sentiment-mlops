
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "reddit_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = OUTPUT_DIR / "train.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"

# 2. Paramètres du découpage
TEST_SIZE = 0.20
RANDOM_STATE = 42

def main():

    print("Chargement des données...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    # Vérifier les colonnes
    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Le dataset doit contenir les colonnes text et label."
        )

    # Garder uniquement les colonnes nécessaires
    df = df[["text", "label"]].copy()

    # Vérifier les labels
    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise"
    ).astype(int)

    if set(df["label"].unique()) != {-1, 0, 1}:
        raise ValueError("Les trois classes -1, 0 et 1 sont requises.")

    print(f"\nNombre total : {len(df)}")

    # 3. Séparer Train et Test
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # 4. Construire les DataFrames
    train_df = pd.DataFrame({
        "text": X_train,
        "label": y_train
    })

    test_df = pd.DataFrame({
        "text": X_test,
        "label": y_test
    })

    # 5. Sauvegarder les fichiers
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        TRAIN_PATH,
        index=False,
        encoding="utf-8"
    )

    test_df.to_csv(
        TEST_PATH,
        index=False,
        encoding="utf-8"
    )

    # 6. Afficher les résultats
    print("\n========== RESULTATS ==========")

    print(f"Train : {len(train_df)} commentaires")
    print(f"Test  : {len(test_df)} commentaires")

    print("\nDistribution Train :")
    print(
        train_df["label"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nDistribution Test :")
    print(
        test_df["label"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nFichiers enregistrés :")
    print(TRAIN_PATH)
    print(TEST_PATH)

    print("\nDécoupage terminé avec succès !")


if __name__ == "__main__":
    main()
