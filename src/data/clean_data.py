
from pathlib import Path
import pandas as pd
import re
import html
import unicodedata

# 1. Chemins des fichiers
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "reddit.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reddit_clean.csv"

# 2. Fonction de nettoyage
def clean_text(text):
    if pd.isna(text):
        return ""

    text = html.unescape(str(text))

    # Supprimer les URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Supprimer les mentions Reddit et les usernames
    text = re.sub(
        r"(?i)(?:/u/|u/|@)[A-Za-z0-9_-]+",
        " ",
        text
    )

    # Supprimer les références aux subreddits
    text = re.sub(
        r"(?i)(?:/r/|r/)[A-Za-z0-9_-]+",
        " ",
        text
    )

    # Supprimer la ponctuation en conservant les emojis
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )

    # Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():

    print("Chargement du dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Nombre initial : {len(df)}")

    # 3. Identifier les colonnes
    text_col = next(
        (c for c in ["clean_comment", "text"] if c in df.columns),
        None
    )

    label_col = next(
        (c for c in ["category", "label"] if c in df.columns),
        None
    )

    if text_col is None or label_col is None:
        raise ValueError("Colonnes de texte ou de label introuvables.")

    # 4. Sélectionner et renommer les colonnes
    df = df[[text_col, label_col]].copy()

    df.columns = ["text", "label"]

    # 5. Nettoyer les valeurs manquantes
    df = df.dropna(subset=["text", "label"])

    # 6. Vérifier les labels
    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = df.dropna(subset=["label"])

    df = df[df["label"].isin([-1, 0, 1])]

    df["label"] = df["label"].astype(int)

    # 7. Nettoyer les commentaires
    df["text"] = df["text"].apply(clean_text)

    # 8. Supprimer les commentaires vides
    df = df[df["text"].str.strip() != ""]

    # 9. Supprimer les doublons exacts
    df = df.drop_duplicates(subset=["text", "label"])

    # 10. Réinitialiser les index
    df = df.reset_index(drop=True)

    # 11. Créer le dossier de destination
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 12. Sauvegarder les données
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    # 13. Afficher les statistiques
    print("\n========== RESULTATS ==========")

    print(f"Commentaires apres nettoyage : {len(df)}")

    print("\nColonnes finales :")
    print(df.columns.tolist())

    print("\nDistribution des sentiments :")
    print(df["label"].value_counts().sort_index())

    print("\nLongueur moyenne des commentaires :")
    print(df["text"].str.len().mean())

    print("\nExemples de commentaires :")
    print(df.head())

    print("\nFichier sauvegarde :")
    print(OUTPUT_PATH)

    print("\nNettoyage termine avec succes !")


if __name__ == "__main__":
    main()
