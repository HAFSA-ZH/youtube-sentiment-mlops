
from pathlib import Path
import requests
import pandas as pd

# 1. URL du dataset
URL = (
    "https://raw.githubusercontent.com/"
    "Himanshu-1703/reddit-sentiment-analysis/"
    "refs/heads/main/data/reddit.csv"
)

# 2. Déterminer le dossier du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 3. Définir le chemin de sauvegarde
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "reddit.csv"

# 4. Télécharger le dataset
try:
    print("Telechargement du dataset Reddit...")

    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    # Sauvegarder le fichier CSV original
    CSV_PATH.write_bytes(response.content)

    print("Telechargement termine avec succes !")
    print(f"Fichier sauvegarde : {CSV_PATH}")

    # 5. Charger le dataset
    df = pd.read_csv(CSV_PATH)

    # 6. Afficher les statistiques
    print("\n========== STATISTIQUES ==========")

    print(f"Nombre de commentaires : {len(df)}")
    print(f"Nombre de colonnes : {len(df.columns)}")

    print("\nColonnes disponibles :")
    print(df.columns.tolist())

    print("\nDistribution des sentiments :")

    label_column = next(
        (col for col in ["category", "label", "sentiment"]
         if col in df.columns),
        None
    )

    if label_column:
        print(df[label_column].value_counts().sort_index())
    else:
        print("Colonne des labels introuvable.")

    print("\nPremiers commentaires :")
    print(df.head())

    print("\n==================================")

except requests.exceptions.RequestException as e:
    print(f"Erreur de telechargement : {e}")

except Exception as e:
    print(f"Erreur : {e}")
