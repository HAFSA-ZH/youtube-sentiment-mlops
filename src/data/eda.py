
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chemins
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "reddit_clean.csv"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 2. Charger les données
df = pd.read_csv(DATA_PATH)

# 3. Préparer les textes
df["text"] = df["text"].fillna("").astype(str)

df["char_length"] = df["text"].str.len()
df["word_length"] = df["text"].str.split().str.len()

# 4. Statistiques générales
print("\n========== ANALYSE EXPLORATOIRE ==========")

print("\nDimensions du dataset :")
print(df.shape)

print("\nValeurs manquantes :")
print(df.isnull().sum())

print("\nNombre de doublons :")
print(df.duplicated(subset=["text", "label"]).sum())

# 5. Distribution des sentiments
print("\nDistribution des classes :")

counts = df["label"].value_counts().reindex(
    [-1, 0, 1], fill_value=0
)

percentages = df["label"].value_counts(
    normalize=True
).reindex([-1, 0, 1], fill_value=0) * 100

distribution = pd.DataFrame({
    "Nombre": counts,
    "Pourcentage": percentages.round(2)
})

print(distribution)

# 6. Longueur des commentaires
print("\nStatistiques de longueur (caractères) :")
print(df["char_length"].describe())

print("\nStatistiques de longueur (mots) :")
print(df["word_length"].describe())

# 7. Graphique 1 : distribution des sentiments
sentiment_names = {
    -1: "Négatif",
    0: "Neutre",
    1: "Positif"
}

plot_data = distribution.reset_index()
plot_data["sentiment"] = plot_data["label"].map(
    sentiment_names
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=plot_data,
    x="sentiment",
    y="Nombre"
)

plt.title("Distribution des sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Nombre de commentaires")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "class_distribution.png",
    dpi=300
)

plt.close()

# 8. Graphique 2 : longueur des commentaires
plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="char_length",
    bins=50,
    kde=True
)

plt.title("Distribution de la longueur des commentaires")
plt.xlabel("Nombre de caractères")
plt.ylabel("Fréquence")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "text_length_distribution.png",
    dpi=300
)

plt.close()

# 9. Graphique 3 : boxplot des longueurs
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="char_length"
)

plt.title("Répartition des longueurs des commentaires")
plt.xlabel("Nombre de caractères")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "text_length_boxplot.png",
    dpi=300
)

plt.close()

print("\nGraphiques enregistrés dans :")
print(FIGURES_DIR)

print("\nAnalyse exploratoire terminée !")
