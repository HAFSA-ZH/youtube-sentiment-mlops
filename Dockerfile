
# Image Python compatible avec le modèle entraîne
FROM python:3.14-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=7860

# Répertoire de travail
WORKDIR /app

# Installer les dépendances de production
COPY requirements-prod.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-prod.txt

# Copier le code de l'API
COPY src/ ./src/

# Copier le point d'entrée
COPY app_api.py .

# Copier le modèle final
COPY models/sentiment_svm_pipeline.joblib ./models/sentiment_svm_pipeline.joblib

# Port de l'application
EXPOSE 7860

# Démarrer FastAPI
CMD ["sh", "-c", "uvicorn app_api:app --host 0.0.0.0 --port ${PORT:-7860}"]
