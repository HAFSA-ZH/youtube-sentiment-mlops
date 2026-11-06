
"""
Point d'entrée de l'API pour le déploiement Docker.

L'application et le chargement du modèle sont définis
dans src.api.main.
"""

from src.api.main import app

__all__ = ["app"]
