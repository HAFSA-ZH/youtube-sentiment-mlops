
from time import perf_counter
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    PredictBatchRequest,
    PredictBatchResponse,
    BatchStatistics,
    Sentiment
)

from src.api.model_service import ModelService


# ========================================
# 1. CONFIGURATION
# ========================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description=(
        "API de classification des sentiments "
        "des commentaires YouTube."
    ),
    version="1.0.0"
)


# ========================================
# 2. CORS
# ========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# ========================================
# 3. CHARGEMENT DU MODELE
# ========================================

model_service = ModelService()


# ========================================
# 4. ENDPOINT HEALTH
# ========================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True,
        "model": "Calibrated Linear SVM"
    }


# ========================================
# 5. ENDPOINT PREDICT BATCH
# ========================================

@app.post(
    "/predict_batch",
    response_model=PredictBatchResponse
)
def predict_batch(request: PredictBatchRequest):

    start_time = perf_counter()

    try:

        # Prediction
        results = model_service.predict(request.comments)

        # Calcul des statistiques
        names = ["negative", "neutral", "positive"]

        counts = {
            name: 0 for name in names
        }

        for result in results:
            counts[result["sentiment"]] += 1

        total = len(results)

        percentages = {
            name: round(counts[name] / total * 100, 2)
            for name in names
        }

        inference_time = (
            perf_counter() - start_time
        ) * 1000

        statistics = BatchStatistics(
            total=total,
            counts=counts,
            percentages=percentages
        )

        logger.info(
            "Batch analyse : %d commentaires, %.2f ms",
            total,
            inference_time
        )

        return PredictBatchResponse(
            predictions=results,
            statistics=statistics,
            inference_time_ms=round(inference_time, 2)
        )

    except Exception as error:

        logger.exception("Erreur pendant la prediction")

        raise HTTPException(
            status_code=500,
            detail="Erreur interne pendant l'analyse des sentiments."
        ) from error
