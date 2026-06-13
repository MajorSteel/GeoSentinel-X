from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(
    title="GeoSentinel-X API",
    description="Inference API for Earth Observation Foundation Models",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    bbox: list[float]  # [min_lon, min_lat, max_lon, max_lat]
    start_date: str
    end_date: str
    task: str  # e.g., 'lulc', 'segmentation', 'change', 'forecast'

@app.get("/")
def read_root():
    return {"status": "GeoSentinel-X API is running"}

@app.post("/api/v1/predict")
async def predict_task(request: InferenceRequest) -> Dict[str, Any]:
    """
    Unified endpoint for running downstream GeoAI tasks.
    """
    # TODO: Implement model loading and inference logic
    return {
        "status": "processing",
        "task": request.task,
        "bbox": request.bbox,
        "message": "Inference queued successfully."
    }

@app.post("/api/v1/chat/query")
async def geollm_query(query: str, bbox: list[float]) -> Dict[str, Any]:
    """
    GeoLLM RAG query endpoint.
    """
    # TODO: Connect to Llama 3 / RAG agent
    return {
        "query": query,
        "response": "This area shows significant urban expansion over the last 5 years based on Sentinel-1 backscatter and Sentinel-2 NDVI trends."
    }
