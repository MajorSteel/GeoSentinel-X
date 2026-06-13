# GeoSentinel-X: Foundation Models for Multi-Temporal Earth Intelligence

GeoSentinel-X is a next-generation Earth Observation Intelligence System designed to significantly exceed the capabilities of existing datasets and models like SEN12MS. It provides multi-modal, multi-temporal learning, change detection, semantic segmentation, and future forecasting using foundation models and GeoLLMs.

## Project Structure

- `data/`: Raw and processed satellite imagery (S1, S2, DEM, Climate).
- `src/`: Main source code.
  - `data_pipeline/`: Data ingestion, STAC integration, Dask/Zarr processing.
  - `models/`: Self-supervised foundation models (ViT, Fusion Transformers).
  - `downstream/`: Task-specific models (LULC, Segmentation, Change Detection, Forecasting).
  - `explainability/`: XAI tools (SHAP, GradCAM).
  - `geollm/`: RAG architecture with Llama 3 for spatial intelligence.
  - `api/`: FastAPI backend for inference and database interactions.
- `ui/`: Next.js dashboard with Mapbox/Deck.gl.
- `mlops/`: Kubernetes, Docker, and MLflow configurations.

## Quick Start (Development)

1. **Install dependencies:**
   ```bash
   pip install -e .[dev]
   ```

2. **Start the API:**
   ```bash
   cd src/api
   uvicorn main:app --reload
   ```

3. **Start the UI:**
   ```bash
   cd ui
   npm run dev
   ```

## License
MIT
