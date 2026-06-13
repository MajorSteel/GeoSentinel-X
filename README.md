<div align="center">
  
# GeoSentinel-X
### Foundation Models for Multi-Temporal Earth Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Next.js](https://img.shields.io/badge/Next.js-black?style=flat&logo=next.js&logoColor=white)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A next-generation Earth Observation platform combining multi-modal learning, change detection, forecasting, and geospatial LLMs.*

</div>

---

## The Core Vision

GeoSentinel-X is designed to significantly exceed the capabilities of existing Earth Observation datasets like SEN12MS. It is a fully automated, multi-petabyte scale system capable of answering:

* **What is on the ground?** (LULC & Semantic Segmentation)
* **What changed?** (Multi-Temporal Anomaly Detection)
* **Why did it change?** (GeoLLM Explainability)
* **What will likely happen next?** (Spatio-Temporal Forecasting)
* **How confident is the prediction?** (XAI & GradCAM)

---

## Project Architecture & Tree

The platform is designed to scale from local experimentation to Kubernetes-based global production.
For a deep dive into the system design, see [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md).

```text
geosentinel-x/
├── data/                      # Data lake for multi-modal sources
│   ├── raw/                   # Raw satellite imagery (S1, S2, DEM, ERA5)
│   ├── processed/             # Tiled Zarr/Parquet files chunked for ViT
│   └── catalogs/              # STAC catalogs for indexing
├── docs/                      # Architectural specs and research outlines
│   └── ARCHITECTURE.md        # Mermaid diagrams and database schema
├── mlops/                     # Deployment and scaling configurations
│   ├── docker/                # Multi-container local stack (FastAPI, PostGIS, Redis)
│   ├── kubernetes/            # Helm charts and GPU auto-scaling manifests
│   └── workflows/             # GitHub Actions / Kubeflow CI/CD
├── src/                       # Core Python intelligence backend
│   ├── api/                   # FastAPI backend for real-time inference
│   ├── data_pipeline/         # Dask-powered STAC ingestion and Xarray preprocessing
│   ├── downstream/            # Finetuned task heads (SegFormer, ConvLSTM, Siamese)
│   ├── explainability/        # PyTorch GradCAM and SHAP hooks
│   ├── geollm/                # HuggingFace RAG agent powered by Llama-3
│   └── models/                # ViT branches, Cross-Attention, and MAE pretraining
├── ui/                        # Next.js interactive dashboard
│   ├── src/app/               # Deck.gl & react-map-gl interactive maps
│   └── components/            # LLM Chat panels and layer controls
├── tests/                     # Unit and integration test suites
├── pyproject.toml             # Python dependencies and metadata
└── LICENSE                    # MIT License
```

---

## Capabilities

GeoSentinel-X relies on a multi-branch Vision Transformer (ViT) pre-trained via Masked Autoencoders (MAE) and self-supervised learning (DINOv2) to fuse:
1. **Sentinel-1 SAR** (VV, VH, Textures)
2. **Sentinel-2 Optical** (13 Bands, NDVI, NDWI)
3. **Copernicus DEM** (Elevation, Slope, Aspect)
4. **ERA5 Climate** (Rainfall, Temp, Moisture)

### Supported Downstream Tasks:
- **Task 1:** Land Use Land Cover (LULC) Classification (12 classes).
- **Task 2:** Sub-pixel Semantic Segmentation via SegFormer.
- **Task 3:** Multi-Temporal Change Detection (Deforestation, Urban Growth).
- **Task 4:** Future Land Cover Prediction (T+1 forecasting via ConvLSTM).

---

## The GeoLLM Copilot

Integrated directly into the Next.js UI is the **GeoLLM Copilot**. Powered by an LLM RAG agent (e.g., Llama 3), users can select a bounding box on the Deck.gl map and ask:
> *"Why is the vegetation decreasing in the eastern sector over the last 3 years?"*

The agent analyzes the underlying ViT latent representations, GradCAM heatmaps, and climate trends to provide natural-language analytical reports.

---

## Quick Start (Development)

### 1. Python Environment (Backend)
```bash
# Clone the repository
git clone https://github.com/MajorSteel/GeoSentinel-X.git
cd GeoSentinel-X

# Install dependencies (requires Python 3.10+)
pip install -e .[dev]

# Start the FastAPI Inference Engine
cd src/api
uvicorn main:app --reload --port 8000
```

### 2. Next.js Dashboard (Frontend)
```bash
cd ui
npm install

# Start the Deck.gl interface
npm run dev
```

### 3. Docker Compose (Full Stack)
To spin up the API, Redis cache, and PostGIS database locally:
```bash
docker-compose -f mlops/docker/docker-compose.yml up --build
```

---

## Success Criteria & Research Metrics
* LULC Accuracy >95%
* Segmentation IoU >90%
* Change Detection F1 >92%
* Future Prediction Accuracy >88%
* Global-scale inference latency < 2 seconds.

---

## License
This project is licensed under the [MIT License](LICENSE).
