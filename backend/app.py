"""Main FastAPI Application serving REST API endpoints and the React + Plotly.js Web UI."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routes import centrality, dashboard, explainability, models, network, proteins
from src.config import PROJECT_ROOT

app = FastAPI(
    title="Centrality-Driven Graph Models for Cancer Protein Prediction",
    description=(
        "Research platform for prioritizing potentially cancer-associated proteins in "
        "Human PPI Networks using centrality metrics, functional annotations, Graph Neural Networks, "
        "and SHAP explainability."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(dashboard.router)
app.include_router(proteins.router)
app.include_router(centrality.router)
app.include_router(network.router)
app.include_router(models.router)
app.include_router(explainability.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "cancer-ppi-prediction"}


# Serve generated static figures & React frontend
figures_dir = PROJECT_ROOT / "results" / "figures"
if figures_dir.exists():
    app.mount("/figures", StaticFiles(directory=str(figures_dir)), name="figures")

frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="frontend_static")


@app.get("/")
def serve_index():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "API running. Frontend index.html not found."}
