from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.routes import upload
import os
from pathlib import Path

# Create FastAPI app
app = FastAPI(
    title="Advanced Data Analytics Platform",
    description="AI-Powered Data Analytics with Dynamic ML and Gemini Integration",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(upload.router)

# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/")
async def home(request: Request):
    """Render the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})

# ============================================
# HEALTH CHECK ENDPOINT
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "upload_dir": str(UPLOAD_DIR),
        "models_dir": str(MODELS_DIR)
    }

# ============================================
# APP INFO ENDPOINT
# ============================================

@app.get("/info")
async def app_info():
    """Get application information."""
    return {
        "name": "Advanced Data Analytics Platform",
        "version": "2.0.0",
        "features": [
            "Dynamic dataset analysis",
            "Automatic KPI calculation",
            "Interactive visualizations",
            "ML model training & predictions",
            "AI-powered recommendations (Gemini)",
            "Intelligent chat assistant"
        ]
    }