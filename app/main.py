from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.routes import upload

app = FastAPI(title="Telco Tower Analytics", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(upload.router)

@app.get("/")
async def home(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})