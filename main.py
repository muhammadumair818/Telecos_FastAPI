from fastapi import FastAPI
from app.services.upload import router as upload_router

app = FastAPI(
    title="Telecos Tower Analysis API",
    description="API for telecom tower productivity analysis, ML predictions, and AI insights.",
    version="1.0.0"
)

# Include the upload and analysis routes
app.include_router(upload_router, tags=["Upload & Analysis"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Telecos FastAPI. Visit /docs for the API documentation."}