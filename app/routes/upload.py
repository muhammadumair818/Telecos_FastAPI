from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.file_handler import save_uploaded_file, load_dataframe
from app.services.analysis import compute_kpis, generate_plots
from app.services.ml_model import train_revenue_model, train_cost_model, train_classification_model
import uuid
import pandas as pd

router = APIRouter()

# In-memory storage for demo (use database in production)
STORAGE = {}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload, save it, and return a session ID."""
    try:
        # Save file to data directory
        file_path = save_uploaded_file(file)
        
        # Load and preprocess dataframe
        df = load_dataframe(file_path)
        if df is None:
            raise HTTPException(status_code=400, detail="Invalid file format or missing columns.")
        
        # Generate a session ID
        session_id = str(uuid.uuid4())
        STORAGE[session_id] = {
            "file_path": file_path,
            "dataframe": df.to_dict(orient="records")  # store serializable
        }
        
        # Compute KPIs and plots (store them for later use)
        kpis = compute_kpis(df)
        plots = generate_plots(df)
        
        # Store also in session storage
        STORAGE[session_id]["kpis"] = kpis
        STORAGE[session_id]["plots"] = plots
        
        # Train ML models if enough data
        if len(df) >= 2:
            rev_model, rev_score, _ = train_revenue_model(df)
            cost_model, cost_score, _ = train_cost_model(df)
            clf_model, clf_acc, _ = train_classification_model(df)
            # In a real app, save models to disk and store path in session
            # For demo, we just store scores
            STORAGE[session_id]["rev_score"] = rev_score
            STORAGE[session_id]["cost_score"] = cost_score
            STORAGE[session_id]["clf_acc"] = clf_acc
        
        return {"session_id": session_id, "kpis": kpis, "plots": plots}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/{session_id}")
async def analyze(session_id: str):
    """Return analysis results for a given session."""
    if session_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "kpis": STORAGE[session_id].get("kpis"),
        "plots": STORAGE[session_id].get("plots"),
        "rev_score": STORAGE[session_id].get("rev_score"),
        "cost_score": STORAGE[session_id].get("cost_score"),
        "clf_acc": STORAGE[session_id].get("clf_acc")
    }

@router.post("/predict/{session_id}")
async def predict(session_id: str, active_tenants: int = 10, energy_cost: float = 1000.0, opex: float = 2000.0):
    """Predict revenue using the trained model."""
    if session_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # In production, load the actual model from disk
    # For demo, we'll retrain quickly (inefficient, but just for illustration)
    df_data = STORAGE[session_id]["dataframe"]
    df = pd.DataFrame(df_data)
    
    from app.services.ml_model import train_revenue_model
    model, score, _ = train_revenue_model(df)
    if model is None:
        raise HTTPException(status_code=400, detail="Model could not be trained.")
    
    pred = model.predict([[active_tenants, energy_cost, opex]])[0]
    return {"predicted_revenue": pred}