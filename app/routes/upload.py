from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from app.utils.file_handler import save_uploaded_file, load_dataframe
from app.services.analysis import analyze_dataset, get_ai_recommendations_dynamic, get_chat_response
from app.services.ml_model import train_models_dynamic, make_prediction, get_available_models_info
import uuid
import pandas as pd
import json
from typing import Optional, Dict, Any

router = APIRouter()

# In-memory storage for sessions (use database in production)
SESSION_STORAGE: Dict[str, Dict[str, Any]] = {}

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_session_data(session_id: str) -> Dict[str, Any]:
    """Get session data or raise 404."""
    if session_id not in SESSION_STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    return SESSION_STORAGE[session_id]

def store_session_data(session_id: str, data: Dict[str, Any]) -> None:
    """Store data in session."""
    SESSION_STORAGE[session_id] = data

# ============================================
# MAIN UPLOAD ENDPOINT
# ============================================

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload CSV/Excel file, analyze dataset dynamically, train ML models.
    Returns session_id and all analysis results.
    """
    try:
        # Save uploaded file
        file_path = save_uploaded_file(file)
        
        # Load dataframe
        df = load_dataframe(file_path)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Invalid file format or empty file")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Run dynamic analysis
        analysis_results = analyze_dataset(df)
        
        # Train ML models dynamically
        ml_results = train_models_dynamic(df, analysis_results.get("schema", {}))
        
        # Store everything in session
        session_data = {
            "file_path": file_path,
            "dataframe": df.to_dict(orient="records"),
            "analysis": analysis_results,
            "ml_results": ml_results,
            "chat_history": [],
            "df_shape": df.shape,
            "df_columns": df.columns.tolist()
        }
        store_session_data(session_id, session_data)
        
        # Prepare response (exclude large data to avoid payload size issues)
        response_data = {
            "session_id": session_id,
            "shape": analysis_results.get("shape", {}),
            "columns": analysis_results.get("columns", [])[:20],  # Limit columns
            "kpis": {k: v for k, v in list(analysis_results.get("kpis", {}).items())[:30]},
            "totals": analysis_results.get("totals", {}),
            "averages": analysis_results.get("averages", {}),
            "correlations": analysis_results.get("correlations", {}).get("strongest", {}),
            "trends": analysis_results.get("trends", {}),
            "schema": analysis_results.get("schema", {}),
            "ml_available": bool(ml_results.get("target_detected")),
            "ml_target": ml_results.get("target_detected"),
            "ml_regression_score": ml_results.get("regression", {}).get("best_r2"),
            "ml_classification_acc": ml_results.get("classification", {}).get("accuracy")
        }
        
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# ANALYZE ENDPOINT
# ============================================

@router.get("/analyze/{session_id}")
async def get_analysis(session_id: str):
    """Get full analysis results for a session."""
    session = get_session_data(session_id)
    
    return {
        "shape": session["analysis"].get("shape", {}),
        "columns": session["analysis"].get("columns", []),
        "kpis": session["analysis"].get("kpis", {}),
        "totals": session["analysis"].get("totals", {}),
        "averages": session["analysis"].get("averages", {}),
        "correlations": session["analysis"].get("correlations", {}),
        "trends": session["analysis"].get("trends", {}),
        "category_aggregates": session["analysis"].get("category_aggregates", {}),
        "schema": session["analysis"].get("schema", {})
    }

# ============================================
# PLOTS ENDPOINT
# ============================================

@router.get("/plots/{session_id}")
async def get_plots(session_id: str):
    """Get all generated plots as JSON."""
    session = get_session_data(session_id)
    plots = session["analysis"].get("plots", {})
    
    return {"plots": plots}

# ============================================
# ML PREDICTION ENDPOINTS
# ============================================

@router.get("/ml/info/{session_id}")
async def get_ml_info(session_id: str):
    """Get ML model information for the session."""
    session = get_session_data(session_id)
    ml_results = session.get("ml_results", {})
    
    return {
        "target_detected": ml_results.get("target_detected"),
        "features_used": ml_results.get("features_used", []),
        "regression": ml_results.get("regression", {}),
        "classification": ml_results.get("classification", {}),
        "saved_models": get_available_models_info()
    }

@router.post("/predict/{session_id}")
async def predict_value(
    session_id: str,
    request: Request
):
    """Make prediction using trained model."""
    session = get_session_data(session_id)
    
    # Parse input values from request
    try:
        body = await request.json()
        input_values = body.get("inputs", {})
    except:
        input_values = {}
    
    # Get dataframe
    df = pd.DataFrame(session.get("dataframe", []))
    
    # Make prediction
    prediction = make_prediction(df, session, input_values)
    
    return prediction

# ============================================
# AI RECOMMENDATIONS ENDPOINT
# ============================================

@router.post("/recommend/{session_id}")
async def get_recommendations(session_id: str):
    """Generate AI recommendations (Gemini) - called on button click."""
    session = get_session_data(session_id)
    
    analysis = session.get("analysis", {})
    schema = analysis.get("schema", {})
    kpis = analysis.get("kpis", {})
    trends = analysis.get("trends", {})
    
    # Get AI recommendations
    recommendations = get_ai_recommendations_dynamic(
        pd.DataFrame(session.get("dataframe", [])),
        schema,
        kpis,
        trends
    )
    
    return {"recommendations": recommendations}

# ============================================
# AI CHAT ENDPOINT
# ============================================

@router.post("/chat/{session_id}")
async def chat_with_data(session_id: str, message: str = Form(...)):
    """Chat with AI about the dataset."""
    session = get_session_data(session_id)
    
    # Get stored data
    df = pd.DataFrame(session.get("dataframe", []))
    analysis = session.get("analysis", {})
    schema = analysis.get("schema", {})
    kpis = analysis.get("kpis", {})
    chat_history = session.get("chat_history", [])
    
    # Get AI response
    response = get_chat_response(message, df, schema, kpis)
    
    # Store in chat history
    chat_history.append({"user": message, "assistant": response})
    session["chat_history"] = chat_history
    store_session_data(session_id, session)
    
    return {"reply": response}

# ============================================
# SESSION INFO ENDPOINT
# ============================================

@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information."""
    session = get_session_data(session_id)
    
    return {
        "session_id": session_id,
        "shape": session.get("df_shape"),
        "columns_count": len(session.get("df_columns", [])),
        "chat_history_length": len(session.get("chat_history", [])),
        "ml_trained": bool(session.get("ml_results", {}).get("target_detected"))
    }

# ============================================
# CLEANUP ENDPOINT (Optional)
# ============================================

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session data."""
    if session_id in SESSION_STORAGE:
        del SESSION_STORAGE[session_id]
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")