from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks # Added BackgroundTasks
from app.utils.file_handler import save_uploaded_file, load_dataframe
from app.services.analysis import compute_kpis, generate_plots, get_ai_recommendations, get_chat_response, get_gemini_response_unified # Updated import for unified Gemini function
from app.services.ml_model import train_revenue_model, train_cost_model, train_classification_model
import os
import uuid
import pandas as pd
from typing import Optional
import json
from PIL import Image # Moved from inside chat_with_image
import io # Moved from inside chat_with_image

router = APIRouter()

# In-memory storage for demo (use database in production)
# CRITICAL: This in-memory storage is NOT suitable for production.
# It will not scale, is not multi-worker safe, and will lead to memory exhaustion.
# Consider using Redis, a dedicated database, or persistent file storage for session data.
STORAGE = {}

def _get_data_summary(df: pd.DataFrame) -> str:
    """Helper to generate a consistent data summary for AI prompts."""
    avg_productivity = df['Productivity'].mean()
    worst_tower = df.groupby('Tower_ID')['Productivity'].mean().idxmin()
    highest_diesel = df.groupby('Tower_ID')['Diesel_Dependency'].mean().idxmax()
    underutilized = df.groupby('Tower_ID')['Utilization'].mean()
    underutilized_towers = underutilized[underutilized < 0.5].index.tolist()
    
    return f"""
Data Summary:
- Average productivity (Revenue/(Energy Cost+OPEX)): {avg_productivity:.2f}
- Tower with lowest average productivity: {worst_tower}
- Tower with highest diesel dependency: {highest_diesel}
- Underutilized towers (utilization < 0.5): {underutilized_towers if underutilized_towers else 'None'}
- write like a human 
- don't write like a computer 
- don't add * # 
"""

def _process_uploaded_data_in_background(session_id: str, file_path: str):
    """
    Background task to process data and train models without blocking the upload response.
    """
    try:
        from app.utils.file_handler import load_dataframe
        from app.services.analysis import compute_kpis, generate_plots, get_ai_recommendations
        from app.services.ml_model import train_revenue_model, train_cost_model, train_classification_model

        df = load_dataframe(file_path)
        if df is None:
            return

        # Generate and store data summary once to speed up Chat endpoints
        data_summary = _get_data_summary(df)

        # Compute and store analytical results
        STORAGE[session_id].update({
            "kpis": compute_kpis(df),
            "plots": generate_plots(df),
            "ai_recommendations": get_ai_recommendations(df),
            "data_summary": data_summary
        })

        # Train and store ML models
        if len(df) >= 2:
            rev_model, rev_score, _ = train_revenue_model(df)
            cost_model, cost_score, _ = train_cost_model(df)
            clf_model, clf_acc, _ = train_classification_model(df)
            
            STORAGE[session_id].update({
                "rev_model": rev_model,
                "rev_score": rev_score,
                "cost_model": cost_model,
                "cost_score": cost_score,
                "clf_model": clf_model,
                "clf_acc": clf_acc
            })
    except Exception as e:
        print(f"Error in background processing for session {session_id}: {e}")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
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
        
        # Store only file_path and chat_history initially
        STORAGE[session_id] = {
            "file_path": file_path,
            "chat_history": []
        }
        
        # Add background task to process data and train models
        background_tasks.add_task(_process_uploaded_data_in_background, session_id, file_path)
        
        # Return immediate response with session ID
        return {
            "session_id": session_id,
            "message": "File uploaded successfully. Data processing and model training are running in the background."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/{session_id}")
async def analyze(session_id: str):
    """Return analysis results for a given session."""
    if session_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if background task has completed and results are available
    if "kpis" not in STORAGE[session_id]:
        raise HTTPException(status_code=202, detail="Analysis still in progress. Please try again shortly.")

    return {
        "kpis": STORAGE[session_id].get("kpis"),
        "plots": STORAGE[session_id].get("plots"),
        "ai_recommendations": STORAGE[session_id].get("ai_recommendations"),
        "rev_score": STORAGE[session_id].get("rev_score"),
        "cost_score": STORAGE[session_id].get("cost_score"),
        "clf_acc": STORAGE[session_id].get("clf_acc")
    }

# Helper to load DataFrame for endpoints that need it
def _get_df_from_session(session_id: str) -> pd.DataFrame:
    if session_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    file_path = STORAGE[session_id].get("file_path")
    if not file_path:
        raise HTTPException(status_code=500, detail="File path not found for session.")
    df = load_dataframe(file_path)
    if df is None:
        raise HTTPException(status_code=500, detail="Failed to load dataframe from file.")
    return df

@router.post("/predict/{session_id}")
async def predict(
    session_id: str,
    active_tenants: int = Form(10), 
    energy_cost: float = Form(1000.0), 
    opex: float = Form(2000.0)
):
    """Predict revenue using the trained model."""
    if session_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get the stored model
    model = STORAGE[session_id].get("rev_model")
    
    if model is None:
        # If no stored model, try to train one (this is inefficient, better to wait for background task)
        # Or, ideally, models should be loaded from persistent storage if available.
        df = _get_df_from_session(session_id)
        model, score, _ = train_revenue_model(df)
        if model is None:
            raise HTTPException(status_code=400, detail="Model could not be trained or is not ready. Please wait for analysis to complete.")
        # Optionally store the newly trained model if it wasn't ready from background task
        STORAGE[session_id]["rev_model"] = model
    
    # It's safer to predict using a DataFrame with named columns
    input_data = pd.DataFrame([[active_tenants, energy_cost, opex]], 
                              columns=['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex']) # Assuming these are the feature names
    pred = model.predict(input_data)[0]
    return {"predicted_revenue": pred}

@router.post("/predict_opex/{session_id}")
async def predict_opex(
    session_id: str,
    diesel_liters: float = Form(100.0),
    electricity_kwh: float = Form(500.0),
    maint_cost: float = Form(500.0),
    repair_cost: float = Form(300.0),
    staff_visits: int = Form(5)
):
    """Predict OPEX using the trained model."""
    if session_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    model = STORAGE[session_id].get("cost_model")
    
    if model is None:
        df = _get_df_from_session(session_id)
        model, score, _ = train_cost_model(df)
        if model is None:
            raise HTTPException(status_code=400, detail="Model could not be trained or is not ready. Please wait for analysis to complete.")
        STORAGE[session_id]["cost_model"] = model
    
    input_data = pd.DataFrame([[diesel_liters, electricity_kwh, maint_cost, repair_cost, staff_visits]],
                              columns=['Diesel_Liters', 'Electricity_kWh', 'Maintenance_Cost', 'Repair_Cost', 'Staff_Visits'])
    pred = model.predict(input_data)[0]
    return {"predicted_opex": pred}

@router.post("/classify/{session_id}")
async def classify(
    session_id: str,
    active_tenants: int = Form(10),
    energy_cost: float = Form(1000.0),
    opex: float = Form(2000.0)
):
    """Classify productivity label."""
    if session_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    model = STORAGE[session_id].get("clf_model")
    
    if model is None:
        df = _get_df_from_session(session_id)
        model, acc, _ = train_classification_model(df)
        if model is None:
            raise HTTPException(status_code=400, detail="Model could not be trained or is not ready. Please wait for analysis to complete.")
        STORAGE[session_id]["clf_model"] = model
    
    input_data = pd.DataFrame([[active_tenants, energy_cost, opex]],
                              columns=['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex'])
    pred = model.predict(input_data)[0]
    return {"productivity_label": pred}

@router.post("/chat/{session_id}")
async def chat_with_data(session_id: str, message: str = Form(...)):
    """Chat with AI about the dataset."""
    try:
        # Get session data
        if session_id not in SESSION_STORAGE:
            return {"reply": "❌ Session expired. Please upload your file again."}
        
        session = SESSION_STORAGE[session_id]
        
        # Get stored data
        df = pd.DataFrame(session.get("dataframe", []))
        if df.empty:
            return {"reply": "❌ No data found. Please upload a file first."}
        
        analysis = session.get("analysis", {})
        schema = analysis.get("schema", {})
        kpis = analysis.get("kpis", {})
        chat_history = session.get("chat_history", [])
        
        # Prepare data summary for AI
        data_summary = f"""
Dataset Summary:
- Rows: {df.shape[0]}, Columns: {df.shape[1]}
- Column names: {', '.join(df.columns.tolist())}
- Numeric columns: {', '.join(schema.get('metrics', []))}
- Categorical columns: {', '.join(schema.get('categories', []))}
- Key metrics: {json.dumps({k: v for k, v in list(kpis.items())[:15]}, indent=2)}
- don't give me *, # etc in the response
- write like human 
- don't write like a computer 
"""
        
        # Get AI response
        response = get_chat_response(message, df, schema, kpis)
        
        # Store in chat history
        chat_history.append({"user": message, "assistant": response})
        session["chat_history"] = chat_history
        SESSION_STORAGE[session_id] = session
        
        return {"reply": response}
    
    except Exception as e:
        print(f"Chat error: {e}")
        return {"reply": f"❌ Error: {str(e)}"}

@router.post("/chat_with_image/{session_id}")
async def chat_with_image(
    session_id: str,
    message: str = Form(""),
    image: UploadFile = File(None)
):
    """Chat with AI with optional image upload."""
    if session_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Optimized: Use pre-calculated summary if available
    data_summary = STORAGE[session_id].get("data_summary")
    if not data_summary:
        df = _get_df_from_session(session_id)
        data_summary = _get_data_summary(df)

    chat_history = STORAGE[session_id].get("chat_history", [])
    
    # Build conversation history
    conversation = "\n".join(chat_history[-10:])
    
    system_prompt = f"""
You are a telecom operations expert. You have been given the following data summary from a telecom tower dataset:
{data_summary}

Previous conversation:
{conversation}

Now answer the user's question.
- write like a human 
- don't write like a computer 
- don't add * # 
"""
    
    # Handle image if uploaded
    image_data = None
    if image:
        contents = await image.read()
        image_data = Image.open(io.BytesIO(contents))
    
    # Call Gemini API with or without image
    reply = get_gemini_response_unified(system_prompt, message, image_data) # Using unified Gemini function
    
    # Store in chat history
    chat_history.append(f"User: {message}" + (" [with image]" if image else ""))
    chat_history.append(f"Assistant: {reply}")
    STORAGE[session_id]["chat_history"] = chat_history
    
    return {"reply": reply}