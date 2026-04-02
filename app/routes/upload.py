from fastapi import APIRouter, UploadFile, File, HTTPException, Form
<<<<<<< HEAD
from app.utils.file_handler import save_uploaded_file
from app.services.analysis import compute_kpis, generate_plots, get_ai_recommendations, load_dataframe
=======
from app.utils.file_handler import save_uploaded_file, load_dataframe
from app.services.analysis import compute_kpis, generate_plots, get_ai_recommendations, preprocess_data
>>>>>>> 45a793c2cbf6890d37bf20852a02144e23db00f0
from app.services.ml_model import train_revenue_model, train_cost_model, train_classification_model
import os
import uuid
from app.services.analysis import compute_kpis, generate_plots, get_ai_recommendations, load_dataframe
import pandas as pd
<<<<<<< HEAD
import numpy as np
=======
from typing import Optional
import json
>>>>>>> 45a793c2cbf6890d37bf20852a02144e23db00f0

router = APIRouter()

# In-memory storage for demo (use database in production)
STORAGE = {}

<<<<<<< HEAD
def convert_to_serializable(obj):
    """Convert NumPy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj
=======
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
"""
>>>>>>> 45a793c2cbf6890d37bf20852a02144e23db00f0

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
        
        # Store dataframe and file path
        STORAGE[session_id] = {
            "file_path": file_path,
            "dataframe": df.to_dict(orient="records"),
            "chat_history": []
        }
        
        # Compute KPIs and plots
        kpis = compute_kpis(df)
        plots = generate_plots(df)
        
        # Get AI recommendations
        ai_recommendations = get_ai_recommendations(df)
        
        # Store in session
        STORAGE[session_id]["kpis"] = kpis
        STORAGE[session_id]["plots"] = plots
        STORAGE[session_id]["ai_recommendations"] = ai_recommendations
        
        # Train ML models if enough data
        if len(df) >= 2:
            rev_model, rev_score, _ = train_revenue_model(df)
            cost_model, cost_score, _ = train_cost_model(df)
            clf_model, clf_acc, _ = train_classification_model(df)
            
            STORAGE[session_id]["rev_score"] = rev_score
            STORAGE[session_id]["cost_score"] = cost_score
            STORAGE[session_id]["clf_acc"] = clf_acc
            STORAGE[session_id]["rev_model"] = rev_model
            STORAGE[session_id]["cost_model"] = cost_model
            STORAGE[session_id]["clf_model"] = clf_model
        
<<<<<<< HEAD
        # Convert all values to serializable types before returning
        response_data = {
            "session_id": session_id, 
            "kpis": convert_to_serializable(kpis), 
            "plots": plots,
            "ai_recommendations": ai_recommendations,
            "rev_score": convert_to_serializable(STORAGE[session_id].get("rev_score")),
            "cost_score": convert_to_serializable(STORAGE[session_id].get("cost_score")),
            "clf_acc": convert_to_serializable(STORAGE[session_id].get("clf_acc"))
        }
        
        return response_data
=======
        return {
            "session_id": session_id, 
            "kpis": kpis, 
            "plots": plots,
            "ai_recommendations": ai_recommendations,
            "rev_score": STORAGE[session_id].get("rev_score"),
            "cost_score": STORAGE[session_id].get("cost_score"),
            "clf_acc": STORAGE[session_id].get("clf_acc")
        }
>>>>>>> 45a793c2cbf6890d37bf20852a02144e23db00f0
    
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
        "ai_recommendations": STORAGE[session_id].get("ai_recommendations"),
        "rev_score": STORAGE[session_id].get("rev_score"),
        "cost_score": STORAGE[session_id].get("cost_score"),
        "clf_acc": STORAGE[session_id].get("clf_acc")
    }

<<<<<<< HEAD
@router.post("/chat/{session_id}")
async def chat(session_id: str, message: str = Form(...)):
    """Chat with AI about the data."""
    if session_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get stored data
    df_data = STORAGE[session_id]["dataframe"]
    df = pd.DataFrame(df_data)
    chat_history = STORAGE[session_id].get("chat_history", [])
    
    # Prepare detailed data summary
    avg_productivity = float(df['Productivity'].mean())
    worst_tower = df.groupby('Tower_ID')['Productivity'].mean().idxmin()
    highest_diesel = df.groupby('Tower_ID')['Diesel_Dependency'].mean().idxmax()
    underutilized = df.groupby('Tower_ID')['Utilization'].mean()
    underutilized_towers = underutilized[underutilized < 0.5].index.tolist()
    
    # Get tower rankings
    top_revenue = df.groupby('Tower_ID')['Revenue'].sum().nlargest(3).index.tolist()
    top_cost = df.groupby('Tower_ID')['Total_Opex'].sum().nlargest(3).index.tolist()
    
    data_summary = f"""
TELECOM TOWER DATA SUMMARY:

Overall Metrics:
- Total Towers: {df['Tower_ID'].nunique()}
- Total Records: {len(df)}
- Average Productivity: {avg_productivity:.2f}
- Average Utilization: {float(df['Utilization'].mean() * 100):.1f}%
- Total Revenue: ${float(df['Revenue'].sum()):,.2f}
- Total Profit: ${float(df['Profit'].sum()):,.2f}
- Average Diesel Dependency: {float(df['Diesel_Dependency'].mean() * 100):.1f}%

Top Performers:
- Highest Revenue Towers: {', '.join(top_revenue)}
- Highest Cost Towers: {', '.join(top_cost)}

Problem Areas:
- Lowest Productivity Tower: {worst_tower}
- Highest Diesel Dependency: {highest_diesel}
- Underutilized Towers: {', '.join(map(str, underutilized_towers)) if underutilized_towers else 'None'}

Tower Details:
{df.groupby('Tower_ID').agg({
    'Revenue': 'sum',
    'Total_Opex': 'sum',
    'Profit': 'sum',
    'Productivity': 'mean',
    'Utilization': 'mean',
    'Diesel_Dependency': 'mean'
}).round(2).to_string()}
"""
    
    system_prompt = f"""
You are a telecom operations expert. You have access to the following tower data:

{data_summary}

Answer user questions based ONLY on this data. Be specific - mention tower names and numbers. If asked about something not in the data, explain what additional information would be needed.
=======
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
        # If no stored model, try to train one
        df_data = STORAGE[session_id]["dataframe"]
        df = pd.DataFrame(df_data)
        model, score, _ = train_revenue_model(df)
        if model is None:
            raise HTTPException(status_code=400, detail="Model could not be trained.")
    
    pred = model.predict([[active_tenants, energy_cost, opex]])[0]
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
        df_data = STORAGE[session_id]["dataframe"]
        df = pd.DataFrame(df_data)
        model, score, _ = train_cost_model(df)
        if model is None:
            raise HTTPException(status_code=400, detail="Model could not be trained.")
    
    pred = model.predict([[diesel_liters, electricity_kwh, maint_cost, repair_cost, staff_visits]])[0]
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
        df_data = STORAGE[session_id]["dataframe"]
        df = pd.DataFrame(df_data)
        model, acc, _ = train_classification_model(df)
        if model is None:
            raise HTTPException(status_code=400, detail="Model could not be trained.")
    
    pred = model.predict([[active_tenants, energy_cost, opex]])[0]
    return {"productivity_label": pred}

@router.post("/chat/{session_id}")
async def chat(session_id: str, message: str = Form(...)):
    """Chat with AI about the data."""
    if session_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get stored data
    df_data = STORAGE[session_id]["dataframe"]
    df = pd.DataFrame(df_data)
    chat_history = STORAGE[session_id].get("chat_history", [])
    
    # Get data summary using helper
    data_summary = _get_data_summary(df)
    
    # Build conversation history
    conversation = "\n".join(chat_history[-10:])  # Last 10 messages
    
    system_prompt = f"""
You are a telecom operations expert. You have been given the following data summary from a telecom tower dataset:
{data_summary}

Your task is to answer follow‑up questions from the user based on this data.

Previous conversation:
{conversation}

Now answer the user's latest question concisely and helpfully.
>>>>>>> 45a793c2cbf6890d37bf20852a02144e23db00f0
"""
    
    # Call Gemini API
    from app.services.analysis import get_gemini_response
    reply = get_gemini_response(system_prompt, message)
    
    # Store in chat history
    chat_history.append(f"User: {message}")
    chat_history.append(f"Assistant: {reply}")
    STORAGE[session_id]["chat_history"] = chat_history
    
<<<<<<< HEAD
=======
    return {"reply": reply}

@router.post("/chat_with_image/{session_id}")
async def chat_with_image(
    session_id: str,
    message: str = Form(""),
    image: UploadFile = File(None)
):
    """Chat with AI with optional image upload."""
    if session_id not in STORAGE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get stored data
    df_data = STORAGE[session_id]["dataframe"]
    df = pd.DataFrame(df_data)
    chat_history = STORAGE[session_id].get("chat_history", [])
    
    # Get data summary using helper
    data_summary = _get_data_summary(df)
    
    # Build conversation history
    conversation = "\n".join(chat_history[-10:])
    
    system_prompt = f"""
You are a telecom operations expert. You have been given the following data summary from a telecom tower dataset:
{data_summary}

Previous conversation:
{conversation}

Now answer the user's question.
"""
    
    # Handle image if uploaded
    image_data = None
    if image:
        from PIL import Image
        import io
        contents = await image.read()
        image_data = Image.open(io.BytesIO(contents))
    
    # Call Gemini API with or without image
    from app.services.analysis import get_gemini_response_with_image
    reply = get_gemini_response_with_image(system_prompt, message, image_data)
    
    # Store in chat history
    chat_history.append(f"User: {message}" + (" [with image]" if image else ""))
    chat_history.append(f"Assistant: {reply}")
    STORAGE[session_id]["chat_history"] = chat_history
    
>>>>>>> 45a793c2cbf6890d37bf20852a02144e23db00f0
    return {"reply": reply}