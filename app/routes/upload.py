from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.utils.file_handler import save_uploaded_file
from app.services.analysis import compute_kpis, generate_plots, get_ai_recommendations, load_dataframe
from app.services.ml_model import train_revenue_model, train_cost_model, train_classification_model
import uuid
from app.services.analysis import compute_kpis, generate_plots, get_ai_recommendations, load_dataframe
import pandas as pd
import numpy as np

router = APIRouter()

# In-memory storage for demo (use database in production)
STORAGE = {}

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
"""
    
    # Call Gemini API
    from app.services.analysis import get_gemini_response
    reply = get_gemini_response(system_prompt, message)
    
    # Store in chat history
    chat_history.append(f"User: {message}")
    chat_history.append(f"Assistant: {reply}")
    STORAGE[session_id]["chat_history"] = chat_history
    
    return {"reply": reply}