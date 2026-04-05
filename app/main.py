from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware


import uuid
import pandas as pd
import os

# Import your analysis functions
from .analysis import (
    load_data_from_bytes,
    preprocess_data,
    compute_kpis,
    generate_all_plots,
    train_revenue_model,
    train_cost_model,
    train_classification_model,
    get_ai_recommendations,
    get_chat_response
)

# ---------------------------
# INIT APP
# ---------------------------
app = FastAPI(title="Telco Tower Analytics")

# ---------------------------
# GLOBAL SESSION STORE (FIXED)
# ---------------------------
sessions = {}

# ---------------------------
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# PATH SETUP (ROBUST)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ---------------------------
# HELPER
# ---------------------------
def get_session(sid: str):
    if sid not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[sid]

# ---------------------------
# ROOT (HTML PAGE)
# ---------------------------
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------------------
# UPLOAD + ANALYSIS
# ---------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        df = load_data_from_bytes(contents, file.filename)
        df_clean = preprocess_data(df)

        kpis = compute_kpis(df_clean)
        plots = generate_all_plots(df_clean)

        # ML models
        rev_model, rev_score, _ = train_revenue_model(df_clean)
        cost_model, cost_score, _ = train_cost_model(df_clean)
        clf_model, clf_acc, _ = train_classification_model(df_clean)

        session_id = str(uuid.uuid4())

        sessions[session_id] = {
            "dataframe": df_clean.to_dict(orient="records"),
            "rev_model": rev_model,
            "cost_model": cost_model,
            "clf_model": clf_model,
            "rev_score": rev_score,
            "cost_score": cost_score,
            "clf_acc": clf_acc
        }

        return {
            "session_id": session_id,
            "kpis": kpis,
            "plots": plots,
            "rev_score": rev_score,
            "cost_score": cost_score,
            "clf_acc": clf_acc
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ---------------------------
# PREDICTION
# ---------------------------
@app.post("/predict/{session_id}")
async def predict_revenue(
    session_id: str,
    active_tenants: float = Form(...),
    energy_cost: float = Form(...),
    opex: float = Form(...)
):
    sess = get_session(session_id)

    model = sess.get("rev_model")
    if model is None:
        raise HTTPException(status_code=400, detail="Revenue model not trained")

    try:
        pred = model.predict([[active_tenants, energy_cost, opex]])[0]
        return {"predicted_revenue": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# RECOMMENDATIONS (AI)
# ---------------------------
@app.post("/recommend/{session_id}")
async def recommend(session_id: str):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])

    try:
        rec = get_ai_recommendations(df)
        return {"recommendations": rec}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# CHAT
# ---------------------------
@app.post("/chat/{session_id}")
async def chat(session_id: str, message: str = Form(...)):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])

    try:
        reply = get_chat_response(message, df)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# GET PLOTS AGAIN
# ---------------------------
@app.get("/plots/{session_id}")
async def get_plots(session_id: str):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])

    try:
        plots = generate_all_plots(df)
        return {"plots": plots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))