from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uuid
import pandas as pd
import os
from typing import Optional

# Import analysis functions (ensure analysis.py exists in same directory)
try:
    from .analysis import (
        load_data_from_bytes,
        preprocess_data,
        compute_kpis,
        generate_all_plots,
        train_forecasting_model,
        train_cost_driver_model,
        detect_anomalies,
        get_future_forecast_kpis,
        get_ai_recommendations,
        get_chat_response
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise

# ---------------------------
# INIT APP & PATH SETUP
# ---------------------------
app = FastAPI(title="Factory Cost Analytics", version="2.0.0")

# Correct static folder mounting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of main.py (app/)
static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)  # create if missing

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ---------------------------
# GLOBAL SESSION STORE
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
# HELPERS
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
# UPLOAD + BASIC ANALYSIS
# ---------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = load_data_from_bytes(contents, file.filename)
        df_clean = preprocess_data(df)

        kpis = compute_kpis(df_clean)
        plots = generate_all_plots(df_clean, freq='D')

        # ML models – wrap in try/except to avoid breaking if data insufficient
        future_kpis = {}
        driver_importance = {}
        anomaly_score = None
        try:
            forecast_model, forecast_test, future_dates, future_pred = train_forecasting_model(df_clean, horizon=30)
            if future_pred is not None:
                future_kpis = get_future_forecast_kpis(df_clean, horizon=30)
        except Exception as e:
            print(f"Forecasting error: {e}")
        try:
            driver_model, driver_r2, driver_importance = train_cost_driver_model(df_clean)
        except Exception as e:
            print(f"Driver model error: {e}")
        try:
            anomaly_score, df_with_anomalies = detect_anomalies(df_clean)
        except Exception as e:
            print(f"Anomaly detection error: {e}")

        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "dataframe": df_clean.to_dict(orient="records"),
            "forecast_model": None,  # models not pickled; retrain on demand
            "driver_model": None,
            "anomaly_score": anomaly_score,
            "future_kpis": future_kpis,
            "driver_importance": driver_importance,
            "future_dates": [],
            "future_pred": []
        }

        return {
            "session_id": session_id,
            "kpis": kpis,
            "plots": plots,
            "future_kpis": future_kpis,
            "driver_importance": driver_importance,
            "anomaly_score": anomaly_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ---------------------------
# PLOTS (with frequency)
# ---------------------------
@app.get("/plots/{session_id}")
async def get_plots(session_id: str, freq: str = Query('D', regex='^(D|W|M)$')):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])
    plots = generate_all_plots(df, freq=freq)
    return {"plots": plots}

# ---------------------------
# FORECAST (next N days)
# ---------------------------
@app.post("/forecast/{session_id}")
async def forecast(session_id: str, horizon: int = Form(30)):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])
    model, test_data, future_dates, future_pred = train_forecasting_model(df, horizon=horizon)
    if future_pred is None:
        raise HTTPException(400, "Cannot generate forecast – insufficient data")
    return {
        "future_dates": [d.isoformat() for d in future_dates],
        "future_pred": future_pred.tolist(),
        "total_next_month": sum(future_pred)
    }

# ---------------------------
# COST DRIVER ANALYSIS
# ---------------------------
@app.get("/cost_drivers/{session_id}")
async def cost_drivers(session_id: str):
    sess = get_session(session_id)
    importance = sess.get("driver_importance", {})
    if not importance:
        raise HTTPException(400, "Cost driver model not trained")
    return {"importance": importance}

# ---------------------------
# ANOMALY DETECTION
# ---------------------------
@app.get("/anomalies/{session_id}")
async def anomalies(session_id: str):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])
    score, df_anom = detect_anomalies(df)
    anomalies_list = df_anom[df_anom['Anomaly'] == True][['Date', 'Total_OpEx']].to_dict(orient='records') if 'Date' in df_anom else []
    return {"anomaly_score": score, "anomalies": anomalies_list}

# ---------------------------
# FUTURE KPIs (summary cards)
# ---------------------------
@app.get("/future_kpis/{session_id}")
async def future_kpis(session_id: str):
    sess = get_session(session_id)
    kpis = sess.get("future_kpis", {})
    return kpis

# ---------------------------
# AI RECOMMENDATIONS & CHAT
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
# WHAT-IF SCENARIO
# ---------------------------
@app.post("/whatif/{session_id}")
async def what_if(session_id: str, energy_price_increase_pct: float = Form(10)):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])
    df_whatif = df.copy()
    for col in ['Energy_Type_1', 'Energy_Type_2']:
        if col in df_whatif.columns:
            df_whatif[col] = df_whatif[col] * (1 + energy_price_increase_pct / 100)
    df_whatif['Total_OpEx'] = df_whatif[['HR_Cost', 'Operation_Cost', 'Admin_Cost', 'Other_Cost']].sum(axis=1) + df_whatif[['Energy_Type_1', 'Energy_Type_2']].sum(axis=1)
    total_new_opex = df_whatif['Total_OpEx'].sum()
    original_opex = df['Total_OpEx'].sum()
    increase = total_new_opex - original_opex
    return {
        "original_total_opex": original_opex,
        "new_total_opex": total_new_opex,
        "absolute_increase": increase,
        "percent_increase": (increase / original_opex * 100) if original_opex != 0 else 0
    }