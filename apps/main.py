from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uuid
import pandas as pd
import os
from typing import Optional, Dict, Any

# Import enhanced analysis functions (using new names)
try:
    from .analysis import (
        compute_kpis_for_frontend,
        load_data_from_bytes,
        clean_dataframe,               # new: robust cleaning
        compute_enhanced_kpis,        # new: returns profit, efficiency, rankings
        get_filter_options,           # new: returns unique values for filters
        apply_filters,                # new: filter dataframe
        get_line_chart,               # new: line chart JSON
        get_stacked_bar,              # new: stacked bar JSON
        get_heatmap,                  # new: heatmap JSON
        get_pie_energy,               # new: pie chart JSON
        train_forecasting_model_with_eval,  # new: returns model + eval metrics
        train_cost_driver_model_enhanced,
        detect_anomalies_enhanced,
        get_ai_recommendations_enhanced,
        get_chat_response_enhanced,
        get_future_forecast_kpis
    )
    # Also import old aliases for backward compatibility (they still work)
    from .analysis import (
        preprocess_data,
        compute_kpis,
        generate_all_plots,
        train_forecasting_model,
        train_cost_driver_model,
        detect_anomalies,
        get_ai_recommendations,
        get_chat_response
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise

# ---------------------------
# INIT APP & PATH SETUP
# ---------------------------
app = FastAPI(title="Factory Cost Analytics", version="2.1.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ---------------------------
# GLOBAL SESSION STORE
# ---------------------------
sessions: Dict[str, Dict[str, Any]] = {}

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

def rename_columns_to_original(df: pd.DataFrame) -> pd.DataFrame:
    """Convert lowercase column names (used internally) to original capitalised names expected by frontend."""
    mapping = {
        'date': 'Date',
        'total_opex': 'Total_OpEx',
        'total_energy': 'Total_Energy',
        'hr_cost': 'HR_Cost',
        'operation_cost': 'Operation_Cost',
        'admin_cost': 'Admin_Cost',
        'other_cost': 'Other_Cost',
        'energy_type_1': 'Energy_Type_1',
        'energy_type_2': 'Energy_Type_2',
        'factory': 'Factory',
        'shift': 'Shift',
        'shift_code': 'Shift_Code',
        'anomaly': 'Anomaly'
    }
    return df.rename(columns=mapping)

# ---------------------------
# ROOT (HTML PAGE)
# ---------------------------
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------------------
# UPLOAD + BASIC ANALYSIS (Enhanced)
# ---------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Load and clean using new robust function
        df = load_data_from_bytes(contents, file.filename)  # returns cleaned, lowercase columns
        # Keep a copy with original naming for frontend compatibility
        df_original = rename_columns_to_original(df.copy())

        # Compute enhanced KPIs (from lowercase df)
        kpis = compute_enhanced_kpis(df)
        # Also compute old-style KPIs for backward compatibility (will be same as enhanced but with old keys)
        kpis_old = compute_kpis(df_original)  # uses old function expecting original columns

        # Generate plots for Track tab (using old function to keep same JSON keys)
        plots = generate_all_plots(df_original, freq='D')

        # Train and store forecasting model (use lowercase df)
        forecast_model, forecast_metrics, future_dates, future_pred, _ = train_forecasting_model_with_eval(df, horizon=30)
        # Train cost driver model
        driver_model, driver_metrics, driver_importance = train_cost_driver_model_enhanced(df)
        # Detect anomalies
        anomaly_score, anomalies_list = detect_anomalies_enhanced(df)

        # Compute future KPIs
        future_kpis = get_future_forecast_kpis(df, horizon=30)

        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "dataframe": df_original.to_dict(orient="records"),  # store original-named DataFrame
            "forecast_model": forecast_model,   # store model object
            "forecast_metrics": forecast_metrics,
            "driver_model": driver_model,
            "driver_importance": driver_importance,
            "driver_metrics": driver_metrics,
            "anomaly_score": anomaly_score,
            "anomalies": anomalies_list,
            "future_kpis": future_kpis
        }

        return {
            "session_id": session_id,
            "kpis": kpis_old,               # old-style KPIs for frontend compatibility
            "plots": plots,
            "future_kpis": future_kpis,
            "driver_importance": driver_importance,
            "anomaly_score": anomaly_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/filter_options/{session_id}")
async def filter_options(session_id: str):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])
    # Convert to lowercase for consistent column naming
    df_lower = df.rename(columns={col: col.lower() for col in df.columns})
    options = {}
    if 'factory' in df_lower.columns:
        options['factories'] = sorted(df_lower['factory'].dropna().unique().tolist())
    if 'shift' in df_lower.columns:
        options['shifts'] = sorted(df_lower['shift'].dropna().unique().tolist())
    if 'energy_type_1' in df_lower.columns:
        options['energy1_range'] = [float(df_lower['energy_type_1'].min()), float(df_lower['energy_type_1'].max())]
    if 'energy_type_2' in df_lower.columns:
        options['energy2_range'] = [float(df_lower['energy_type_2'].min()), float(df_lower['energy_type_2'].max())]
    return options

# ---------------------------
# DYNAMIC FILTER ENDPOINT (NEW)
# ---------------------------
@app.post("/filter")
async def filter_data(
    session_id: str = Form(...),
    date_range_start: Optional[str] = Form(None),
    date_range_end: Optional[str] = Form(None),
    factory: Optional[str] = Form(None),
    shift: Optional[str] = Form(None),
    energy1_min: Optional[float] = Form(None),
    energy1_max: Optional[float] = Form(None),
    energy2_min: Optional[float] = Form(None),
    energy2_max: Optional[float] = Form(None),
):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])
    # Convert original columns back to lowercase for filtering (since filters use lowercase names)
    df_lower = df.rename(columns={col: col.lower() for col in df.columns})
    
    filters = {}
    if date_range_start and date_range_end:
        filters['date_range'] = (pd.to_datetime(date_range_start), pd.to_datetime(date_range_end))
    if factory:
        filters['factory'] = factory
    if shift:
        filters['shift'] = shift
    if energy1_min is not None:
        filters['energy1_min'] = energy1_min
    if energy1_max is not None:
        filters['energy1_max'] = energy1_max
    if energy2_min is not None:
        filters['energy2_min'] = energy2_min
    if energy2_max is not None:
        filters['energy2_max'] = energy2_max
    
    filtered_df = apply_filters(df_lower, filters)
    # Convert back to original naming for consistent output
    filtered_df = rename_columns_to_original(filtered_df)
    
    # Compute KPIs on filtered data (use old compute_kpis to match frontend expectations)
    kpis = compute_kpis_for_frontend(filtered_df)
    
    # Generate new charts (using original-named columns)
    line_chart = get_line_chart(filtered_df, metric='Total_OpEx')
    stacked_bar = get_stacked_bar(filtered_df)
    heatmap = get_heatmap(filtered_df)
    pie_energy = get_pie_energy(filtered_df)
    
    return {
        "kpis": kpis,
        "line_chart": line_chart,
        "stacked_bar": stacked_bar,
        "heatmap": heatmap,
        "pie_energy": pie_energy
    }

# ---------------------------
# PLOTS (with frequency) - unchanged but uses stored df
# ---------------------------
@app.get("/plots/{session_id}")
async def get_plots(session_id: str, freq: str = Query('D', regex='^(D|W|M)$')):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])
    plots = generate_all_plots(df, freq=freq)
    return {"plots": plots}

# ---------------------------
# FORECAST (next N days) - uses stored model if available
# ---------------------------
@app.post("/forecast/{session_id}")
async def forecast(session_id: str, horizon: int = Form(30)):
    sess = get_session(session_id)
    model = sess.get("forecast_model")
    if model is None:
        # fallback: retrain
        df = pd.DataFrame(sess["dataframe"])
        df_lower = df.rename(columns={col: col.lower() for col in df.columns})
        model, metrics, future_dates, future_pred, _ = train_forecasting_model_with_eval(df_lower, horizon=horizon)
        if future_pred is None:
            raise HTTPException(400, "Cannot generate forecast – insufficient data")
        sess["forecast_model"] = model  # store for future
    else:
        # Use stored model to generate predictions
        # Need to recompute future dates and features based on last date in stored data
        df = pd.DataFrame(sess["dataframe"])
        df_lower = df.rename(columns={col: col.lower() for col in df.columns})
        # Get last date from data
        last_date = df_lower['date'].max()
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon)]
        # Build features (must match training features)
        # We need the same feature engineering as in train_forecasting_model_with_eval
        # Simpler: just use the stored model's prediction method with new features
        # Recreate features (assuming model was trained on dayofweek, month, day, days_since_start)
        df_ts = df_lower.set_index('date').resample('D').sum().fillna(0).reset_index()
        base_date = df_ts['date'].min()
        future_features = []
        for d in future_dates:
            future_features.append([d.dayofweek, d.month, d.day, (d - base_date).days])
        future_pred = model.predict(future_features)
    
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
    # Use stored anomalies list
    anomalies_list = sess.get("anomalies", [])
    score = sess.get("anomaly_score", 0)
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
# AI RECOMMENDATIONS & CHAT (enhanced prompts)
# ---------------------------
@app.post("/recommend/{session_id}")
async def recommend(session_id: str):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])
    # Convert to lowercase for enhanced functions
    df_lower = df.rename(columns={col: col.lower() for col in df.columns})
    kpis = compute_enhanced_kpis(df_lower)
    driver_importance = sess.get("driver_importance", {})
    anomalies = sess.get("anomalies", [])
    try:
        rec = get_ai_recommendations_enhanced(df_lower, kpis, driver_importance, anomalies)
        return {"recommendations": rec}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/{session_id}")
async def chat(session_id: str, message: str = Form(...)):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])
    df_lower = df.rename(columns={col: col.lower() for col in df.columns})
    kpis = compute_enhanced_kpis(df_lower)
    try:
        reply = get_chat_response_enhanced(message, df_lower, kpis)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# WHAT-IF SCENARIO (unchanged, but use lowercase columns internally)
# ---------------------------
@app.post("/whatif/{session_id}")
async def what_if(session_id: str, energy_price_increase_pct: float = Form(10)):
    sess = get_session(session_id)
    df = pd.DataFrame(sess["dataframe"])
    df_lower = df.rename(columns={col: col.lower() for col in df.columns})
    df_whatif = df_lower.copy()
    for col in ['energy_type_1', 'energy_type_2']:
        if col in df_whatif.columns:
            df_whatif[col] = df_whatif[col] * (1 + energy_price_increase_pct / 100)
    # Recalculate total_opex
    cost_cols = ['hr_cost', 'operation_cost', 'admin_cost', 'other_cost']
    df_whatif['total_opex'] = df_whatif[cost_cols].sum(axis=1) + df_whatif[['energy_type_1', 'energy_type_2']].sum(axis=1)
    total_new_opex = df_whatif['total_opex'].sum()
    original_opex = df_lower['total_opex'].sum()
    increase = total_new_opex - original_opex
    return {
        "original_total_opex": original_opex,
        "new_total_opex": total_new_opex,
        "absolute_increase": increase,
        "percent_increase": (increase / original_opex * 100) if original_opex != 0 else 0
    }