import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from google import genai
import io
import os
from dotenv import load_dotenv
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ---------------------------
# 1. Advanced Data Cleaning & Preprocessing
# ---------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: strip, lowercase, replace spaces with underscores."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive cleaning: fix column names, handle missing values,
    convert date columns, ensure numeric types.
    """
    df = normalize_columns(df)
    df = df.copy()
    
    # Auto‑detect date columns (try to convert any column with 'date' in name)
    date_cols = [c for c in df.columns if 'date' in c]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        if df[col].notna().sum() > 0:
            df = df.dropna(subset=[col])
            df = df.sort_values(col)
            # rename to standard 'date'
            if col != 'date':
                df.rename(columns={col: 'date'}, inplace=True)
    
    # Ensure numeric columns are float/int
    numeric_candidates = [
        'energy_type_1', 'energy_type_2', 'hr_cost', 'operation_cost',
        'admin_cost', 'other_cost', 'revenue', 'profit'
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Fill missing categoricals
    for col in ['factory', 'shift']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str)
    
    # Derive total cost and energy (if components exist)
    cost_components = [c for c in ['hr_cost', 'operation_cost', 'admin_cost', 'other_cost'] if c in df.columns]
    if len(cost_components) > 0:
        df['total_opex'] = df[cost_components].sum(axis=1)
    if 'energy_type_1' in df.columns and 'energy_type_2' in df.columns:
        df['total_energy'] = df['energy_type_1'] + df['energy_type_2']
    
    # Encode shift for heatmap (if exists)
    if 'shift' in df.columns:
        le = LabelEncoder()
        df['shift_code'] = le.fit_transform(df['shift'])
    
    return df

def load_data_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load CSV/Excel and apply cleaning."""
    try:
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
        df = clean_dataframe(df)
        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

# ---------------------------
# 2. Dynamic Filter Helpers
# ---------------------------
def get_filter_options(df: pd.DataFrame) -> dict:
    """Return unique values for dynamic filters (factory, shift, energy types)."""
    options = {}
    if 'factory' in df.columns:
        options['factories'] = sorted(df['factory'].dropna().unique().tolist())
    if 'shift' in df.columns:
        options['shifts'] = sorted(df['shift'].dropna().unique().tolist())
    if 'energy_type_1' in df.columns:
        options['energy1_range'] = [float(df['energy_type_1'].min()), float(df['energy_type_1'].max())]
    if 'energy_type_2' in df.columns:
        options['energy2_range'] = [float(df['energy_type_2'].min()), float(df['energy_type_2'].max())]
    return options

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply dynamic filters (date range, factory, shift, energy ranges)."""
    filtered = df.copy()
    if 'date_range' in filters:
        start, end = filters['date_range']
        if 'date' in filtered.columns:
            filtered = filtered[(filtered['date'] >= start) & (filtered['date'] <= end)]
    if 'factory' in filters and filters['factory']:
        filtered = filtered[filtered['factory'] == filters['factory']]
    if 'shift' in filters and filters['shift']:
        filtered = filtered[filtered['shift'] == filters['shift']]
    if 'energy1_min' in filters:
        filtered = filtered[filtered['energy_type_1'] >= filters['energy1_min']]
    if 'energy1_max' in filters:
        filtered = filtered[filtered['energy_type_1'] <= filters['energy1_max']]
    if 'energy2_min' in filters:
        filtered = filtered[filtered['energy_type_2'] >= filters['energy2_min']]
    if 'energy2_max' in filters:
        filtered = filtered[filtered['energy_type_2'] <= filters['energy2_max']]
    return filtered

# ---------------------------
# 3. Advanced KPI Calculation (with dynamic columns)
# ---------------------------
def compute_enhanced_kpis(df: pd.DataFrame) -> dict:
    """Return both basic and advanced KPIs (profit, efficiency, rankings)."""
    kpis = {}
    
    # Basic cost KPIs
    if 'total_opex' in df.columns:
        kpis['total_opex'] = float(df['total_opex'].sum())
    if 'total_energy' in df.columns:
        kpis['total_energy'] = float(df['total_energy'].sum())
    
    # Revenue and profit (if revenue exists)
    if 'revenue' in df.columns:
        kpis['total_revenue'] = float(df['revenue'].sum())
        if 'total_opex' in df.columns:
            kpis['profit'] = float(df['revenue'].sum() - df['total_opex'].sum())
            kpis['profit_margin'] = float((kpis['profit'] / kpis['total_revenue']) * 100) if kpis['total_revenue'] != 0 else 0
    
    # Energy efficiency ratio (output/energy – use revenue as proxy if exists, else total_opex)
    if 'total_energy' in df.columns and 'revenue' in df.columns:
        kpis['energy_efficiency'] = float(df['revenue'].sum() / df['total_energy'].sum()) if df['total_energy'].sum() != 0 else 0
    elif 'total_energy' in df.columns and 'total_opex' in df.columns:
        kpis['energy_efficiency'] = float(df['total_opex'].sum() / df['total_energy'].sum()) if df['total_energy'].sum() != 0 else 0
    
    # Factory and shift rankings (if available)
    if 'factory' in df.columns and 'total_opex' in df.columns:
        factory_cost = df.groupby('factory')['total_opex'].sum().sort_values(ascending=False)
        kpis['top_factory'] = factory_cost.index[0] if len(factory_cost) > 0 else 'N/A'
        kpis['bottom_factory'] = factory_cost.index[-1] if len(factory_cost) > 0 else 'N/A'
    if 'shift' in df.columns and 'total_opex' in df.columns:
        shift_cost = df.groupby('shift')['total_opex'].mean().sort_values(ascending=False)
        kpis['most_expensive_shift'] = shift_cost.index[0] if len(shift_cost) > 0 else 'N/A'
        kpis['cheapest_shift'] = shift_cost.index[-1] if len(shift_cost) > 0 else 'N/A'
    
    # HR and Admin percentages (if components exist)
    if 'hr_cost' in df.columns and 'total_opex' in df.columns:
        total_hr = df['hr_cost'].sum()
        total_opex = df['total_opex'].sum()
        kpis['hr_percent'] = float((total_hr / total_opex) * 100) if total_opex != 0 else 0
    if 'admin_cost' in df.columns and 'total_opex' in df.columns:
        total_admin = df['admin_cost'].sum()
        total_opex = df['total_opex'].sum()
        kpis['admin_percent'] = float((total_admin / total_opex) * 100) if total_opex != 0 else 0
    
    return kpis

def compute_kpis_for_frontend(df: pd.DataFrame) -> dict:
    """Return KPIs with capitalised names expected by the frontend."""
    kpis = compute_enhanced_kpis(df)
    mapping = {
        'total_opex': 'Total OpEx',
        'total_energy': 'Total Energy',
        'total_revenue': 'Total Revenue',
        'profit': 'Profit',
        'profit_margin': 'Profit Margin (%)',
        'energy_efficiency': 'Energy Efficiency',
        'hr_percent': 'HR Cost %',
        'admin_percent': 'Admin Overhead %',
        'top_factory': 'Top Factory',
        'bottom_factory': 'Bottom Factory',
        'most_expensive_shift': 'Most Expensive Shift',
        'cheapest_shift': 'Cheapest Shift'
    }
    frontend_kpis = {}
    for key, value in kpis.items():
        if key in mapping:
            frontend_kpis[mapping[key]] = value
        else:
            frontend_kpis[key] = value
    return frontend_kpis

# ---------------------------
# 4. Plot Generation for Filtered Data (case‑insensitive, works with capitalised names)
# ---------------------------
def get_line_chart(df: pd.DataFrame, metric: str = 'Total_OpEx') -> dict:
    """Line chart of metric over date (if date column exists)."""
    # Find date column (case‑insensitive)
    date_col = None
    for col in df.columns:
        if col.lower() == 'date':
            date_col = col
            break
    if not date_col or metric not in df.columns:
        return None
    daily = df.groupby(date_col)[metric].sum().reset_index()
    fig = px.line(daily, x=date_col, y=metric, title=f'{metric} Trend')
    return fig.to_json()

def get_stacked_bar(df: pd.DataFrame) -> dict:
    """Stacked bar: factory vs cost components (if components exist)."""
    # Find factory column
    factory_col = None
    for col in df.columns:
        if col.lower() == 'factory':
            factory_col = col
            break
    cost_cols = [c for c in ['HR_Cost', 'Operation_Cost', 'Admin_Cost', 'Other_Cost'] if c in df.columns]
    if not cost_cols or not factory_col:
        return None
    grouped = df.groupby(factory_col)[cost_cols].sum().reset_index()
    fig = px.bar(grouped, x=factory_col, y=cost_cols, title='Cost Composition by Factory', barmode='stack')
    return fig.to_json()

def get_heatmap(df: pd.DataFrame) -> dict:
    """Heatmap: factory vs shift total cost."""
    # Find factory, shift, and total_opex columns (case‑insensitive)
    factory_col = None
    shift_col = None
    cost_col = None
    for col in df.columns:
        if col.lower() == 'factory':
            factory_col = col
        elif col.lower() == 'shift':
            shift_col = col
        elif col.lower() == 'total_opex':
            cost_col = col
    if not factory_col or not shift_col or not cost_col:
        return None
    pivot = df.pivot_table(index=factory_col, columns=shift_col, values=cost_col, aggfunc='sum').fillna(0)
    fig = px.imshow(pivot, text_auto=True, aspect='auto', title='Heatmap: Factory vs Shift (Total Cost)')
    return fig.to_json()

def get_pie_energy(df: pd.DataFrame) -> dict:
    """Pie chart of energy type distribution (if both energy types exist)."""
    e1_col = None
    e2_col = None
    for col in df.columns:
        if col.lower() == 'energy_type_1':
            e1_col = col
        elif col.lower() == 'energy_type_2':
            e2_col = col
    if not e1_col or not e2_col:
        return None
    total1 = df[e1_col].sum()
    total2 = df[e2_col].sum()
    if total1 + total2 == 0:
        return None
    fig = px.pie(values=[total1, total2], names=['Energy Type 1', 'Energy Type 2'], title='Energy Mix')
    return fig.to_json()

# ---------------------------
# 5. ML Model Training (store model, return evaluation)
# ---------------------------
def train_forecasting_model_with_eval(df: pd.DataFrame, horizon: int = 30):
    """
    Train linear regression on daily total_opex, return model, evaluation metrics,
    and future predictions.
    """
    if 'date' not in df.columns or 'total_opex' not in df.columns:
        return None, None, None, None, None
    # Resample to daily
    df_ts = df.set_index('date').resample('D').sum().fillna(0).reset_index()
    df_ts['dayofweek'] = df_ts['date'].dt.dayofweek
    df_ts['month'] = df_ts['date'].dt.month
    df_ts['day'] = df_ts['date'].dt.day
    df_ts['days_since_start'] = (df_ts['date'] - df_ts['date'].min()).dt.days
    features = ['dayofweek', 'month', 'day', 'days_since_start']
    X = df_ts[features].values
    y = df_ts['total_opex'].values
    if len(X) < 10:
        return None, None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Future predictions
    last_date = df_ts['date'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    future_features = []
    for d in future_dates:
        future_features.append([d.dayofweek, d.month, d.day, (d - df_ts['date'].min()).days])
    future_pred = model.predict(future_features)
    return model, {'r2': r2, 'mae': mae, 'rmse': rmse}, future_dates, future_pred, df_ts

def train_cost_driver_model_enhanced(df: pd.DataFrame):
    """Return feature importance and evaluation metrics for Random Forest."""
    if 'total_opex' not in df.columns:
        return None, None, None
    # Prepare features (numeric + categorical)
    numeric_features = ['energy_type_1', 'energy_type_2', 'hr_cost', 'admin_cost', 'other_cost']
    categorical_features = []
    if 'factory' in df.columns:
        categorical_features.append('factory')
    if 'shift' in df.columns:
        categorical_features.append('shift')
    # One-hot encode categoricals
    X = df[numeric_features].copy()
    for cat in categorical_features:
        dummies = pd.get_dummies(df[cat], prefix=cat)
        X = pd.concat([X, dummies], axis=1)
    X = X.fillna(0)
    y = df['total_opex'].values
    if len(X) < 10:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    importance = dict(zip(X.columns, model.feature_importances_))
    return model, {'r2': r2, 'mae': mae}, importance

def detect_anomalies_enhanced(df: pd.DataFrame):
    """Return anomaly score and flagged rows."""
    if 'total_opex' not in df.columns:
        return None, None
    X = df[['total_opex']].values
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    preds = iso_forest.fit_predict(X)
    df['anomaly'] = preds == -1
    anomaly_score = df['anomaly'].mean() * 100
    anomalies = df[df['anomaly'] == True][['date', 'total_opex']].to_dict(orient='records') if 'date' in df.columns else []
    return anomaly_score, anomalies

# ---------------------------
# 6. AI Recommendation (Enhanced Prompt)
# ---------------------------
def get_ai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def get_ai_recommendations_enhanced(df: pd.DataFrame, kpis: dict, driver_importance: dict = None, anomalies=None) -> str:
    client = get_ai_client()
    if not client:
        return "API key missing. Set GEMINI_API_KEY in .env"
    
    # Build context
    context = f"""
    DATASET SUMMARY:
    - Total Operational Expenditure: {kpis.get('total_opex', 0):,.2f}
    - Total Energy Consumption: {kpis.get('total_energy', 0):,.2f}
    - HR Cost %: {kpis.get('hr_percent', 0):.1f}%
    - Admin Overhead %: {kpis.get('admin_percent', 0):.1f}%
    - Most expensive factory: {kpis.get('top_factory', 'N/A')}
    - Least expensive factory: {kpis.get('bottom_factory', 'N/A')}
    - Most expensive shift: {kpis.get('most_expensive_shift', 'N/A')}
    - Energy Efficiency (Revenue/Energy): {kpis.get('energy_efficiency', 0):.2f}
    """
    if driver_importance:
        top_drivers = sorted(driver_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        context += f"\n- Top cost drivers: {', '.join([f'{k}: {v:.2f}' for k,v in top_drivers])}"
    if anomalies:
        context += f"\n- Anomalies detected: {len(anomalies)} unusual cost spikes."
    
    prompt = f"""
    You are a senior factory operations consultant. Based on the following data, provide **three specific, actionable business recommendations**.
    
    {context}
    
    Format your answer as:
    1. [Short title]
       - Action: [concrete step]
       - Expected Impact: [quantified benefit]
    2. ...
    3. ...
    
    Focus on cost reduction, energy optimization, and workforce efficiency.
    """
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI error: {e}"

def get_chat_response_enhanced(question: str, df: pd.DataFrame, kpis: dict) -> str:
    client = get_ai_client()
    if not client:
        return "API key missing."
    sample = df.head(10).to_string()
    summary = df.describe().to_string()
    prompt = f"""
    You are a factory cost analyst. Answer based on the data.
    
    KPIs: {kpis}
    
    Data sample:
    {sample}
    
    Statistics:
    {summary}
    
    User question: {question}
    
    Provide a concise, data-driven answer.
    """
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# ---------------------------
# 7. Backward Compatibility (Keep old function names)
# ---------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for clean_dataframe (to avoid breaking existing code)."""
    return clean_dataframe(df)

def compute_kpis(df: pd.DataFrame) -> dict:
    """Alias for compute_enhanced_kpis."""
    return compute_enhanced_kpis(df)

def generate_all_plots(df: pd.DataFrame, freq: str = 'D') -> dict:
    """Keep original function signature (for backward compatibility)."""
    return get_line_chart(df, 'Total_OpEx') or {}

def get_ai_recommendations(df: pd.DataFrame) -> str:
    """Alias for enhanced version with dummy driver/anomaly (can be improved)."""
    kpis = compute_enhanced_kpis(df)
    return get_ai_recommendations_enhanced(df, kpis)

def get_chat_response(question: str, df: pd.DataFrame) -> str:
    """Alias for enhanced chat."""
    kpis = compute_enhanced_kpis(df)
    return get_chat_response_enhanced(question, df, kpis)

def get_future_forecast_kpis(df: pd.DataFrame, horizon: int = 30):
    """Keep original function (already uses train_forecasting_model)."""
    model, _, future_dates, future_pred, _ = train_forecasting_model_with_eval(df, horizon)
    if future_pred is None:
        return {}
    total_next_month = future_pred.sum()
    avg_daily = future_pred.mean()
    # Estimate energy demand
    if 'total_energy' in df.columns and 'total_opex' in df.columns:
        energy_ratio = df['total_energy'].sum() / df['total_opex'].sum()
        est_energy = total_next_month * energy_ratio
    else:
        est_energy = 0
    budget_overrun_prob = 30  # placeholder
    most_expensive_factory = df.groupby('factory')['total_opex'].sum().idxmax() if 'factory' in df.columns else 'N/A'
    most_expensive_shift = df.groupby('shift')['total_opex'].mean().idxmax() if 'shift' in df.columns else 'N/A'
    return {
        "next_month_total_cost": total_next_month,
        "estimated_energy_demand": est_energy,
        "budget_overrun_prob": budget_overrun_prob,
        "most_expensive_factory": most_expensive_factory,
        "most_expensive_shift": most_expensive_shift,
        "avg_daily_cost_next_month": avg_daily
    }

def train_forecasting_model(df: pd.DataFrame, horizon: int = 30):
    """Keep original signature for compatibility."""
    model, eval_metrics, future_dates, future_pred, df_ts = train_forecasting_model_with_eval(df, horizon)
    return model, None, future_dates, future_pred

def train_cost_driver_model(df: pd.DataFrame):
    """Keep original signature."""
    model, eval_metrics, importance = train_cost_driver_model_enhanced(df)
    return model, eval_metrics.get('r2') if eval_metrics else None, importance

def detect_anomalies(df: pd.DataFrame):
    """Keep original signature."""
    score, anomalies = detect_anomalies_enhanced(df)
    # Return a DataFrame with 'Anomaly' column (original expects that)
    df_out = df.copy()
    df_out['Anomaly'] = df_out['anomaly'] if 'anomaly' in df_out.columns else False
    return score, df_out