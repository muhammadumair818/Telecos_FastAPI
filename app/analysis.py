import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from google import genai
import io
import os
from dotenv import load_dotenv
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------
def load_data_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load CSV or Excel from bytes."""
    try:
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare factory cost data."""
    df = df.copy()
    
    # 1. Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
    
    # 2. Ensure numeric columns
    numeric_cols = ['Energy_Type_1', 'Energy_Type_2', 'HR_Cost', 'Operation_Cost', 
                    'Admin_Cost', 'Other_Cost']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 3. Fill missing categorical with 'Unknown'
    for col in ['Factory', 'Shift']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
    
    # 4. Derive total columns
    df['Total_OpEx'] = df[['HR_Cost', 'Operation_Cost', 'Admin_Cost', 'Other_Cost']].sum(axis=1)
    df['Total_Energy'] = df[['Energy_Type_1', 'Energy_Type_2']].sum(axis=1)
    
    # 5. Additional KPIs for later
    if 'Shift' in df.columns:
        # Encode shift for heatmap
        le = LabelEncoder()
        df['Shift_Code'] = le.fit_transform(df['Shift'])
    
    return df

# ---------------------------
# 2. KPI Calculations
# ---------------------------
def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute all required KPIs."""
    kpis = {}
    
    # Total Operational Expenditure (OpEx)
    if 'Total_OpEx' in df.columns:
        kpis['Total OpEx'] = float(df['Total_OpEx'].sum())
    
    # Average Cost per Shift
    if 'Total_OpEx' in df.columns and 'Shift' in df.columns:
        shift_avg = df.groupby('Shift')['Total_OpEx'].mean().mean()
        kpis['Avg Cost per Shift'] = float(shift_avg)
    
    # Total Energy Consumption (Type 1 + Type 2)
    if 'Total_Energy' in df.columns:
        kpis['Total Energy'] = float(df['Total_Energy'].sum())
    
    # HR Cost Percentage (HR_Cost / Total_OpEx)
    if 'HR_Cost' in df.columns and 'Total_OpEx' in df.columns:
        total_hr = df['HR_Cost'].sum()
        total_opex = df['Total_OpEx'].sum()
        kpis['HR Cost %'] = float((total_hr / total_opex) * 100) if total_opex != 0 else 0
    
    # Admin Overhead Rate (Admin_Cost / Total_OpEx)
    if 'Admin_Cost' in df.columns and 'Total_OpEx' in df.columns:
        total_admin = df['Admin_Cost'].sum()
        total_opex = df['Total_OpEx'].sum()
        kpis['Admin Overhead %'] = float((total_admin / total_opex) * 100) if total_opex != 0 else 0
    
    return kpis

# ---------------------------
# 3. Plotting Functions (all required charts)
# ---------------------------
def generate_all_plots(df: pd.DataFrame, freq: str = 'D') -> dict:
    """
    Generate all charts as Plotly JSON.
    freq: 'D' (daily), 'W' (weekly), 'M' (monthly) for time aggregation.
    """
    plots = {}
    
    # Helper to resample time series
    if 'Date' in df.columns:
        df_ts = df.set_index('Date').copy()
        if freq == 'W':
            df_ts = df_ts.resample('W').sum()
        elif freq == 'M':
            df_ts = df_ts.resample('M').sum()
        else:
            df_ts = df_ts.resample('D').sum()
        df_ts = df_ts.reset_index()
    else:
        df_ts = df.copy()
    
    # ----- Time Series Charts -----
    # Daily Cost Run-Rate (Line)
    if 'Date' in df.columns and 'Total_OpEx' in df.columns:
        fig = px.line(df_ts, x='Date', y='Total_OpEx', title='Daily Cost Run-Rate')
        plots['daily_cost'] = fig.to_json()
    
    # Weekly Spending Trend (Area)
    if 'Date' in df.columns and 'Total_OpEx' in df.columns and freq != 'D':
        fig = px.area(df_ts, x='Date', y='Total_OpEx', title='Weekly/Monthly Spending Trend')
        plots['weekly_spend'] = fig.to_json()
    
    # MoM Growth (Bar) - only if enough months
    if 'Date' in df.columns and 'Total_OpEx' in df.columns:
        df_month = df.set_index('Date').resample('M').sum().reset_index()
        if len(df_month) > 1:
            df_month['Growth'] = df_month['Total_OpEx'].pct_change() * 100
            fig = px.bar(df_month, x='Date', y='Growth', title='Month-over-Month Growth (%)')
            plots['mom_growth'] = fig.to_json()
    
    # Cumulative Expenditure (Stepped Line)
    if 'Total_OpEx' in df.columns:
        df_cum = df.sort_values('Date') if 'Date' in df.columns else df
        df_cum['Cumulative_OpEx'] = df_cum['Total_OpEx'].cumsum()
        fig = px.line(df_cum, x=df_cum.index if 'Date' not in df.columns else 'Date', 
                      y='Cumulative_OpEx', title='Cumulative Expenditure')
        plots['cumulative'] = fig.to_json()
    
    # ----- Comparative Graphs -----
    # Factory-wise Cost Comparison (Grouped Bar)
    if 'Factory' in df.columns and 'Total_OpEx' in df.columns:
        factory_cost = df.groupby('Factory')['Total_OpEx'].sum().reset_index()
        fig = px.bar(factory_cost, x='Factory', y='Total_OpEx', title='Factory-wise Total Cost')
        plots['factory_cost'] = fig.to_json()
    
    # Shift Performance Benchmarking (Horizontal Bar)
    if 'Shift' in df.columns and 'Total_OpEx' in df.columns:
        shift_cost = df.groupby('Shift')['Total_OpEx'].mean().reset_index()
        fig = px.bar(shift_cost, x='Total_OpEx', y='Shift', orientation='h', 
                     title='Shift-wise Average Cost')
        plots['shift_benchmark'] = fig.to_json()
    
    # Energy Mix Ratio (Donut)
    if 'Energy_Type_1' in df.columns and 'Energy_Type_2' in df.columns:
        total_e1 = df['Energy_Type_1'].sum()
        total_e2 = df['Energy_Type_2'].sum()
        if total_e1 + total_e2 > 0:
            fig = px.pie(values=[total_e1, total_e2], names=['Energy Type 1', 'Energy Type 2'],
                         title='Energy Mix Ratio', hole=0.4)
            plots['energy_mix'] = fig.to_json()
    
    # Expense Distribution (Pie)
    cost_cats = ['HR_Cost', 'Operation_Cost', 'Admin_Cost', 'Other_Cost']
    if all(c in df.columns for c in cost_cats):
        totals = [df[c].sum() for c in cost_cats]
        if sum(totals) > 0:
            fig = px.pie(values=totals, names=cost_cats, title='Expense Distribution')
            plots['expense_pie'] = fig.to_json()
    
    # ----- Correlations & Relationships -----
    # Energy vs Operation Cost (Scatter)
    if 'Total_Energy' in df.columns and 'Operation_Cost' in df.columns:
        fig = px.scatter(df, x='Total_Energy', y='Operation_Cost', 
                         title='Energy vs Operation Cost', trendline='ols')
        plots['energy_op_scatter'] = fig.to_json()
    
    # HR Cost vs Shift Timing (Heatmap)
    if 'Shift_Code' in df.columns and 'HR_Cost' in df.columns:
        # Aggregate average HR cost per shift
        heat_data = df.groupby('Shift_Code')['HR_Cost'].mean().reset_index()
        fig = px.imshow([heat_data['HR_Cost'].values], 
                        x=heat_data['Shift_Code'].values, 
                        y=['Avg HR Cost'], 
                        title='HR Cost by Shift', color_continuous_scale='Reds')
        plots['hr_heatmap'] = fig.to_json()
    
    # Factory Size vs Total Spend (Bubble)
    # Assume size derived from number of records per factory as proxy
    if 'Factory' in df.columns and 'Total_OpEx' in df.columns:
        factory_stats = df.groupby('Factory').agg({'Total_OpEx': 'sum', 'Date': 'count'}).reset_index()
        factory_stats.columns = ['Factory', 'Total_Spend', 'Record_Count']
        fig = px.scatter(factory_stats, x='Record_Count', y='Total_Spend', size='Total_Spend',
                         title='Factory Size (Records) vs Total Spend', hover_name='Factory')
        plots['size_vs_spend'] = fig.to_json()
    
    # Admin Cost vs Total Output (Regression Line)
    # Use Total_Energy as proxy for output
    if 'Admin_Cost' in df.columns and 'Total_Energy' in df.columns:
        fig = px.scatter(df, x='Total_Energy', y='Admin_Cost', 
                         title='Admin Cost vs Output (Energy)', trendline='ols')
        plots['admin_vs_output'] = fig.to_json()
    
    # ----- Monitoring & Anomalies -----
    # Other Cost Spikes (Scatter with outliers)
    if 'Other_Cost' in df.columns:
        mean_other = df['Other_Cost'].mean()
        std_other = df['Other_Cost'].std()
        df['Other_Outlier'] = np.abs(df['Other_Cost'] - mean_other) > 2 * std_other
        fig = px.scatter(df, x=df.index, y='Other_Cost', color='Other_Outlier',
                         title='Other Cost Spikes (Outliers Highlighted)')
        plots['other_spikes'] = fig.to_json()
    
    # Shift-wise Cost Variance (Box Plot)
    if 'Shift' in df.columns and 'Total_OpEx' in df.columns:
        fig = px.box(df, x='Shift', y='Total_OpEx', title='Shift-wise Cost Variance')
        plots['shift_box'] = fig.to_json()
    
    # Resource Efficiency Score (Gauge Chart)
    # Efficiency = (Output / Input) – using Total_Energy / Total_OpEx as proxy
    if 'Total_Energy' in df.columns and 'Total_OpEx' in df.columns:
        total_energy = df['Total_Energy'].sum()
        total_opex = df['Total_OpEx'].sum()
        efficiency = (total_energy / total_opex) if total_opex != 0 else 0
        # Normalize to 0-100 for gauge
        norm_eff = min(100, (efficiency / efficiency.max()) * 100 if hasattr(efficiency, 'max') else efficiency * 100)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = norm_eff,
            title = {'text': "Resource Efficiency Score"},
            domain = {'x': [0,1], 'y': [0,1]},
            gauge = {'axis': {'range': [0,100]}}
        ))
        plots['efficiency_gauge'] = fig.to_json()
    
    return plots

# ---------------------------
# 4. ML Models for Prediction Tab
# ---------------------------
def train_forecasting_model(df: pd.DataFrame, horizon: int = 30):
    """Simple linear regression forecast on Total_OpEx using date features."""
    if 'Date' not in df.columns or 'Total_OpEx' not in df.columns:
        return None, None, None, None
    
    df_ts = df.set_index('Date').resample('D').sum().fillna(0).reset_index()
    df_ts['dayofweek'] = df_ts['Date'].dt.dayofweek
    df_ts['month'] = df_ts['Date'].dt.month
    df_ts['day'] = df_ts['Date'].dt.day
    df_ts['days_since_start'] = (df_ts['Date'] - df_ts['Date'].min()).dt.days
    
    features = ['dayofweek', 'month', 'day', 'days_since_start']
    X = df_ts[features].values
    y = df_ts['Total_OpEx'].values
    
    if len(X) < 10:
        return None, None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict next 'horizon' days
    last_date = df_ts['Date'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    future_features = []
    for d in future_dates:
        future_features.append([d.dayofweek, d.month, d.day, (d - df_ts['Date'].min()).days])
    future_pred = model.predict(future_features)
    
    return model, (X_test, y_test), future_dates, future_pred

def train_cost_driver_model(df: pd.DataFrame):
    """Random Forest to identify top cost drivers."""
    if 'Total_OpEx' not in df.columns:
        return None, None, None
    
    feature_cols = ['Energy_Type_1', 'Energy_Type_2', 'HR_Cost', 'Admin_Cost', 'Other_Cost']
    # also include shift and factory if categorical
    if 'Shift' in df.columns:
        df = pd.get_dummies(df, columns=['Shift'], prefix='Shift')
        feature_cols += [c for c in df.columns if c.startswith('Shift_')]
    if 'Factory' in df.columns:
        df = pd.get_dummies(df, columns=['Factory'], prefix='Factory')
        feature_cols += [c for c in df.columns if c.startswith('Factory_')]
    
    existing = [c for c in feature_cols if c in df.columns]
    if len(existing) < 2:
        return None, None, None
    
    X = df[existing].values
    y = df['Total_OpEx'].values
    
    if len(X) < 10:
        return None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    
    # Feature importance
    importance = dict(zip(existing, model.feature_importances_))
    return model, r2, importance

def detect_anomalies(df: pd.DataFrame):
    """Isolation Forest for anomaly detection on Total_OpEx."""
    if 'Total_OpEx' not in df.columns:
        return None, None
    
    X = df[['Total_OpEx']].values
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    preds = iso_forest.fit_predict(X)
    df['Anomaly'] = preds == -1
    anomaly_score = df['Anomaly'].mean() * 100
    return anomaly_score, df

def get_future_forecast_kpis(df: pd.DataFrame, horizon: int = 30):
    """Return summary KPIs for prediction tab."""
    model, test_data, future_dates, future_pred = train_forecasting_model(df, horizon)
    if future_pred is None:
        return {}
    
    total_next_month = future_pred.sum()
    avg_daily = future_pred.mean()
    
    # Estimate energy demand (using historical ratio of energy to cost)
    if 'Total_Energy' in df.columns and 'Total_OpEx' in df.columns:
        energy_ratio = df['Total_Energy'].sum() / df['Total_OpEx'].sum()
        est_energy = total_next_month * energy_ratio
    else:
        est_energy = 0
    
    # Budget overrun probability (simple heuristic)
    budget_overrun_prob = 30  # placeholder
    # Most expensive factory/shift from historical
    if 'Factory' in df.columns:
        most_expensive_factory = df.groupby('Factory')['Total_OpEx'].sum().idxmax()
    else:
        most_expensive_factory = "N/A"
    if 'Shift' in df.columns:
        most_expensive_shift = df.groupby('Shift')['Total_OpEx'].mean().idxmax()
    else:
        most_expensive_shift = "N/A"
    
    return {
        "next_month_total_cost": total_next_month,
        "estimated_energy_demand": est_energy,
        "budget_overrun_prob": budget_overrun_prob,
        "most_expensive_factory": most_expensive_factory,
        "most_expensive_shift": most_expensive_shift,
        "avg_daily_cost_next_month": avg_daily
    }

# ---------------------------
# 5. AI (Gemini) Functions
# ---------------------------
def get_ai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def get_ai_recommendations(df: pd.DataFrame) -> str:
    client = get_ai_client()
    if not client:
        return "API key missing. Set GEMINI_API_KEY in .env"
    
    # Prepare summary
    total_opex = df['Total_OpEx'].sum() if 'Total_OpEx' in df.columns else 0
    total_energy = df['Total_Energy'].sum() if 'Total_Energy' in df.columns else 0
    hr_pct = (df['HR_Cost'].sum() / total_opex * 100) if total_opex != 0 else 0
    admin_pct = (df['Admin_Cost'].sum() / total_opex * 100) if total_opex != 0 else 0
    top_factory = df.groupby('Factory')['Total_OpEx'].sum().idxmax() if 'Factory' in df.columns else "Unknown"
    
    prompt = f"""
    You are a factory operations cost consultant. Provide actionable recommendations.
    
    DATA SUMMARY:
    - Total OpEx: {total_opex:,.2f}
    - Total Energy Consumption: {total_energy:,.2f}
    - HR Cost %: {hr_pct:.1f}%
    - Admin Overhead %: {admin_pct:.1f}%
    - Highest cost factory: {top_factory}
    
    Based on this, provide:
    1. Cost reduction strategies (focus on high OpEx areas)
    2. Energy efficiency improvements
    3. Workforce optimisation (HR cost)
    4. Risk alerts (budget overruns, anomalies)
    
    Use markdown with bold headers.
    """
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI error: {e}"

def get_chat_response(question: str, df: pd.DataFrame) -> str:
    client = get_ai_client()
    if not client:
        return "API key missing."
    
    sample = df.head(10).to_string()
    summary = df.describe().to_string()
    prompt = f"""
    You are a factory cost analyst. Answer based on this data.
    
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