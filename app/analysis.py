import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import google.generativeai as genai
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Data loading & preprocessing
# ---------------------------
def load_data_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load CSV or Excel from bytes."""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Date, fill missing, derive columns (only if required columns exist)."""
    df = df.copy()
    # Convert Date if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Fill missing numeric with 0
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Derive metrics only if required columns exist
    if all(col in df.columns for col in ['Total_Energy_Cost', 'Total_Opex', 'Revenue']):
        denominator = df['Total_Energy_Cost'] + df['Total_Opex']
        df['Productivity'] = df['Revenue'] / denominator.replace(0, np.nan)
        df['Profit'] = df['Revenue'] - (df['Total_Energy_Cost'] + df['Total_Opex'])
    if 'Total_Energy_Cost' in df.columns and 'Revenue' in df.columns:
        df['Energy_Efficiency'] = df['Revenue'] / df['Total_Energy_Cost'].replace(0, np.nan)
    if 'Total_Opex' in df.columns and 'Revenue' in df.columns:
        df['Cost_Efficiency'] = df['Revenue'] / df['Total_Opex'].replace(0, np.nan)
    if 'Diesel_Cost' in df.columns and 'Total_Energy_Cost' in df.columns:
        df['Diesel_Dependency'] = df['Diesel_Cost'] / df['Total_Energy_Cost'].replace(0, np.nan)
    if 'Active_Tenants' in df.columns and 'Capacity' in df.columns:
        df['Utilization'] = df['Active_Tenants'] / df['Capacity'].replace(0, np.nan)

    df.fillna(0, inplace=True)
    return df

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute KPIs based on available columns."""
    kpis = {}
    if 'Revenue' in df.columns:
        kpis['Total Revenue'] = float(df['Revenue'].sum())
    if 'Total_Opex' in df.columns:
        kpis['Total OPEX'] = float(df['Total_Opex'].sum())
    if 'Total_Energy_Cost' in df.columns:
        kpis['Total Energy Cost'] = float(df['Total_Energy_Cost'].sum())
    if 'Profit' in df.columns:
        kpis['Total Profit'] = float(df['Profit'].sum())
    if 'Productivity' in df.columns:
        kpis['Avg Productivity'] = float(df['Productivity'].mean())
    if 'Energy_Efficiency' in df.columns:
        kpis['Avg Energy Efficiency'] = float(df['Energy_Efficiency'].mean())
    if 'Cost_Efficiency' in df.columns:
        kpis['Avg Cost Efficiency'] = float(df['Cost_Efficiency'].mean())
    if 'Diesel_Dependency' in df.columns:
        kpis['Avg Diesel Dependency'] = float(df['Diesel_Dependency'].mean())
    if 'Utilization' in df.columns:
        kpis['Avg Utilization'] = float(df['Utilization'].mean())
    return kpis

# ---------------------------
# Plotting functions (return Plotly JSON)
# ---------------------------
def plot_cost_vs_revenue(df):
    if 'Total_Energy_Cost' in df.columns and 'Revenue' in df.columns and 'Tower_ID' in df.columns:
        fig = px.scatter(df, x='Total_Energy_Cost', y='Revenue', color='Tower_ID',
                         title='Cost vs Revenue', hover_data=['Date'] if 'Date' in df.columns else None)
        return fig.to_json()
    return None

def plot_energy_vs_revenue(df):
    if 'Total_Energy_Cost' in df.columns and 'Revenue' in df.columns and 'Tower_ID' in df.columns:
        fig = px.scatter(df, x='Total_Energy_Cost', y='Revenue', color='Tower_ID',
                         title='Energy Cost vs Revenue', hover_data=['Date'] if 'Date' in df.columns else None)
        return fig.to_json()
    return None

def plot_utilization_vs_revenue(df):
    if 'Utilization' in df.columns and 'Revenue' in df.columns and 'Tower_ID' in df.columns:
        fig = px.scatter(df, x='Utilization', y='Revenue', color='Tower_ID',
                         title='Utilization vs Revenue', hover_data=['Date'] if 'Date' in df.columns else None)
        return fig.to_json()
    return None

def plot_diesel_vs_grid(df):
    if 'Diesel_Cost' in df.columns and 'Grid_Energy_Cost' in df.columns:
        diesel_total = df['Diesel_Cost'].sum()
        grid_total = df['Grid_Energy_Cost'].sum()
        fig = px.pie(values=[diesel_total, grid_total], names=['Diesel Cost', 'Grid Energy Cost'],
                     title='Diesel vs Grid Energy Cost')
        return fig.to_json()
    return None

def plot_opex_breakdown(df):
    opex_cols = [c for c in ['Maintenance_Cost', 'Repair_Cost', 'Staff_Visits'] if c in df.columns]
    if opex_cols:
        breakdown = df[opex_cols].sum().to_dict()
        fig = px.bar(x=list(breakdown.keys()), y=list(breakdown.values()),
                     title='OPEX Breakdown (sum over dataset)',
                     labels={'x': 'Category', 'y': 'Total Cost'})
        return fig.to_json()
    return None

def generate_all_plots(df):
    """Return dict of all available plots as JSON strings."""
    plots = {}
    p = plot_cost_vs_revenue(df)
    if p: plots['cost_vs_revenue'] = p
    p = plot_energy_vs_revenue(df)
    if p: plots['energy_vs_revenue'] = p
    p = plot_utilization_vs_revenue(df)
    if p: plots['utilization_vs_revenue'] = p
    p = plot_diesel_vs_grid(df)
    if p: plots['diesel_vs_grid'] = p
    p = plot_opex_breakdown(df)
    if p: plots['opex_breakdown'] = p
    return plots

# ---------------------------
# ML Models (exactly as in Streamlit)
# ---------------------------
def train_revenue_model(df):
    required = ['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex', 'Revenue']
    if not all(col in df.columns for col in required):
        return None, None, None
    features = ['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex']
    X = df[features].values
    y = df['Revenue'].values
    if len(X) < 2:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score, (X_test, y_test)

def train_cost_model(df):
    required = ['Diesel_Liters', 'Electricity_kWh', 'Maintenance_Cost', 'Repair_Cost', 'Staff_Visits', 'Total_Opex']
    if not all(col in df.columns for col in required):
        return None, None, None
    features = ['Diesel_Liters', 'Electricity_kWh', 'Maintenance_Cost', 'Repair_Cost', 'Staff_Visits']
    X = df[features].values
    y = df['Total_Opex'].values
    if len(X) < 2:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score, (X_test, y_test)

def train_classification_model(df):
    if 'Productivity' not in df.columns:
        return None, None, None
    df['Productivity_Label'] = pd.cut(df['Productivity'],
                                      bins=[-np.inf, 1, 1.5, np.inf],
                                      labels=['Low', 'Medium', 'High'])
    features = ['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex']
    if not all(col in df.columns for col in features):
        return None, None, None
    target = 'Productivity_Label'
    df_clean = df.dropna(subset=[target]).copy()
    if len(df_clean) < 2 or len(df_clean[target].unique()) < 2:
        return None, None, None
    X = df_clean[features].values
    y = df_clean[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, (X_test, y_test)

# ---------------------------
# AI / Gemini functions
# ---------------------------
def get_gemini_api_key():
    return os.getenv("GEMINI_API_KEY")

def get_ai_recommendations(df):
    api_key = get_gemini_api_key()
    if not api_key:
        return "API key not found. Please set GEMINI_API_KEY in .env"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')  # use stable model

    # Use only available columns
    avg_productivity = df['Productivity'].mean() if 'Productivity' in df.columns else 0
    worst_tower = None
    if 'Tower_ID' in df.columns and 'Productivity' in df.columns:
        worst_tower = df.groupby('Tower_ID')['Productivity'].mean().idxmin()
    highest_diesel = None
    if 'Tower_ID' in df.columns and 'Diesel_Dependency' in df.columns:
        highest_diesel = df.groupby('Tower_ID')['Diesel_Dependency'].mean().idxmax()
    underutilized_towers = []
    if 'Tower_ID' in df.columns and 'Utilization' in df.columns:
        underutilized = df.groupby('Tower_ID')['Utilization'].mean()
        underutilized_towers = underutilized[underutilized < 0.5].index.tolist()

    prompt = f"""
You are a telecom operations expert. Based on the following summary data from a telecom tower dataset, provide:
1. Cost reduction strategies
2. Energy optimization suggestions
3. Revenue improvement ideas
4. Risk alerts

Summary:
- Average productivity (Revenue/(Energy Cost+OPEX)): {avg_productivity:.2f}
- Tower with lowest average productivity: {worst_tower}
- Tower with highest diesel dependency: {highest_diesel}
- Underutilized towers (utilization < 0.5): {underutilized_towers if underutilized_towers else 'None'}

Please give actionable recommendations.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"

def get_chat_response(question: str, df: pd.DataFrame, chat_history: list = None):
    api_key = get_gemini_api_key()
    if not api_key:
        return "API key not found."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')

    # Build a concise summary of the data
    sample = df.head(5).to_string()
    cols = ', '.join(df.columns)
    prompt = f"""
You are a telecom operations expert. Use the following data to answer the user's question.

Dataset columns: {cols}
Sample data (first 5 rows):
{sample}

User question: {question}

Answer concisely and helpfully based only on the data.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"