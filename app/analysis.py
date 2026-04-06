import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from google import genai
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
    
    # 1. Convert Date if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # 2. FORCE columns to be numeric (Important for Plotly)
    cols_to_fix = ['Revenue', 'Total_Opex', 'Total_Energy_Cost', 'Diesel_Cost', 
                   'Grid_Energy_Cost', 'Active_Tenants', 'Capacity', 
                   'Maintenance_Cost', 'Repair_Cost', 'Staff_Visits']
    
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing numeric with 0
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # 3. Derive metrics
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
    # Use sum/mean only if column exists
    mapping = {
        'Revenue': 'Total Revenue',
        'Total_Opex': 'Total OPEX',
        'Total_Energy_Cost': 'Total Energy Cost',
        'Profit': 'Total Profit'
    }
    for col, label in mapping.items():
        if col in df.columns:
            kpis[label] = float(df[col].sum())

    avg_mapping = {
        'Productivity': 'Avg Productivity',
        'Energy_Efficiency': 'Avg Energy Efficiency',
        'Cost_Efficiency': 'Avg Cost Efficiency',
        'Diesel_Dependency': 'Avg Diesel Dependency',
        'Utilization': 'Avg Utilization'
    }
    for col, label in avg_mapping.items():
        if col in df.columns:
            kpis[label] = float(df[col].mean())
            
    return kpis

# ---------------------------
# Plotting functions (return Plotly JSON)
# ---------------------------

def plot_cost_vs_revenue(df):
    # Removed strict 'Tower_ID' check so it plots even if Tower_ID is missing
    if 'Total_Energy_Cost' in df.columns and 'Revenue' in df.columns:
        color_param = 'Tower_ID' if 'Tower_ID' in df.columns else None
        fig = px.scatter(df, x='Total_Energy_Cost', y='Revenue', color=color_param,
                         title='Cost vs Revenue', 
                         hover_data=['Date'] if 'Date' in df.columns else None)
        return fig.to_json()
    return None

def plot_energy_vs_revenue(df):
    if 'Total_Energy_Cost' in df.columns and 'Revenue' in df.columns:
        color_param = 'Tower_ID' if 'Tower_ID' in df.columns else None
        fig = px.scatter(df, x='Total_Energy_Cost', y='Revenue', color=color_param,
                         title='Energy Cost vs Revenue')
        return fig.to_json()
    return None

def plot_utilization_vs_revenue(df):
    if 'Utilization' in df.columns and 'Revenue' in df.columns:
        color_param = 'Tower_ID' if 'Tower_ID' in df.columns else None
        fig = px.scatter(df, x='Utilization', y='Revenue', color=color_param,
                         title='Utilization vs Revenue')
        return fig.to_json()
    return None

def plot_diesel_vs_grid(df):
    if 'Diesel_Cost' in df.columns and 'Grid_Energy_Cost' in df.columns:
        diesel_total = df['Diesel_Cost'].sum()
        grid_total = df['Grid_Energy_Cost'].sum()
        # Only plot if there is actual data to show
        if (diesel_total + grid_total) > 0:
            fig = px.pie(values=[diesel_total, grid_total], names=['Diesel Cost', 'Grid Energy Cost'],
                         title='Diesel vs Grid Energy Cost')
            return fig.to_json()
    return None

def plot_opex_breakdown(df):
    opex_cols = [c for c in ['Maintenance_Cost', 'Repair_Cost', 'Staff_Visits'] if c in df.columns]
    if opex_cols:
        breakdown = df[opex_cols].sum()
        if breakdown.sum() > 0:
            fig = px.bar(x=breakdown.index, y=breakdown.values,
                         title='OPEX Breakdown',
                         labels={'x': 'Category', 'y': 'Total Cost'})
            return fig.to_json()
    return None

def generate_all_plots(df):
    plots = {}
    # Use a list of functions to avoid repeating code
    funcs = {
        'cost_vs_revenue': plot_cost_vs_revenue,
        'energy_vs_revenue': plot_energy_vs_revenue,
        'utilization_vs_revenue': plot_utilization_vs_revenue,
        'diesel_vs_grid': plot_diesel_vs_grid,
        'opex_breakdown': plot_opex_breakdown
    }
    for key, func in funcs.items():
        res = func(df)
        if res:
            plots[key] = res
    return plots
# ---------------------------
# ML Models (Names preserved for HTML link)
# ---------------------------

def train_revenue_model(df):
    required = ['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex', 'Revenue']
    # Check if columns exist
    if not all(col in df.columns for col in required):
        return None, None, None
    
    # 1. Drop rows with missing values specifically in these columns
    df_clean = df.dropna(subset=required)
    
    if len(df_clean) < 10:  # Need enough data to split 80/20
        return None, None, None

    features = ['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex']
    X = df_clean[features].values
    y = df_clean['Revenue'].values
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return model, score, (X_test, y_test)
    except:
        return None, None, None

def train_cost_model(df):
    required = ['Diesel_Liters', 'Electricity_kWh', 'Maintenance_Cost', 'Repair_Cost', 'Staff_Visits', 'Total_Opex']
    if not all(col in df.columns for col in required):
        return None, None, None
    
    # Clean NaNs
    df_clean = df.dropna(subset=required)
    
    if len(df_clean) < 10:
        return None, None, None

    features = ['Diesel_Liters', 'Electricity_kWh', 'Maintenance_Cost', 'Repair_Cost', 'Staff_Visits']
    X = df_clean[features].values
    y = df_clean['Total_Opex'].values
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return model, score, (X_test, y_test)
    except:
        return None, None, None

def train_classification_model(df):
    # Check if Productivity exists; if not, the classifier can't run
    if 'Productivity' not in df.columns:
        return None, None, None
    
    # 1. Create labels
    df = df.copy()
    df['Productivity_Label'] = pd.cut(df['Productivity'],
                                      bins=[-np.inf, 1, 1.5, np.inf],
                                      labels=['Low', 'Medium', 'High'])
    
    features = ['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex']
    target = 'Productivity_Label'
    
    if not all(col in df.columns for col in features):
        return None, None, None

    # 2. Clean data (Remove NaNs and Infs)
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target])
    
    # 3. Validation: Must have at least 2 classes and enough rows
    if len(df_clean) < 10 or len(df_clean[target].unique()) < 2:
        return None, None, None
        
    X = df_clean[features].values
    y = df_clean[target].astype(str).values # Ensure y is string for classifier
    
    try:
        # Note: Removed stratify=y because it crashes on very small datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return model, acc, (X_test, y_test)
    except:
        return None, None, None

def get_ai_client():
    """Initialize the new Gemini Client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def get_ai_recommendations(df):
    client = get_ai_client()
    if not client:
        return "API key not found. Please set GEMINI_API_KEY in .env"
    
    # --- Data Extraction (Unchanged) ---
    stats = {
        "avg_prod": df['Productivity'].mean() if 'Productivity' in df.columns else "N/A",
        "total_rev": df['Revenue'].sum() if 'Revenue' in df.columns else "N/A",
        "total_opex": df['Total_Opex'].sum() if 'Total_Opex' in df.columns else "N/A",
        "diesel_ratio": (df['Diesel_Cost'].sum() / df['Total_Energy_Cost'].sum() * 100) 
                        if 'Diesel_Cost' in df.columns and 'Total_Energy_Cost' in df.columns else "N/A"
    }

    worst_tower = df.groupby('Tower_ID')['Productivity'].mean().idxmin() if 'Tower_ID' in df.columns and 'Productivity' in df.columns else "Unknown"
    underutilized = []
    if 'Tower_ID' in df.columns and 'Utilization' in df.columns:
        u_map = df.groupby('Tower_ID')['Utilization'].mean()
        underutilized = u_map[u_map < 0.5].index.tolist()

    prompt = f"""
    SYSTEM: You are a Senior Telecom Operations Consultant. Your goal is to maximize EBITDA and Operational Efficiency.
    
    DATA SUMMARY:
    - Network Productivity (Avg): {stats['avg_prod']}
    - Total Portfolio Revenue: {stats['total_rev']}
    - Total Portfolio OPEX: {stats['total_opex']}
    - Diesel Dependency Ratio: {stats['diesel_ratio']}%
    - Critical Underperformer (Tower ID): {worst_tower}
    - Underutilized Assets (<50%): {underutilized}

    TASK: Provide a professional Executive Briefing including:
    1. STRATEGIC COST REDUCTION: Focus on OPEX and Energy.
    2. ENERGY TRANSITION: Suggestions to reduce diesel dependency.
    3. REVENUE MAXIMIZATION: Strategies for underutilized towers (e.g., colocation).
    4. OPERATIONAL RISK ASSESSMENT: Identify critical failure points based on the data.

    FORMAT: Use Markdown with bold headers. Keep it technical, actionable, and concise.
    """

    try:
        # New SDK syntax for content generation
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Operational Error: {e}"

def get_chat_response(question: str, df: pd.DataFrame, chat_history: list = None):
    client = get_ai_client()
    if not client:
        return "System Error: API Key missing."

    data_profile = df.describe().to_string()
    cols = ', '.join(df.columns)
    
    prompt = f"""
    SYSTEM: You are an Expert Telecom Data Analyst. You have access to a tower infrastructure dataset.
    
    DATA CONTEXT:
    - Available Metrics: {cols}
    - Statistical Snapshot:
    {data_profile}

    USER QUERY: {question}

    INSTRUCTIONS:
    - Answer using data-driven insights.
    - If the user asks for a specific tower or trend, refer to the metrics provided.
    - Maintain a professional, helpful, and objective tone.
    - If the data doesn't contain the answer, politely inform the user what is missing.
    """
    
    try:
        # New SDK syntax for content generation
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Analytic Error: {e}"