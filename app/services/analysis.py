import pandas as pd
import numpy as np
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import json

def validate_columns(df: pd.DataFrame, required_cols: list) -> bool:
    """Check if all required columns are present."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False
    return True

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Date, handle missing values, add derived columns."""
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    denominator = df['Total_Energy_Cost'] + df['Total_Opex']
    df['Productivity'] = df['Revenue'] / denominator.replace(0, np.nan)
    df['Energy_Efficiency'] = df['Revenue'] / df['Total_Energy_Cost'].replace(0, np.nan)
    df['Cost_Efficiency'] = df['Revenue'] / df['Total_Opex'].replace(0, np.nan)
    df['Diesel_Dependency'] = df['Diesel_Cost'] / df['Total_Energy_Cost'].replace(0, np.nan)
    df['Utilization'] = df['Active_Tenants'] / df['Capacity'].replace(0, np.nan)
    df['Profit'] = df['Revenue'] - (df['Total_Energy_Cost'] + df['Total_Opex'])
    df.fillna(0, inplace=True)
    return df

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregated KPIs."""
    return {
        'Total Revenue': df['Revenue'].sum(),
        'Total OPEX': df['Total_Opex'].sum(),
        'Total Energy Cost': df['Total_Energy_Cost'].sum(),
        'Total Profit': df['Profit'].sum(),
        'Avg Productivity': df['Productivity'].mean(),
        'Avg Energy Efficiency': df['Energy_Efficiency'].mean(),
        'Avg Cost Efficiency': df['Cost_Efficiency'].mean(),
        'Avg Diesel Dependency': df['Diesel_Dependency'].mean(),
        'Avg Utilization': df['Utilization'].mean(),
    }

def generate_plots(df: pd.DataFrame) -> Dict[str, str]:
    """Generate Plotly figures and return them as JSON strings."""
    plots = {}

    # Cost vs Revenue scatter
    fig = px.scatter(df, x='Total_Energy_Cost', y='Revenue', color='Tower_ID',
                     title='Cost vs Revenue', hover_data=['Date'])
    plots['cost_vs_revenue'] = fig.to_json()

    # Energy vs Revenue scatter
    fig = px.scatter(df, x='Total_Energy_Cost', y='Revenue', color='Tower_ID',
                     title='Energy Cost vs Revenue', hover_data=['Date'])
    plots['energy_vs_revenue'] = fig.to_json()

    # Utilization vs Revenue
    fig = px.scatter(df, x='Utilization', y='Revenue', color='Tower_ID',
                     title='Utilization vs Revenue', hover_data=['Date'])
    plots['utilization_vs_revenue'] = fig.to_json()

    # Diesel vs Grid pie
    diesel_total = df['Diesel_Cost'].sum()
    grid_total = df['Grid_Energy_Cost'].sum()
    fig = px.pie(values=[diesel_total, grid_total], names=['Diesel Cost', 'Grid Energy Cost'],
                 title='Diesel vs Grid Energy Cost')
    plots['diesel_vs_grid'] = fig.to_json()

    # OPEX breakdown
    opex_cols = ['Maintenance_Cost', 'Repair_Cost', 'Staff_Visits']
    breakdown = df[opex_cols].sum().to_dict()
    fig = px.bar(x=list(breakdown.keys()), y=list(breakdown.values()),
                 title='OPEX Breakdown (sum over dataset)',
                 labels={'x': 'Category', 'y': 'Total Cost'})
    plots['opex_breakdown'] = fig.to_json()

    return plots

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load CSV or Excel file into DataFrame."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        # Validate required columns
        required_cols = [
            'Transaction_ID', 'Date', 'Tower_ID', 'Location', 'Transaction_Type',
            'Diesel_Liters', 'Electricity_kWh', 'Grid_Energy_Cost', 'Diesel_Cost',
            'Total_Energy_Cost', 'Maintenance_Cost', 'Repair_Cost', 'Staff_Visits',
            'Total_Opex', 'Revenue', 'Active_Tenants', 'Capacity'
        ]
        if not validate_columns(df, required_cols):
            return None
        df = preprocess_data(df)
        return df
    except Exception:
        return None