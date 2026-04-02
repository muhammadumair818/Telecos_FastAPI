import pandas as pd
import numpy as np
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# ============================================
# DATA PROCESSING FUNCTIONS
# ============================================

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
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# ============================================
# KPI AND PLOTTING FUNCTIONS
# ============================================

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregated KPIs with proper type conversion."""
    return {
        'Total Revenue': float(df['Revenue'].sum()),
        'Total OPEX': float(df['Total_Opex'].sum()),
        'Total Energy Cost': float(df['Total_Energy_Cost'].sum()),
        'Total Profit': float(df['Profit'].sum()),
        'Avg Productivity': float(df['Productivity'].mean()),
        'Avg Energy Efficiency': float(df['Energy_Efficiency'].mean()),
        'Avg Cost Efficiency': float(df['Cost_Efficiency'].mean()),
        'Avg Diesel Dependency': float(df['Diesel_Dependency'].mean()),
        'Avg Utilization': float(df['Utilization'].mean()),
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
    diesel_total = float(df['Diesel_Cost'].sum())
    grid_total = float(df['Grid_Energy_Cost'].sum())
    fig = px.pie(values=[diesel_total, grid_total], names=['Diesel Cost', 'Grid Energy Cost'],
                 title='Diesel vs Grid Energy Cost')
    plots['diesel_vs_grid'] = fig.to_json()

    # OPEX breakdown
    opex_cols = ['Maintenance_Cost', 'Repair_Cost', 'Staff_Visits']
    breakdown = {col: float(df[col].sum()) for col in opex_cols}
    fig = px.bar(x=list(breakdown.keys()), y=list(breakdown.values()),
                 title='OPEX Breakdown (sum over dataset)',
                 labels={'x': 'Category', 'y': 'Total Cost'})
    plots['opex_breakdown'] = fig.to_json()

    return plots

# ============================================
# GEMINI API FUNCTIONS with gemini-3-flash-preview
# ============================================

def get_gemini_api_key():
    """Read Gemini API key from .env file"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Try loading directly from .env file
        try:
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass
    return api_key

def call_gemini_api(prompt: str) -> str:
    """Call Gemini API with gemini-3-flash-preview model"""
    api_key = get_gemini_api_key()
    if not api_key:
        return "⚠️ API key not found. Please add GEMINI_API_KEY to .env file"
    
    # Method 1: Using google-genai package (recommended)
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",  # Latest free model
            contents=prompt
        )
        return response.text
    except ImportError:
        pass
    except Exception as e:
        print(f"GenAI error: {e}")
    
    # Method 2: Using google-generativeai package
    try:
        import google.generativeai as genai_old
        genai_old.configure(api_key=api_key)
        model = genai_old.GenerativeModel('gemini-3-flash-preview')
        response = model.generate_content(prompt)
        return response.text
    except ImportError:
        pass
    except Exception as e:
        print(f"GenerativeAI error: {e}")
    
    # Method 3: Direct REST API call
    try:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response from API")
        else:
            return f"API Error ({response.status_code}): {response.text[:200]}"
    except Exception as e:
        return f"Error calling Gemini API: {e}"

def get_ai_recommendations(df: pd.DataFrame) -> str:
    """Generate AI recommendations using Gemini."""
    if df is None or df.empty:
        return "No data available for analysis."
    
    avg_productivity = float(df['Productivity'].mean())
    worst_tower = df.groupby('Tower_ID')['Productivity'].mean().idxmin()
    highest_diesel = df.groupby('Tower_ID')['Diesel_Dependency'].mean().idxmax()
    underutilized = df.groupby('Tower_ID')['Utilization'].mean()
    underutilized_towers = underutilized[underutilized < 0.5].index.tolist()
    
    # Get tower statistics
    tower_stats = df.groupby('Tower_ID').agg({
        'Revenue': 'sum',
        'Total_Opex': 'sum',
        'Total_Energy_Cost': 'sum',
        'Profit': 'sum',
        'Productivity': 'mean',
        'Diesel_Dependency': 'mean',
        'Utilization': 'mean'
    }).round(2)
    
    prompt = f"""
You are a senior telecom operations expert. Analyze this tower data and provide specific, actionable recommendations.

DATA SUMMARY:
- Total Towers: {df['Tower_ID'].nunique()}
- Total Records: {len(df)}
- Average Productivity: {avg_productivity:.2f} (Target: >1.0)
- Worst Performing Tower: {worst_tower}
- Highest Diesel Dependency: {highest_diesel}
- Underutilized Towers (<50%): {', '.join(map(str, underutilized_towers)) if underutilized_towers else 'None'}

TOWER STATISTICS:
{tower_stats.to_string()}

Based on this data, provide:
1. IMMEDIATE ACTIONS (next 7 days)
2. SHORT-TERM STRATEGIES (next 30 days)
3. LONG-TERM RECOMMENDATIONS
4. RISK ALERTS

Be specific. Mention tower names. Give concrete numbers where possible.
don't give me spcial symbol like * # remove all the special symbol from the output write like a human readable text.
don't write like ai write like expert write like human
"""
    
    return call_gemini_api(prompt)

def get_gemini_response(system_prompt: str, user_message: str) -> str:
    """Get response from Gemini API with context about the data."""
    api_key = get_gemini_api_key()
    if not api_key:
        return "⚠️ API key not found. Please add GEMINI_API_KEY to .env file"
    
    full_prompt = f"""{system_prompt}

User Question: {user_message}

Please answer the user's question based ONLY on the data provided in the system prompt. If the question cannot be answered from the data, explain what additional information would be needed. Be specific and mention tower names and numbers when relevant.
"""
    
    return call_gemini_api(full_prompt)

def get_gemini_response_with_image(system_prompt: str, user_message: str, image=None) -> str:
    """Get response from Gemini API with optional image."""
    api_key = get_gemini_api_key()
    if not api_key:
        return "⚠️ API key not found. Please add GEMINI_API_KEY to .env file"
    
    full_prompt = f"{system_prompt}\n\nUser Question: {user_message}\n\nAnswer based on the data and image provided."
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        if image:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[full_prompt, image]
            )
        else:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=full_prompt
            )
        return response.text
    except Exception as e:
        return f"Error with image: {e}\n\n{call_gemini_api(full_prompt)}"