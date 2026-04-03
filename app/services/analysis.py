import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# ============================================
# AI SCHEMA MANAGEMENT
# ============================================

def detect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Automatically detect column roles using AI or heuristics.
    Returns schema with date, metrics, categories, id_columns.
    """
    schema = {
        "date": None,
        "metrics": [],
        "categories": [],
        "id_columns": []
    }
    
    # Detect date columns
    for col in df.columns:
        try:
            if pd.to_datetime(df[col], errors='coerce').notna().sum() > len(df) * 0.8:
                schema["date"] = col
                break
        except:
            pass
    
    # Detect numeric metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        # Skip ID-like columns
        if 'id' in col.lower() or 'code' in col.lower():
            schema["id_columns"].append(col)
        else:
            schema["metrics"].append(col)
    
    # Detect categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        if df[col].nunique() < 20:  # Low cardinality = category
            schema["categories"].append(col)
    
    return schema

# ============================================
# DYNAMIC KPI FUNCTIONS
# ============================================

def compute_kpis_dynamic(df: pd.DataFrame, schema: Dict) -> Dict[str, Any]:
    """Compute KPIs based on detected schema."""
    kpis = {}
    
    # Metrics KPIs
    for metric in schema["metrics"]:
        kpis[f"Total_{metric}"] = float(df[metric].sum())
        kpis[f"Avg_{metric}"] = float(df[metric].mean())
        kpis[f"Min_{metric}"] = float(df[metric].min())
        kpis[f"Max_{metric}"] = float(df[metric].max())
    
    # Category distribution
    for cat in schema["categories"]:
        top_cat = df[cat].value_counts().index[0] if len(df[cat].value_counts()) > 0 else None
        kpis[f"Top_{cat}"] = str(top_cat) if top_cat else "None"
        kpis[f"Unique_{cat}"] = int(df[cat].nunique())
    
    return kpis

def compute_totals(df: pd.DataFrame, schema: Dict) -> Dict[str, float]:
    """Compute totals for all metrics."""
    totals = {}
    for metric in schema["metrics"]:
        totals[metric] = float(df[metric].sum())
    return totals

def compute_averages(df: pd.DataFrame, schema: Dict) -> Dict[str, float]:
    """Compute averages for all metrics."""
    averages = {}
    for metric in schema["metrics"]:
        averages[metric] = float(df[metric].mean())
    return averages

def compute_correlations(df: pd.DataFrame, schema: Dict) -> Dict[str, Any]:
    """Compute correlations between numeric columns."""
    if len(schema["metrics"]) < 2:
        return {}
    
    numeric_df = df[schema["metrics"]]
    correlations = numeric_df.corr().to_dict()
    
    # Find strongest correlation
    strongest = {"col1": None, "col2": None, "value": 0}
    for col1 in correlations:
        for col2, value in correlations[col1].items():
            if col1 != col2 and abs(value) > abs(strongest["value"]):
                strongest = {"col1": col1, "col2": col2, "value": float(value)}
    
    return {
        "matrix": correlations,
        "strongest": strongest
    }

def compute_trends(df: pd.DataFrame, schema: Dict) -> Dict[str, Any]:
    """Compute trends over time if date column exists."""
    if not schema["date"] or not schema["metrics"]:
        return {}
    
    df_sorted = df.sort_values(schema["date"])
    trends = {}
    
    for metric in schema["metrics"]:
        metric_values = df_sorted[metric].values
        if len(metric_values) > 1:
            # Simple trend: compare first vs last
            first_val = float(metric_values[0])
            last_val = float(metric_values[-1])
            change = last_val - first_val
            percent_change = (change / first_val * 100) if first_val != 0 else 0
            
            trends[metric] = {
                "first": first_val,
                "last": last_val,
                "change": float(change),
                "percent_change": float(percent_change),
                "direction": "up" if change > 0 else "down" if change < 0 else "stable"
            }
    
    return trends

def compute_category_aggregates(df: pd.DataFrame, schema: Dict) -> Dict[str, Any]:
    """Compute aggregates by category."""
    if not schema["categories"] or not schema["metrics"]:
        return {}
    
    aggregates = {}
    for cat in schema["categories"]:
        agg_data = {}
        for metric in schema["metrics"]:
            agg_data[metric] = df.groupby(cat)[metric].sum().to_dict()
        aggregates[cat] = agg_data
    
    return aggregates

# ============================================
# DYNAMIC PLOTTING FUNCTIONS
# ============================================

def generate_plots_dynamic(df: pd.DataFrame, schema: Dict) -> Dict[str, str]:
    """Generate all applicable plots based on schema."""
    plots = {}
    
    # Time series plot (if date exists)
    if schema["date"] and schema["metrics"]:
        fig = px.line(df, x=schema["date"], y=schema["metrics"],
                      title=f"Time Series: {', '.join(schema['metrics'])}")
        plots["time_series"] = fig.to_json()
    
    # Bar chart for categorical data
    if schema["categories"] and schema["metrics"]:
        top_cat = schema["categories"][0]
        top_metric = schema["metrics"][0]
        cat_agg = df.groupby(top_cat)[top_metric].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=cat_agg.index, y=cat_agg.values,
                     title=f"Top {top_cat} by {top_metric}")
        plots["category_bar"] = fig.to_json()
    
    # Correlation heatmap
    if len(schema["metrics"]) >= 2:
        corr_matrix = df[schema["metrics"]].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Heatmap")
        plots["correlation_heatmap"] = fig.to_json()
    
    # Distribution histograms for metrics
    if schema["metrics"]:
        for metric in schema["metrics"][:3]:  # Limit to first 3 metrics
            fig = px.histogram(df, x=metric, title=f"Distribution: {metric}")
            plots[f"histogram_{metric}"] = fig.to_json()
    
    # Scatter plot for two strongest correlated metrics
    if len(schema["metrics"]) >= 2:
        correlations = compute_correlations(df, schema)
        if correlations.get("strongest") and correlations["strongest"]["col1"]:
            s = correlations["strongest"]
            fig = px.scatter(df, x=s["col1"], y=s["col2"],
                             title=f"Correlation: {s['col1']} vs {s['col2']} (r={s['value']:.2f})")
            plots["scatter_top"] = fig.to_json()
    
    # Pie chart for category distribution
    if schema["categories"]:
        top_cat = schema["categories"][0]
        cat_counts = df[top_cat].value_counts().head(8)
        fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                     title=f"Distribution: {top_cat}")
        plots["pie_category"] = fig.to_json()
    
    return plots

# ============================================
# AI INTEGRATION FUNCTIONS
# ============================================

def get_gemini_api_key():
    """Read Gemini API key from .env file."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except:
            pass
    return api_key

def call_gemini_api(prompt: str) -> str:
    """Call Gemini API with the prompt."""
    api_key = get_gemini_api_key()
    if not api_key:
        return "⚠️ API key not found. Please add GEMINI_API_KEY to .env file"
    
    # Try multiple methods
    try:
        # Method 1: google-genai package
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text
    except ImportError:
        pass
    except Exception as e:
        print(f"GenAI error: {e}")
    
    try:
        # Method 2: google-generativeai package
        import google.generativeai as genai_old
        genai_old.configure(api_key=api_key)
        model = genai_old.GenerativeModel('gemini-3-flash-preview')
        response = model.generate_content(prompt)
        return response.text
    except ImportError:
        pass
    except Exception as e:
        print(f"Old API error: {e}")
    
    # Method 3: Direct REST API
    try:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response")
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def get_ai_recommendations_dynamic(df: pd.DataFrame, schema: Dict, kpis: Dict, trends: Dict) -> str:
    """Generate AI recommendations based on detected schema and computed metrics."""
    prompt = f"""
You are a senior data analyst. Analyze this dataset and provide actionable recommendations.

DATASET SCHEMA:
- Date column: {schema.get('date', 'None')}
- Metrics (numeric): {', '.join(schema.get('metrics', []))}
- Categories (grouping): {', '.join(schema.get('categories', []))}
- ID columns: {', '.join(schema.get('id_columns', []))}

KEY PERFORMANCE INDICATORS:
{json.dumps({k: v for k, v in list(kpis.items())[:20]}, indent=2)}

TRENDS:
{json.dumps(trends, indent=2)}

Based on this data, provide:
1. KEY INSIGHTS - What patterns do you notice?
2. RECOMMENDATIONS - What actions should be taken?
3. RISKS - What potential problems exist?
4. OPPORTUNITIES - Where can we improve?

Be specific and data-driven. Focus on actionable insights.
"""
    return call_gemini_api(prompt)

def get_chat_response(question: str, df: pd.DataFrame, schema: Dict, kpis: Dict) -> str:
    """Get AI response for user questions about the data."""
    sample_data = df.head(10).to_string()
    
    prompt = f"""
You are a data analyst assistant. Answer questions based on this dataset.

DATASET INFO:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}
- Schema: {json.dumps(schema, indent=2)}

KEY STATISTICS:
{json.dumps(kpis, indent=2)[:2000]}

SAMPLE DATA (first 10 rows):
{sample_data}

USER QUESTION: {question}

Answer the question using ONLY the data provided. Be specific and helpful. If the question cannot be answered from the data, explain what information would be needed.
"""
    return call_gemini_api(prompt)

# ============================================
# MAIN EXPORT FUNCTION
# ============================================

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Main function to analyze any dataset dynamically."""
    # Detect schema
    schema = detect_schema(df)
    
    # Compute KPIs
    kpis = compute_kpis_dynamic(df, schema)
    totals = compute_totals(df, schema)
    averages = compute_averages(df, schema)
    correlations = compute_correlations(df, schema)
    trends = compute_trends(df, schema)
    category_aggregates = compute_category_aggregates(df, schema)
    
    # Generate plots
    plots = generate_plots_dynamic(df, schema)
    
    return {
        "schema": schema,
        "kpis": kpis,
        "totals": totals,
        "averages": averages,
        "correlations": correlations,
        "trends": trends,
        "category_aggregates": category_aggregates,
        "plots": plots,
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": df.columns.tolist()
    }