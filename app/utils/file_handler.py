import os
import shutil
import pandas as pd
from pathlib import Path
from fastapi import UploadFile
from typing import Optional

# Directories
UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

# ============================================
# FILE SAVING FUNCTIONS
# ============================================

def save_uploaded_file(file: UploadFile) -> str:
    """
    Save uploaded file to data directory.
    Returns the file path.
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Generate unique filename to avoid collisions
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
    file_path = UPLOAD_DIR / safe_filename
    
    # Save file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return str(file_path)

# ============================================
# DATA LOADING FUNCTIONS
# ============================================

def load_dataframe(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load CSV or Excel file into pandas DataFrame.
    Returns None if loading fails.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return None
        
        # Basic validation
        if df.empty:
            return None
        
        # Clean column names (remove spaces, make lowercase)
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        return df
    
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# ============================================
# DATA PREVIEW FUNCTIONS
# ============================================

def get_data_preview(df: pd.DataFrame, rows: int = 10) -> dict:
    """
    Get preview of dataframe as dictionary.
    """
    return {
        "head": df.head(rows).to_dict(orient="records"),
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

def get_data_stats(df: pd.DataFrame) -> dict:
    """
    Get basic statistics of dataframe.
    """
    stats = {
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().sum() / len(df) * 100).to_dict(),
        "numeric_stats": {}
    }
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        stats["numeric_stats"][col] = {
            "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
            "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
            "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
            "std": float(df[col].std()) if not pd.isna(df[col].std()) else None
        }
    
    return stats

# ============================================
# CLEANUP FUNCTIONS
# ============================================

def delete_uploaded_file(file_path: str) -> bool:
    """
    Delete an uploaded file.
    Returns True if successful.
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        return False
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
        return False

def cleanup_old_files(max_age_hours: int = 24):
    """
    Delete files older than specified hours.
    """
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    deleted_count = 0
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
                deleted_count += 1
    
    return {"deleted_count": deleted_count}