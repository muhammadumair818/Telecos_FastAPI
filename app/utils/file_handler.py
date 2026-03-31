import os
import shutil
from pathlib import Path
from fastapi import UploadFile
import pandas as pd

UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to data directory and return the file path."""
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return str(file_path)

def load_dataframe(file_path: str):
    """Import here to avoid circular imports."""
    from app.services.analysis import load_dataframe as _load
    return _load(file_path)