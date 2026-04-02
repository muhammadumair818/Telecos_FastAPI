# test_env.py
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")
print(f"First 10 chars: {api_key[:10] if api_key else 'None'}...")