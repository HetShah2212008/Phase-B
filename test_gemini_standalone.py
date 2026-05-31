"""
Standalone Gemini connectivity test.
Run this OUTSIDE FastAPI to verify the API key works before starting the server.

Usage:
    python test_gemini_standalone.py

Expected output when key is valid:
    Key prefix : AIzaSy...
    Response   : Hello! ...
    SUCCESS: Gemini is working correctly.

If you see 401 UNAUTHENTICATED, the key is wrong or not enabled for Gemini API.
If you see 429 RESOURCE_EXHAUSTED, the key is valid but quota is exceeded.
"""

import sys
from pathlib import Path

# Load the key from .env so we test the exact same value the app uses
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GEMINI_API_KEY", "")
model   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

print(f"Key prefix  : {api_key[:10]}...")
print(f"Key length  : {len(api_key)}")
print(f"Starts AIza : {api_key.startswith('AIza')}  ← must be True for a valid API key")
print(f"Model       : {model}")
print()

if not api_key.startswith("AIza"):
    print("ERROR: This does not look like a Gemini API key.")
    print("  - Valid keys start with 'AIza' and are 39 characters long.")
    print("  - Keys starting with 'AQ.' are OAuth2 access tokens — wrong type.")
    print("  - Get a real key at: https://aistudio.google.com/app/apikey")
    sys.exit(1)

from google import genai

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model=model,
        contents="Say hello in one short sentence.",
    )
    print(f"Response    : {response.text}")
    print()
    print("SUCCESS: Gemini is working correctly.")
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")
    sys.exit(1)
