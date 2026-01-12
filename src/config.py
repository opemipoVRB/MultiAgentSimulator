# config.py
import os
from dotenv import load_dotenv

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODELS_DIR = os.getenv(
    "OLLAMA_MODELS_DIR",
    os.path.expanduser("~/.ollama/models")
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_MODE = os.getenv("OLLAMA_MODE", "external")  # external | embedded
