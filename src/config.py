# config.py
import os
from dotenv import load_dotenv

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODELS_DIR = os.getenv(
    "OLLAMA_MODELS_DIR",
    os.path.expanduser("~/.ollama/models")
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_MODE = os.getenv("OLLAMA_MODE", "embedded")  # external | embedded
