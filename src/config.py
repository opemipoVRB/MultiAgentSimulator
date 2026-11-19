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
# Context window (tokens) of every model request: the prompt and the reply must fit in it, or the server silently cuts the
# prompt (at its 4096 default the longest world prompt, 4,758 tokens, was cut to 2,050). One fixed value, so the server does not
# reload the model between requests.
SLM_NUM_CTX = int(os.getenv("SLM_NUM_CTX", "20480"))
OLLAMA_MODE = os.getenv("OLLAMA_MODE", "embedded")  # external | embedded

# Every run of this system is one process per drone, each pinned to its own core (src/drone_process.py): a drone knows only the start information and one
# Observation per step. There is no other mode, so the setting is forced here (src/config.py is imported by everything that runs an experiment).
import os as _os
_os.environ["DRONE_PROCESSES"] = "1"
