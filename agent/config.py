"""Cau hinh cho ai-tech-agent."""
import os
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "ai-tech-agent"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
CHUNKS_DIR = DATA_DIR / "chunks"

for d in (DATA_DIR, DOCS_DIR, CHUNKS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CHROMA_DIR = Path(user_data_dir(APP_NAME)) / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", "qwen3.5:4b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "vi")

TTS_VOICE = os.environ.get("TTS_VOICE", "vi-VN-HoaiMyNeural")
TTS_RATE = os.environ.get("TTS_RATE", "+0%")

SAMPLE_RATE = 16000
SILENCE_THRESHOLD_MS = int(os.environ.get("SILENCE_THRESHOLD_MS", "800"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

TECH_CATEGORIES = {
    "python": {"name": "Python", "docs_url": "https://docs.python.org/3/", "priority": 1},
    "javascript": {"name": "JavaScript/TypeScript", "docs_url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript", "priority": 1},
    "linux": {"name": "Linux/OS", "docs_url": "https://www.kernel.org/doc/html/latest/", "priority": 1},
    "networking": {"name": "Networking", "docs_url": "https://developer.mozilla.org/en-US/docs/Web/HTTP", "priority": 2},
    "databases": {"name": "Databases", "docs_url": "", "priority": 2},
    "devops": {"name": "DevOps", "docs_url": "", "priority": 2},
    "ai_ml": {"name": "AI/ML", "docs_url": "", "priority": 2},
    "git": {"name": "Git/VCS", "docs_url": "https://git-scm.com/doc", "priority": 2},
    "cs": {"name": "CS Fundamentals", "docs_url": "", "priority": 3},
    "go": {"name": "Go", "docs_url": "https://go.dev/doc/", "priority": 2},
    "rust": {"name": "Rust", "docs_url": "https://doc.rust-lang.org/book/", "priority": 2},
    "cpp": {"name": "C/C++", "docs_url": "", "priority": 3},
    "kotlin": {"name": "Kotlin", "docs_url": "https://kotlinlang.org/docs/home.html", "priority": 3},
    "shell": {"name": "Shell/Bash", "docs_url": "https://www.gnu.org/software/bash/manual/", "priority": 2},
}
