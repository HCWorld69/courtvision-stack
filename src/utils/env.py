from pathlib import Path
import os
from dotenv import load_dotenv


def load_env(env_path: str | None = None) -> str:
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required. Set it in .env.")
    return api_key
