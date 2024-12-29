from dotenv import load_dotenv
from pathlib import Path
import os

def load_envs():
    env_paths = [
        Path(__file__).parent.parent.parent / ".env_public",
        Path(__file__).parent.parent.parent / ".env_private"
    ]
    for env_path in env_paths:
        assert os.path.exists(env_path)
        load_dotenv(dotenv_path=str(env_path), override=True)  
