from pathlib import Path
import os

def get_default_cache_dir():
    if os.name == 'nt':  # Windows
        return Path(os.getenv('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    elif os.name == 'posix':  # macOS and Linux
        return Path(os.getenv('XDG_CACHE_HOME', Path.home() / '.cache'))
    else:
        raise NotImplementedError("Unsupported operating system")

if __name__ == "main":
    cache_dir = get_default_cache_dir()
    print(f"Default cache directory: {cache_dir}")
