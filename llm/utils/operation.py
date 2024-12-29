from pathlib import Path
import os

def get_default_cache_dir():
    if os.name == 'nt':  # Windows
        return Path(os.getenv('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    elif os.name == 'posix':  # macOS and Linux
        return Path(os.getenv('XDG_CACHE_HOME', Path.home() / '.cache'))
    else:
        raise NotImplementedError("Unsupported operating system")

def get_param(param, ENV_NAME, default_value, **config):
    '''
    param: cache_dir
    env_nameL MODEL_CACHE_DIR
    default value: ~/.cache
    '''
    if config.get(param, None):
        return config.get(param)
    else:
        return os.getenv(ENV_NAME, default_value) 

def get_cache_dir(**kwargs):
    return kwargs.get("cache_dir", None) or os.getenv("MODEL_CACHE_DIR", str(get_default_cache_dir()))    

def get_device():
    try:    
        import torch

        if torch.cuda.is_available():
            # device = torch.device("cuda")
            # print("CUDA is available. Using GPU.") # pylint 
            device = "cuda"
        else:
            device = torch.device("cpu")
            # print("CUDA is not available. Using CPU.")
            device = "cpu"

        return device
    except:
        return "cpu"    
    
if __name__ == "main":
    cache_dir = get_default_cache_dir()
    print(f"Default cache directory: {cache_dir}")
