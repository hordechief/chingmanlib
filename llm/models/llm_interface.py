import abc
import os
from chingmanlib.llm.utils.operation import get_default_cache_dir

class LLMInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run(self, prompt: str) -> str:
        """
        Parameters:
        - prompt (str): The message or question to send to the agent.

        Returns:
        - str: the response from the model.
        """
        raise NotImplementedError
    
    @staticmethod        
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
        
    @staticmethod
    def get_cache_dir(**config):
        if config.get("cache_dir", None):
            return config.get("cache_dir")
        else:
            return os.getenv("MODEL_CACHE_DIR", get_default_cache_dir())        
        
    @staticmethod
    def get_param(param, ENV_NAME, default_value, **config):
        # print(config)
        if config.get(param, None):
            return config.get("cache_dir")
        else:
            return os.getenv(ENV_NAME, default_value)                