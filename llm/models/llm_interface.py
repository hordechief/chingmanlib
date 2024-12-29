import abc
import os
from chingmanlib.llm.utils.operation import get_device, get_cache_dir

class LLMInterface(metaclass=abc.ABCMeta):

    def __init__(self, **config):
        self.device = get_device()    
        self.cache_dir = get_cache_dir(**config)

    @abc.abstractmethod
    def run(self, prompt: str) -> str:
        """
        Parameters:
        - prompt (str): The message or question to send to the agent.

        Returns:
        - str: the response from the model.
        """
        raise NotImplementedError   

class EmbeddingsInterface(metaclass=abc.ABCMeta):
    def __init__(self, **config):
        self.device = get_device()    
        self.cache_dir = get_cache_dir(**config)

    # @abc.abstractmethod
    # def embed_query(text):
    #     raise NotImplemented
    
    # @abc.abstractmethod
    # def embed_documents(documents):
    #     raise NotImplemented    