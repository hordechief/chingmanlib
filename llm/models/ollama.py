# from langchain_community.llms import Ollama # deprecated
from langchain_ollama import OllamaLLM as Ollama

from .llm_interface import LLMInterface

class LLMOllamaExecutor(LLMInterface):
    
    def __init__(self, **config):        
        model_kwargs = config.get("model_kwargs", {}) 
        model_name = model_kwargs.get("model_name", "EntropyYue/chatglm3:6b" )    # THUDM/chatglm-6b
        self.llm = Ollama(model=model_name)
    
    def run(self,prompt):
        return self.llm(prompt)    