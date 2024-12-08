from .llm_interface import LLMInterface
from typing import Type

from .hf import LLMHFxecutor
from .llamacpp import LlamaCppExecutor
from .openai import OpenAIExecutor

class LLMFactory:
    @staticmethod
    def create_llm_executor(llm_provider, **kwargs) -> Type[LLMInterface]:
        # print(kwargs)
        if "hf" == llm_provider:
            llm_executor = LLMHFxecutor(
                cache_dir=kwargs.get("cache_dir", None), 
                device=kwargs.get("device", None),
                model_kwargs = kwargs.get("model_kwargs", None)
            )
        elif "lamma_cpp" == llm_provider:
            llm_executor = LlamaCppExecutor(
                cache_dir=kwargs.get("cache_dir", None), 
                device=kwargs.get("device", "cpu"),
                model_kwargs = kwargs.get("model_kwargs", None)                
            )
        elif "openai" == llm_provider:
            llm_executor = OpenAIExecutor(
                **kwargs
            )            
        else:
            raise ValueError("llm provider must be given")
        
        return llm_executor