import os
from chingmanlib.llm.models.llm_factory import LLMFactory

def create_llm(llm_provider, cache_dir):
    assert llm_provider in ['hf','llama_cpp', "openai"]

    kwargs = {}
    if "llama_cpp" == llm_provider:
        model_kwargs = {
            "model_name":"llama-2-7b-chat.Q4_K_M.gguf",
        }
    elif "hf" == llm_provider:
        model_kwargs = {
            "model_name":"NousResearch/Llama-2-7b-chat-hf",
            # "model_name":"NousResearch/Llama-2-7b-hf",
            'temperature':0
        }
    elif "openai" == llm_provider:
        model_kwargs = {
            "model_name":"gpt-3.5-turbo",
            'temperature':0
        }
        kwargs["openai_wrapper_service"] = "LangchainOpenAI"
        # kwargs["openai_model_type"] = "OpenAI"
        kwargs["openai_model_type"] = "ChatOpenAI"
        
    kwargs.update({
        "cache_dir": cache_dir,
        "model_kwargs":model_kwargs})    

    llm_executor = LLMFactory.create_llm_executor(
            llm_provider=llm_provider, **kwargs
    )

    # llm = llm_executor.llm

    return llm_executor