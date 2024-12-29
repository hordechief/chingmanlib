import os
from chingmanlib.llm.models.llm_factory import LLMFactory

def create_llm(llm_provider, cache_dir=None, **kwargs):
    '''
    model_name
    model_type: generation, chat (OpenAI/ChatOpenAI, Ollama/ChatOllama)
    model_kwargs
    llm_provider / service_wrapper: hugggingface, lammacpp, openai, ollama
    variant: AzureOpenAI / OpenAI
    '''
    assert llm_provider in ['hf','llama_cpp', "openai", "ollama"]

    model_kwargs = {}
    llm_kwargs = {}

    if "model_name" in kwargs:
        model_kwargs["model_name"] = kwargs["model_name"]
    model_kwargs["temperature"] = kwargs["temperature"] if "temperature" in kwargs else 0
            
    if "llama_cpp" == llm_provider:
        # model_kwargs = {
        #     "model_name": kwargs.get("model_name", "llama-2-7b-chat.Q4_K_M.gguf"),
        # }
        pass
    elif "hf" == llm_provider:
        # model_kwargs = {
        #     # "model_name": kwargs.get("model_name", "NousResearch/Llama-2-7b-chat-hf"),
        #     # "model_name":"NousResearch/Llama-2-7b-hf",
        #     'temperature':0
        # }
        pass
    elif "openai" == llm_provider:
        # model_kwargs = {
        #     "model_name":kwargs.get("model_name", "gpt-3.5-turbo"),
        #     'temperature':0
        # }
        llm_kwargs["openai_wrapper_service"] = "LangchainOpenAI"
        llm_kwargs["openai_model_type"] = "ChatOpenAI"
    elif "ollama" == llm_provider:
        # model_kwargs = {
        #     "model_name": kwargs.get("model_name", "EntropyYue/chatglm3:6b"),
        # }
        pass

    llm_kwargs.update({
        "cache_dir": cache_dir,
        "model_kwargs":model_kwargs})    

    llm_executor = LLMFactory.create_llm_executor(
            llm_provider=llm_provider, **llm_kwargs
    )

    # llm = llm_executor.llm

    return llm_executor