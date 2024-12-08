# from langchain.llms import LlamaCpp
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_experimental.chat_models import Llama2Chat
from os.path import expanduser

import os

from .llm_interface import LLMInterface
# from .. import utils

# https://python.langchain.com/v0.2/docs/integrations/chat/llama2_chat/

# pip install llama-cpp-python

class LlamaCppExecutor(LLMInterface):
    def __init__(self, **config):
        self.cache_dir = self.get_cache_dir(config)
                
        self.device = config.get("device", "cpu")
        
        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # model 
        model_kwargs_default = {
            'temperature': 0.75,
            'max_tokens': 4000, 
            "n_ctx":4096,
            'top_p' :1,
            "n_batch": 1024,
            "n_gpu_layers": 40,
            "streaming": True,
            "verbose": True,  # Verbose is required to pass to the callback manager
            "callback_manager": callback_manager,            
        }

        model_kwargs = config.get("model_kwargs", {})
        model_name = model_kwargs.get("model_name", "llama-2-7b-chat.Q4_K_M.gguf")
        model_path = expanduser(os.path.join(self.cache_dir, model_name))
        model_kwargs["model_path"] = model_path
        
        model_kwargs_default.update(model_kwargs)
            
        # Make sure the model path is correct for your system!
        self.llm = LlamaCpp(
            # model_path = model_path,
            # temperature=0.75,
            # max_tokens=4000,
            # n_ctx=4096,
            # top_p=1,
            # n_batch=1024,
            # n_gpu_layers=40,
            # streaming=True,
            # verbose=True,  # Verbose is required to pass to the callback manager
            # callback_manager=callback_manager,
            **model_kwargs_default
        )
        
        self.llm_chat = Llama2Chat(llm=self.llm)
                
    def run(self,prompt):
        self.llm(prompt)
        
if __name__ == "__main__":
    config = {
        "base_dir" : "C:\\workings\\workspace",
        "cache_dir": os.path.join("C:\\workings\\workspace","models","llama"),
        "device": "cpu",
        "model_name": "llama-2-7b-chat.Q4_K_M.gguf",
    }
        
    llama_llm = LlamaCppExecutor(**config)
    llama_llm.run("What is Python?")