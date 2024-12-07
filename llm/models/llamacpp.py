# from langchain.llms import LlamaCpp
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_experimental.chat_models import Llama2Chat
from os.path import expanduser

import os

# https://python.langchain.com/v0.2/docs/integrations/chat/llama2_chat/

# 加载模型
# llama-2-7b-chat.Q4_K_M.gguf
#     https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf
#         https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
# llama-2-13b-chat.Q4_K_M.gguf

# pip install llama-cpp-python

class LlamaCppExecutor():
    def __init__(self, **config):
        self.cache_dir = config.get("cache_dir", None)
        if not self.cache_dir:
            base_dir = config.get("base_dir", "/home/aurora")
            if base_dir:
                os.path.join(base_dir,"models","llama")
            else :
                print(f"cache_dir:{self.cache_dir} or base_dir: {base_dir} is not corrected setting")
                
        self.device = config.get("device", "cpu")
        self.model_name = config.get("model_name", "llama-2-7b-chat.Q4_K_M.gguf")
        
        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

#         MODEL_DIR = "/home/aurora/models/llama"
#         model_name = "llama-2-7b-chat.Q4_K_M.gguf"
        model_path = os.path.join(self.cache_dir,self.model_name)
        model_path = expanduser(model_path)

        model_kwargs = {
            "model_path": model_path,
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
        
        if config.get("model_kwargs", None):
            model_kwargs.update(config.get("model_kwargs"))
            
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
            ** model_kwargs
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