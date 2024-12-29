from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
# from langchain import HuggingFacePipeline # deprecated
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline # deprecated
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage
# from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings # deprecated
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings as LEH_HuggingFaceEmbeddings # deprecated
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# https://python.langchain.com/api_reference/huggingface/llms/langchain_huggingface.llms.huggingface_pipeline.HuggingFacePipeline.html

import torch
import os 

from .llm_interface import LLMInterface, EmbeddingsInterface
from chingmanlib.llm.prompts.utils import PromptUtils

# from transformers import AutoTokenizer, CLIPTextModel
# clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32",proxies = {"http":"127.0.0.1:7890"})
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32",proxies = {"http":"127.0.0.1:7890"})

# pip install transformers accelerate

class LLMHFxecutor(LLMInterface):

    def __init__(self, **config): 
        
        super().__init__(**config)

        model_kwargs = config.get("model_kwargs", {})
        # NousResearch/Llama-2-7b-chat-hf, NousResearch/Llama-2-7b-hf
        model_name = model_kwargs.get("model_name", "NousResearch/Llama-2-7b-chat-hf") 
        # model_name = model_kwargs.get("model_name", "THUDM/chatglm-6b" ) 

        # AutoTokenizer 是 Hugging Face 中的自动化工具，用于根据指定的模型名称或路径自动加载适合该模型的分词器（Tokenizer）。
        # 分词器的作用是将文本输入转换为模型能够理解的数字化表示形式（即 token 或 token ID）。
        # 不同的模型可能使用不同的分词方式，因此 AutoTokenizer 提供了统一的接口，自动选择正确的分词器。
        tokenizer_kwargs = {"cache_dir": self.cache_dir}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,**tokenizer_kwargs)

        # AutoModelForCausalLM 是用于加载因果语言建模（Causal Language Modeling）任务的预训练模型的自动化工具。
        # 因果语言模型是一种典型的生成模型，主要用于生成文本。
        # AutoModelForCausalLM 会自动根据指定的模型名称或路径加载一个合适的模型，比如 GPT-2、GPT-3 等。
        # 这个工具提供了一个统一的接口，简化了加载模型的过程。
        auto_model_kwargs = {
            'device_map': 'auto',
            'torch_dtype': torch.float16,
            'load_in_4bit': True,
            'bnb_4bit_quant_type': "nf4",
            'bnb_4bit_compute_dtype': torch.float16
        }
        if self.cache_dir:
            auto_model_kwargs['cache_dir']=self.cache_dir
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    # "NousResearch/Llama-2-7b-chat-hf",
                                                    # device_map='auto',
                                                    # torch_dtype=torch.float16,
                                                    # load_in_4bit=True,
                                                    # bnb_4bit_quant_type="nf4",
                                                    # bnb_4bit_compute_dtype=torch.float16,
                                                    # cache_dir = self.cache_dir,
                                                     **auto_model_kwargs)

        # pipeline 是 Hugging Face 提供的一个高层封装工具，简化了使用预训练模型的流程。
        # 它集成了模型、分词器、推理逻辑等，可以直接用于执行特定任务，如文本生成、情感分析、翻译等。
        # 使用 pipeline 可以快速构建一个端到端的任务执行流程，而无需手动处理分词、模型加载、推理等细节。
            
        pipeline_kwargs = {
            "model":self.model,
            "tokenizer": self.tokenizer,
            "torch_dtype":torch.float16,
            "device_map":'auto',
            "max_new_tokens":512,
            "do_sample":True, 
            "top_k":30,
            "num_return_sequences":1,
            "eos_token_id":self.tokenizer.eos_token_id,
        }
        
        if config.get("pipeline_kwargs", None):
            pipeline_kwargs.update(config.get("pipeline_kwargs"))
                    
        self.pipe = pipeline("text-generation",
                        # model=self.model,
                        # tokenizer= self.tokenizer,
                        # torch_dtype=torch.float16,
                        # device_map="auto",
                        # max_new_tokens = 512,
                        # do_sample=True,
                        # top_k=30,
                        # num_return_sequences=1,
                        # eos_token_id=self.tokenizer.eos_token_id,
                        **pipeline_kwargs
                        )

        # HuggingFacePipeline 是 LangChain 中的一个组件，用于将 Hugging Face 的模型与 LangChain 框架整合。
        # 它将 Hugging Face 提供的模型封装成一个 LangChain 的管道组件，使得这些模型可以在更复杂的工作流中与其他 NLP 组件协同工作。
        # 这在构建多步骤的自然语言处理应用时非常有用，尤其是当你需要将 Hugging Face 的模型作为其中一个步骤时。

        model_kwargs = {
            'skip_prompt':True,
            'temperature':0.7,
            'max_length': 256, 
            'top_k' :50
        }
        
        model_kwargs.update(config.get("model_kwargs",{}))
            
        self.llm = HuggingFacePipeline(
            pipeline = self.pipe, 
            model_kwargs = model_kwargs
        )
        
        # output of HuggingFacePipeline is string rather than an AIMessage
             
    def run(self,prompt, to_message=False, verbose=False): 
        # resp =  self.llm(prompt) # THE RESPONE WILL INCLUDE BOTH QUESTION & ANSWER
        resp = self.llm.invoke(prompt) # ONLY RETURN ANSWER
        if to_message:
            resp = AIMessage(resp)
        return resp
            
    def generate(self, prompt):        
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
            outputs = self.model.generate(**inputs,
                                     max_new_tokens=512,
                                     eos_token_id=self.tokenizer.eos_token_id,
                                     pad_token_id=self.tokenizer.eos_token_id,
                                     )
            final_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return final_outputs#, outputs

    ####################
    def test_generate(self, text):
        '''
        This function requires run and verify
        '''
        prompt = PromptUtils.get_prompt(text)
        output = self.generate(prompt)
        print("\nOutput of generation:")
        print(output)
        print("\nOutput of generation cut_off_text:")
        final_outputs = PromptUtils.cut_off_text(output, '</s>')
        print(final_outputs)
        print("\nOutput of generation remove_substring:")
        final_outputs = PromptUtils.remove_substring(final_outputs, prompt)
        print(final_outputs)
        return final_outputs
            
class HFEmbeddings(EmbeddingsInterface):
    
    def create_embeddings(self, **kwargs):

        model_name = kwargs.get("model_name","sentence-transformers/all-mpnet-base-v2")

        default_param = {
            "model_name": model_name,
            "model_kwargs":{'device': self.device},
            "encode_kwargs":{'normalize_embeddings': False}
        }

        new_kwargs = {"cache_folder": self.cache_dir}
        new_kwargs.update({k: v for k, v in default_param.items() if k not in new_kwargs})

        verbose = kwargs.get("verbose", False)
        if verbose: print(new_kwargs)
            
        embeddings = HuggingFaceEmbeddings(**new_kwargs)

        return embeddings
    
    def create_instruct_embeddings(self, **kwargs):

        model_name = kwargs.get("model_name","hkunlp/instructor-large")

        default_param = {
            "model_name": model_name,
            "model_kwargs":{'device': self.device},
            # "encode_kwargs":{}
        }
        model_name = kwargs.get("model_name", default_param["model_name"])
        model_kwargs = kwargs.get("model_kwargs", default_param["model_kwargs"])
        # encode_kwargs = kwargs.get("encode_kwargs", default_param["encode_kwargs"])

        instruct_embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            # encode_kwargs=encode_kwargs,
            cache_folder=self.cache_dir)
        
        return instruct_embeddings
                
    def get_embeddings(self, **kwargs):

        if kwargs.get("force_update", None) and self.embeddings:
            return self.embeddings
        
        embeddings = self.create_embeddings(**kwargs)
        
        return embeddings        
            
    def get_instruct_embeddings(self,model_name="hkunlp/instructor-large", **kwargs):

        embeddings = self.create_instruct_embeddings(**kwargs)
        
        return embeddings  
                    
if __name__ == "__main__":
    # pip install python-dotenv
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    env_path = Path(__file__).parent.parent / ".env"
    print(env_path)

    load_dotenv(dotenv_path=str(env_path), override=True)  # 加载 .env 文件    
    config = {
        "cache_dir": os.environ['CACHE_DIR'],
        "device": "cuda",
    }
        
    llm_hf = LLMHFxecutor(**config)

    llm_hf.test_generate("What is Python")

    while True:
        print("\n>> (Press Ctrl+C to exit.)")
        inputs = input(">> ")
        chat_complet = llm_hf.run(inputs, verbose=True)        

        import types
        if isinstance(chat_complet, types.GeneratorType):
            for chunk in chat_complet:
                if chunk:
                    print(chunk, end="")
        else:
            print(chat_complet)

    # python -m chingmanlib.llm.models.hf
