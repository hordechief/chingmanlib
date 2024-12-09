import os, sys

# https://github.com/openai/openai-quickstart-python
# !pip install --upgrade openai
# !pip install -q pip langchain-openai
# https://python.langchain.com/v0.2/docs/integrations/platforms/openai/
import openai
from openai import OpenAI
from openai import AzureOpenAI

# from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings

import langchain_openai
from langchain_openai import AzureChatOpenAI
from langchain_openai import OpenAI as LangchainOpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI  

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from dotenv import load_dotenv
from getpass import getpass

from .llm_interface import LLMInterface


# import your OpenAI key -
# you need to put it in your .env file 
# OPENAI_API_KEY='sk-xxxx'

# with open('.env','r') as f:
#     env_file = f.readlines()
# envs_dict = {key.strip("'"):value for key, value in [(i.split('=')) for i in env_file]}
# os.environ['OPENAI_API_KEY'] = envs_dict['OPENAI_API_KEY']

from enum import Enum
class AIModelType(Enum):
    OPENAI = 1
    CHAT_OPENAI = 2
    AZURE_OPENAI = 3
    
class OpenAIExecutor(LLMInterface):
    def __init__(self, *args, **kwargs):
        # os.environ["OPENAI_API_KEY"] = getpass("OPENAI_API_KEY")
        
        # os.environ["AZURE_OPENAI_API_KEY"] = input("AZURE_OPENAI_API_KEY")
        # os.environ["AZURE_OPENAI_ENDPOINT"] = input("AZURE_OPENAI_ENDPOINT")      
        # 
        print(f"OpenAIExecutor arguments is : {kwargs}")  

        model_kwargs = kwargs.get("model_kwargs", {})         
        model_kwargs.setdefault("api_key", os.environ["OPENAI_API_KEY"])  
        self.model = self.get_param("model_name", "OPENAI_MODEL", "gpt-3.5-turbo", **kwargs)
        # new_items = {
        #     "model_name": "gpt-35-turbo",
        #     "api_key": os.environ["OPENAI_API_KEY"]
        # }
        # model_kwargs.update({k: v for k, v in new_items.items() if k not in model_kwargs})

        embeddings_kwargs = kwargs.get("embeddings_kwargs", {})
        
        # If you are behind an explicit proxy, you can specify the http_client to pass through
        # pip install httpx
        if kwargs.get("proxies"):
            import httpx            
            # httpx_client = httpx.Client(proxies="http://proxy.yourcompany.com:8080")
            # httpx_client = httpx.Client(verify=False)
            model_kwargs.update({"http_client":httpx.Client(verify=False,proxies=kwargs.get("proxies"))})            
            embeddings_kwargs.update({"http_client":httpx.Client(verify=False,proxies=kwargs.get("proxies"))})
                                
        # initialize model accoring to the type and parameter
        openai_model_type = kwargs.get("openai_model_type", "OpenAI")
        openai_wrapper_service = kwargs.get("openai_wrapper_service", "OpenAI")

        if "OpenAI" == openai_wrapper_service:            
            os.environ["OPENAI_MODEL"] = self.model
            if "OpenAI" == openai_model_type: 
                self.client = OpenAI()
                print("OpenAI created")
            elif "ChatOpenAI" == openai_model_type:  
                self.client = OpenAI() # same as ?
                print("ChatOpenAI created")
            self.embeddings = OpenAIEmbeddings()  
        elif "AzureOpenAI" == openai_wrapper_service:
            os.environ["OPENAI_MODEL"] = self.model
            if "OpenAI" == openai_model_type:
                self.client = AzureOpenAI()   # maybe the parameter need to be updated
                print("ChatOpenAI created for LangChain")
            elif "ChatOpenAI" == openai_model_type:  
                model_kwargs.update(openai_api_version="2023-12-01-preview",azure_deployment="gpt-3.5-turbo")
                self.client = AzureChatOpenAI(**model_kwargs)
            embeddings_kwargs.update(azure_deployment="text-embedding-ada-002",openai_api_version="2023-12-01-preview")
            self.embeddings = AzureOpenAIEmbeddings(**embeddings_kwargs)            
        elif "LangchainOpenAI" == openai_wrapper_service:
            model_kwargs.setdefault("model_name", "gpt-3.5-turbo-instruct")
            if "OpenAI" == openai_model_type:                
                self.client = LangchainOpenAI(**model_kwargs)    
                print("OpenAI created for LangChain")
            elif "ChatOpenAI" == openai_model_type:  
                self.client = ChatOpenAI()
                print("ChatOpenAI created for LangChain")
            self.embeddings = OpenAIEmbeddings()              
        # elif "LammaIndexOpenAI" == openai_model_type:
        #     from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
        #     self.client = LlamaIndexOpenAI(model_name="gpt-3-turbo")       
        else:
            self.client = OpenAI()
            self.embeddings = OpenAIEmbeddings()  
            print("default OpenAI created")

        self.llm = self.client
            
    def run(self,prompt):
        return self.client(prompt)
                
    def get_embeddings(self, **kwargs):
        return self.embeddings
                    
    def test_openai_completion_endpoint(self, is_test_api=False):
        # You tried to access openai.Completion, but this is no longer supported in openai>=1.0.0 
        # - see the README at https://github.com/openai/openai-python for the API.

        if is_test_api:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt="Explain the significance of the Turing test in AI.",
                max_tokens=150,
                temperature=0.6,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=1,
                stop=["\n"]  # 可以设置停止符号
            )
        else:
            assert isinstance(self.client, openai.OpenAI)
            response = self.client.Completion.create(
                model="text-davinci-003",
                prompt="Explain the significance of the Turing test in AI."
            )
            response = self.client("Explain the significance of the Turing test in AI.")

        generated_text = response.choices[0].text.strip()
        print(generated_text)
        
    def test_openai_chat_completion_endpoint(self, is_test_api=False):

        if is_test_api:        
            completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
                ]
            )
        else:
            assert isinstance(self.client, openai.OpenAI)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
                ]
            )            
        print(completion.choices[0].message)
        
    def test_langchain_wrapper_llm(self):

        assert isinstance(self.client, langchain_openai.OpenAI)
        
        # with pytest.raises(Exception) as e_info:
        prompt = """
        Question: Translate this sentence from English to French. I love programming.
        """            
        response = self.client(prompt) 
        print(type(response))
        print(response)
    
    def test_langchain_wrapper_chat_llm(self):

        assert isinstance(self.client, langchain_openai.chat_models.ChatOpenAI), f"client type is {self.client}"
        
        messages = [
            SystemMessage(content="Say the opposite of what the user says"),
            HumanMessage(content="I love programming."),
            AIMessage(content='I hate programming.'),
            HumanMessage(content="The moon is out")
        ]
        response = self.client(messages)

        print(type(response))
        print(response.content)
            
        '''
        <class 'langchain_core.messages.ai.AIMessage'>
        content='The moon is not out.' response_metadata={'token_usage': {'completion_tokens': 6, 'prompt_tokens': 39, 'total_tokens': 45, 'prompt_tokens_details': {'cached_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-8f90da98-2875-45c9-a8c6-b002c40777b5-0'
        '''

if __name__ == "__main__":
    
    # legacy - NOT supported
    openai_executor = OpenAIExecutor(
        openai_model_type="OpenAI", openai_wrapper_service = "OpenAI", 
        model_kwargs={"model_name":"text-davinci-003"})
    openai_executor.test_openai_completion_endpoint(is_test_api=True)
    openai_executor.test_openai_completion_endpoint()

    # Chat Completion
    openai_executor = OpenAIExecutor(
        openai_model_type="ChatOpenAI", openai_wrapper_service = "OpenAI", 
        model_kwargs={"model_name":"gpt-3.5-turbo"})
    openai_executor.test_openai_chat_completion_endpoint(is_test_api=True)
    openai_executor.test_openai_chat_completion_endpoint()

    # LangChain wrapper
    openai_executor = OpenAIExecutor(openai_model_type="OpenAI", openai_wrapper_service = "LangchainOpenAI")
    openai_executor.test_langchain_wrapper_llm()    

    openai_executor = OpenAIExecutor(openai_model_type="ChatOpenAI", openai_wrapper_service = "LangchainOpenAI")
    openai_executor.test_langchain_wrapper_chat_llm()

    sys.exit()
    
    # python -m submodules.chingmanlib.llm.models.openai
    # export PYTHONPATH=/home/aurora/repos/dl-ex/submodules/:$PYTHONPATH
