import os, sys

# https://github.com/openai/openai-quickstart-python
# !pip install --upgrade openai
# !pip install -q pip langchain-openai
# https://python.langchain.com/v0.2/docs/integrations/platforms/openai/
import openai
from openai import OpenAI, AzureOpenAI
from langchain_openai.chat_models import ChatOpenAI  
from langchain_openai import AzureChatOpenAI, OpenAI as LangchainOpenAI, AzureOpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI  
    # ...
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
            
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from dotenv import load_dotenv
from getpass import getpass

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
    
class OpenAIExecutor():
    def __init__(self, *args, **kwargs):
        # os.environ["OPENAI_API_KEY"] = getpass("OPENAI_API_KEY")
        
        # os.environ["AZURE_OPENAI_API_KEY"] = input("AZURE_OPENAI_API_KEY")
        # os.environ["AZURE_OPENAI_ENDPOINT"] = input("AZURE_OPENAI_ENDPOINT")        
        
        model_kwargs = {}
        
        embeddings_kwargs = {}
        
        # If you are behind an explicit proxy, you can specify the http_client to pass through
        # pip install httpx
        if kwargs.get("proxies"):
            import httpx            
            # httpx_client = httpx.Client(proxies="http://proxy.yourcompany.com:8080")
            # httpx_client = httpx.Client(verify=False)
            model_kwargs.update({"http_client":httpx.Client(verify=False,proxies=kwargs.get("proxies"))})            
            embeddings_kwargs.update({"http_client":httpx.Client(verify=False,proxies=kwargs.get("proxies"))})
                                
        # initialize model accoring to the type and parameter
        openai_model_type = kwargs.get("openai_model_type", None)
        
        if "OpenAI" == openai_model_type:
            model_kwargs.update(api_key=os.environ["OPENAI_API_KEY"])            
            self.client = OpenAI(**model_kwargs)            
            self.embeddings = OpenAIEmbeddings()            
        elif "ChatOpenAI" == openai_model_type:        
            self.client = ChatOpenAI()            
        elif "AzureOpenAI" == openai_model_type:
            self.client = AzureOpenAI()            
            embeddings_kwargs.update(azure_deployment="text-embedding-ada-002",openai_api_version="2023-12-01-preview")
            self.embeddings = AzureOpenAIEmbeddings(**embeddings_kwargs)
        elif "AzureChatOpenAI" == openai_model_type:
            model_kwargs.update(openai_api_version="2023-12-01-preview",azure_deployment="gpt-35-turbo")
            self.client = AzureChatOpenAI(**model_kwargs)
        elif "LangchainOpenAI" == openai_model_type:
            model_kwargs.update(model_name="gpt-3.5-turbo-instruct")
            self.client = LangchainOpenAI(model_kwargs)
        elif "LammaIndexOpenAI" == openai_model_type:
            self.client = LlamaIndexOpenAI(model_name="gpt-3-turbo")
        elif "Groq" == openai_model_type:
            self.client = Groq(model="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])         
        else:
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            
    def test_llm_completion(self, llm=None):
        assert isinstance(self.client,openai.OpenAI)
        
        completion = self.client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
          ]
        )
        print(completion.choices[0].message)
        
    def test_base_llm(self, llm=None):
        # with pytest.raises(Exception) as e_info:
        prompt = """
        Question: Translate this sentence from English to French. I love programming.
        """            
        response = self.client(prompt) 
        print(type(response))
        print(response)
        
    def test_chat_llm(self,llm=None):
        llm = llm or self.client
        
#         assert isinstance(llm,ChatGroq) or isinstance(llm,ChatOpenAI)
        assert isinstance(llm,ChatOpenAI)
        
        messages = [
            SystemMessage(content="Say the opposite of what the user says"),
            HumanMessage(content="I love programming."),
            AIMessage(content='I hate programming.'),
            HumanMessage(content="The moon is out")
        ]
        response = llm(messages)

        print(type(response))
        print(response.content)
'''
<class 'langchain_core.messages.ai.AIMessage'>
content='The moon is not out.' response_metadata={'token_usage': {'completion_tokens': 6, 'prompt_tokens': 39, 'total_tokens': 45, 'prompt_tokens_details': {'cached_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-8f90da98-2875-45c9-a8c6-b002c40777b5-0'
'''
    
if __name__ == "__main__":
    openai_executor = OpenAIExecutor(openai_model_type="OpenAI")
    print(type(openai_executor.client))
    openai_executor.test_base_llm()
#     openai_executor.test_llm_completion()

#     openai_executor = OpenAIExecutor(openai_model_type="ChatOpenAI")
#     print(type(openai_executor.client))
#     openai_executor.test_chat_llm()
#     sys.exit()
    