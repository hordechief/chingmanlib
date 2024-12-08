# https://console.groq.com/docs/quickstart

import os, getpass

from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from groq import Groq
from langchain_groq import ChatGroq

from .llm_interface import LLMInterface

# GROQ_API_KEY = getpass.getpass()
# llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY, temperature=0)

class LLMGroqxecutor(LLMInterface):
    
    def __init__(self, **config):              
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        llm = Groq(api_key=GROQ_API_KEY)
        model="llama3-70b-8192"
        self.client = Groq(model=model, api_key=GROQ_API_KEY)  
    
def test_groq(llm):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain the importance of fast language models",
            }
        ],
        model="llama3-8b-8192",
    )

    print(chat_completion.choices[0].message.content)    

def test_chat_groq(llm):
    if isinstance(llm,ChatGroq) or isinstance(llm,ChatOpenAI):
        messages = [
            SystemMessage(content="Say the opposite of what the user says"),
            HumanMessage(content="I love programming."),
            AIMessage(content='I hate programming.'),
            HumanMessage(content="The moon is out")
        ]
        response = llm(messages)
    else:
        prompt = """
        Question: Translate this sentence from English to French. I love programming.
        """            
        response = llm(prompt)
    print(type(response))
    print(response)

if __name__ == "__main__":
    llm = LLMGroqxecutor().client
    test_groq(llm)