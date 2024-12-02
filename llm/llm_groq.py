import os, getpass

from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain_groq import ChatGroq
from groq import Groq

# GROQ_API_KEY = getpass.getpass()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
llm = Groq(api_key=GROQ_API_KEY)
# llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY, temperature=0)

def test_base_llm(llm):
    # test base llm mode
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
    
def test_groq(llm):
#     https://console.groq.com/docs/quickstart
    import os
    from groq import Groq

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
if __name__ == "__main__":
    test_groq(llm)