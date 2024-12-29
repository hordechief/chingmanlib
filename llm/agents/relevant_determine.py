import numpy as np
import os
# from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.utils.math import cosine_similarity
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.schema import AIMessage

from chingmanlib.llm.models import create_llm
from chingmanlib.llm.db import create_data_pipeline
from chingmanlib.llm.models.hf import HFEmbeddings
from chingmanlib.llm import load_envs
from chingmanlib.llm.utils.log import LOG

# https://python.langchain.com/docs/how_to/routing/

load_envs()

# Initialize LLM
llm = Ollama(
    # model="llama3.1", 
    model="EntropyYue/chatglm3:6b",
    base_url="http://127.0.0.1:11434")

llm = ChatOllama(
    model="llama3.2",
    # model="EntropyYue/chatglm3:6b", # this model doesn't support tool calling
    base_url="http://127.0.0.1:11434")

# llm_executor = create_llm("hf",os.environ["CACHE_DIR"],model_name="THUDM/chatglm-6b")
# llm_executor = create_llm("hf",os.environ["CACHE_DIR"])
# llm_executor = create_llm("openai")
# llm_executor = create_llm("ollama",os.environ["CACHE_DIR"])
# llm = llm_executor.llm

# llm = OpenAI(temperature=0)
# llm = ChatOpenAI(temperature=0)


embeddings = OllamaEmbeddings(
    # model="llama3.2",
    model="EntropyYue/chatglm3:6b",
    # model="llama3",
    base_url='http://127.0.0.1:11434'
)

embeddings = HFEmbeddings(model_name=os.getenv("HF_EMBEDDINGS","sentence-transformers/all-mpnet-base-v2")).create_embeddings()

# Chroma document store intialization
# embedding_function = OpenAIEmbeddings()

# vectorstore = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings, 
#     persist_directory="/root/.cache/my_db")

vector_store_kwargs = {
    "search_type": "similarity", 
    "search_kwargs": {"k": 3}
}

split_kwargs = {
    "chunk_size": 500,
    "chunk_overlap": 100,
}

kwargs = {
    "vector_store_kwargs": vector_store_kwargs,
    "split_kwargs": split_kwargs,
    # "use_exist_db": True
}

kb_data_dir = os.getenv("KB_FOLDER")
assert kb_data_dir 
vsu = create_data_pipeline(kb_data_dir, embeddings, **kwargs)
retriever = vsu.retriever

# judge the relevance
def check_relevance_cus(query, k=3, verbose=False):
    """ check the relevance between the query and retrieved documents"""
    # 1. retrieve document
    retrieved_docs = vsu.similarity_search(query, k=k)
    
    # 2. calculate cosine similarity between query and retrieved document
    similarities = vsu.calculate_cosine_similarity(embeddings, query, retrieved_docs)
    print(f"similarities is {similarities}")
    similarity = similarities.argmax() 
    # 3. determine similarity, relevant or irelement
    threshold = 0.4  # threshold
    # for similarity in similarities:
    #     if similarity >= threshold:
    #         return "relevant"
    
    # return "irelevant"
    if similarity > threshold:
        return "relevant"
    else:
        return "irelevant"

def calculate_cosine_similarities_cus(query):
    """calculate the cosine similarity between query and documents"""

    LOG.log("calculate_cosine_similarities_cus is called")

    # if this function is encapsulated by RunnableLambda, query will be converted to dict {"input": }
    if isinstance(query,dict):
        query = query["input"]

    LOG.append(f"Query of calculate_cosine_similarities is: {query}")

    return check_relevance_cus(query, k=3, verbose=True)

def check_relevance(query, k=3, verbose=False):
    """ check the relevance between the query and retrieved documents"""
    retrieved_docs = vsu.similarity_search_with_score(query, k=k)
    for doc in retrieved_docs:
        threshold = 0.4  # threshold
        if verbose: print(f"similarity score is: {doc[1]}")
        if doc[1] <= threshold:
            return "relevant"
        
    return "irelevant"

def calculate_cosine_similarities(query):
    """calculate the cosine similarity between query and documents"""

    LOG.log("calculate_cosine_similarities is called")

    # if this function is encapsulated by RunnableLambda, query will be converted to dict {"input": }
    if isinstance(query,dict):
        query = query["input"]

    LOG.append(f"Query of calculate_cosine_similarities is: {query}")

    return check_relevance(query, k=3, verbose=True)

# create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        # ("placeholder", "{chat_history}"),
        ("human", "Determine if the provided query is relevant to the retrieved documents. Respond with 'relevant' or 'irelevant'. The query is: {input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

import textwrap
template = textwrap.dedent('''
            Determine if the provided query is relevant to the retrieved documents. Respond with 'relevant' or 'irelevant'. 

            chat_history: {chat_history}

            The query is: {input}

            agent_scratchpad: {agent_scratchpad}
            ''')

template = textwrap.dedent('''
            Determine if the provided query is relevant to the retrieved documents. 
            Calculte similarity between query and a list of document and respond with 'relevant' or 'irelevant'. 
            You can only answer "relevant" or "irelevant", no other answer allowd.

            The query is: {input}
                                    
            agent_scratchpad: {agent_scratchpad}
            ''')

template = textwrap.dedent("""
            Given the user input below, calculte similarity between input query and a list of document, classify it as either being about `relevant` or `irelevant`.

            Do not respond with more than one word.

            <input>
            {input}
            </input>

            agent_scratchpad: {agent_scratchpad}

            Classification:""")

prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    # optional_variables=['chat_history'],
    template=template)

has_memory = True
from chingmanlib.llm.chains.lcec import ConversationChain
from chingmanlib.llm.chains.lcec import ConversationRetrievalQAChain
conversation_chain_util = ConversationChain(llm=llm, retriever=retriever)
conversation_chain = conversation_chain_util.create_chain()
conversation_rag_chain_util = ConversationRetrievalQAChain(llm=llm, retriever=retriever)
conversation_rag_chain = conversation_rag_chain_util.create_chain(has_memory=has_memory)

def route(info):
    '''
    another option is to use RunnableBranch: https://python.langchain.com/docs/how_to/routing/
    '''
    LOG.log("Route input is: ", info)
    if isinstance(info["topic"], str): # function encapsulated by RunnableLambda, return is str
        answer = info["topic"]
    else: # by agent, the return is dict
        answer = info["topic"]['output']

    if "irelevant" == answer.lower():
        LOG.append("conversation_chain is called")
        return conversation_chain
    elif "relevant" == answer.lower():
        LOG.append("conversation_rag_chain is called")
        return conversation_rag_chain
    else:
        LOG.append(f"no chain is called as the returned value is {answer}, use default chain instead")        
        # raise KeyError(f"{answer} - relevance to db is not provided")
        return conversation_chain
    
USE_AGENT_TO_CALCULATE_RELEVANCE=True
if USE_AGENT_TO_CALCULATE_RELEVANCE:
    # Define tools
    cosine_similarities_tool_cus = Tool(
        name="CosineSimilarityTool",
        func=calculate_cosine_similarities_cus,
        description="This tool calculates the cosine similarity between a query and a list of document texts."
    )

    cosine_similarities_tool = Tool(
        name="CosineSimilarityTool",
        func=calculate_cosine_similarities,
        description="This tool calculates the cosine similarity between a query and a list of document texts."
    )

    # Define AgentExecutor
    tools = [cosine_similarities_tool_cus]

    # Define agent
    # llm = llm.bind_tools(tools) # 在 create_tool_calling_agent 已经接受工具并处理绑定的情况下，你不需要再使用 llm.bind_tools(tools)。代理已经处理了工具的绑定问题，因此你可以直接使用 create_tool_calling_agent 创建代理。

    # https://api.python.langchain.com/en/latest/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html
    agent = create_tool_calling_agent(llm, tools, prompt) # This function requires a .bind_tools method be implemented on the LLM. ChatOpenAI (OK) OpenAI (KO)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        # llm=llm,
        # return_intermediate_steps=True,
        verbose=True,
    )

    full_chain = {
            "topic": agent_executor,  # for route
            "question": lambda x: x["input"],  # for converstion rag
            "input": lambda x: x["input"] # for conversation
            } | RunnableLambda(route)

else:
    # encapsulate calculate_cosine_similarities to RunnableLambda
    calculate_cosine_similarities_runnable = RunnableLambda(calculate_cosine_similarities)

    full_chain = {
            "topic": calculate_cosine_similarities_runnable,  # for route
            "question": lambda x: x["input"],  # for converstion rag
            "input": lambda x: x["input"] # for conversation
            } | RunnableLambda(route)

if __name__ == "__main__":
    while True:
        print("\n>> (Press Ctrl+C to exit.)")
        query = input(">> ")
        # query = "what is the camera specifications of Precision 5470"
        
        if 0:
            result = check_relevance_cus(query, k=3)
            print(result)  
            result = check_relevance(query, k=1, verbose=True)
            print(result)  

            # call AgentExecutor to execute complex tasks
            response = agent.invoke({"input": query})
            print(response)

            result_agent = agent_executor.invoke({"input": query})
            print(result_agent)

        result_full = full_chain.invoke({"input": query}, verbose=True)
        LOG.log("Chain Invoke Result is: ", result_full)
        if isinstance(result_full, dict): # answer + docs from ConversationQAChain
            if isinstance(result_full["answer"], AIMessage): # OpenAI model
                response = result_full["answer"].content
            else:
                response = result_full["answer"]
        else:
            response = result_full

        LOG.log("Chain Invoke Answer is:", response)

        if has_memory:
            conversation_chain_util.memory.save_context({"question": query}, {"answer": response})
            conversation_rag_chain_util.memory.save_context({"question": query}, {"answer": response})
            varaibles = conversation_rag_chain_util.memory.load_memory_variables({})
            LOG.log("Memory Varaibles is :")
            for var in varaibles["history"]:
                LOG.append(f"\n{type(var).__name__}: ", var.content)

    
    # python -m llm.agents.relevant_determine 

# react 实时性
# rag的相关性，以及对chat的影响
# 工具的可靠性