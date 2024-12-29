########## react agents
import langchain
from langchain import hub
# from langchain import OpenAI
# from langchain import Wikipedia # deprecated
from langchain_community.docstore.wikipedia import Wikipedia
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import (
    initialize_agent, 
    Tool, 
    AgentType,
    # load_tools,  # deprecated
    create_openai_functions_agent
)
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents.react.base import DocstoreExplorer

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain.tools.tavily_search import TavilySearchResults

import os
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
assert os.path.exists(env_path)
load_dotenv(dotenv_path=str(env_path), override=True)  

########################################################
#
#             Wiki Explore
#
########################################################      
class ReactWikiExploreExecutor():
    # pip -q install langchain langchain_community huggingface_hub openai google-search-results tiktoken wikipedia

    docstore=DocstoreExplorer(Wikipedia())

    tools = [
        Tool(
            name="Search",
            func=docstore.search,
            description="useful for when you need to ask with search"
        ),
        Tool(
            name="Lookup",
            func=docstore.lookup,
            description="useful for when you need to ask with lookup"
        )
    ]

    llm = OpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo-instruct' #"text-davinci-003"
        )

    react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)  # deprecated    
    
    @classmethod    
    def run(cls, question, verbose=False):
        
        if verbose:
            langchain.debug = True
            
        if verbose: print(cls.react.agent.llm_chain.prompt.template)        

        result = cls.react.run(question)
        
        langchain.debug = False

        if verbose: print(result)

        return result
        
    
########################################################
#
#             Tavily Search
#
########################################################      
class ReactTavilySearch():
    # https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/ 

    tools = [TavilySearchResults(max_results=1)]
    
    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")
    # input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'] 
    # input_types={} partial_variables={} 
    # metadata={'lc_hub_owner': 'hwchase17', 'lc_hub_repo': 'react', 'lc_hub_commit_hash': 'd15fe3c426f1c4b3f37c9198853e4a86e20c425ca7f4752ec0c9b0e97ca7ea4d'} 
    # template='Answer the following questions as best you can. 
    # You have access to the following tools:\n\n{tools}\n\n
    # Use the following format:\n\n
    # Question: the input question you must answer\n
    # Thought: you should always think about what to do\n
    # Action: the action to take, should be one of [{tool_names}]\n
    # Action Input: the input to the action\n
    # Observation: the result of the action\n
    # ... (this Thought/Action/Action Input/Observation can repeat N times)\n
    # Thought: I now know the final answer\n
    # Final Answer: the final answer to the original input question\n\n
    # Begin!\n\nQuestion: {input}\n
    # Thought:{agent_scratchpad}'

    # Choose the LLM to use
    llm = OpenAI()

    # Construct the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)   

    @classmethod
    def run(cls, question="what is LangChain?", verbose=False):
        result = cls.agent_executor.invoke({"input": question})
        if verbose: print(result)
        return result

class TavilySearch():
    tools = [
        TavilySearchResults(api_wrapper=TavilySearchAPIWrapper())      
    ]
    tools.extend(load_tools(['wikipedia']))
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    
    prompt = hub.pull("hwchase17/openai-functions-agent")
    # https://smith.langchain.com/hub/hwchase17/openai-functions-agent

    '''
    system: You are a helpful assistant
    PLACEHOLDER: chat_history
    HUMAN: {input}
    PLACEHOLDER: agent_scratchpad
    '''

    '''
    input_variables=['agent_scratchpad', 'input'] 
    optional_variables=['chat_history'] 
    input_types={'chat_history': list[
    typing.Annotated[typing.Union[typing.Annotated[langchain_core.messages.ai.AIMessage, Tag(tag='ai')], 
    typing.Annotated[langchain_core.messages.human.HumanMessage, Tag(tag='human')], 
    typing.Annotated[langchain_core.messages.chat.ChatMessage, Tag(tag='chat')], 
    typing.Annotated[langchain_core.messages.system.SystemMessage, Tag(tag='system')], 
    typing.Annotated[langchain_core.messages.function.FunctionMessage, Tag(tag='function')], 
    typing.Annotated[langchain_core.messages.tool.ToolMessage, Tag(tag='tool')], 
    typing.Annotated[langchain_core.messages.ai.AIMessageChunk, Tag(tag='AIMessageChunk')], 
    typing.Annotated[langchain_core.messages.human.HumanMessageChunk, Tag(tag='HumanMessageChunk')], 
    typing.Annotated[langchain_core.messages.chat.ChatMessageChunk, Tag(tag='ChatMessageChunk')], 
    typing.Annotated[langchain_core.messages.system.SystemMessageChunk, Tag(tag='SystemMessageChunk')], 
    typing.Annotated[langchain_core.messages.function.FunctionMessageChunk, Tag(tag='FunctionMessageChunk')], 
    typing.Annotated[langchain_core.messages.tool.ToolMessageChunk, Tag(tag='ToolMessageChunk')]], 
    FieldInfo(annotation=NoneType, required=True, discriminator=Discriminator(discriminator=<function _get_type at 0x7781d5155090>, custom_error_type=None, custom_error_message=None, custom_error_context=None))]], 
    'agent_scratchpad': list[
    typing.Annotated[typing.Union[typing.Annotated[langchain_core.messages.ai.AIMessage, Tag(tag='ai')], 
    typing.Annotated[langchain_core.messages.human.HumanMessage, Tag(tag='human')], 
    typing.Annotated[langchain_core.messages.chat.ChatMessage, Tag(tag='chat')], 
    typing.Annotated[langchain_core.messages.system.SystemMessage, Tag(tag='system')], 
    typing.Annotated[langchain_core.messages.function.FunctionMessage, Tag(tag='function')], 
    typing.Annotated[langchain_core.messages.tool.ToolMessage, Tag(tag='tool')], 
    typing.Annotated[langchain_core.messages.ai.AIMessageChunk, Tag(tag='AIMessageChunk')], 
    typing.Annotated[langchain_core.messages.human.HumanMessageChunk, Tag(tag='HumanMessageChunk')], 
    typing.Annotated[langchain_core.messages.chat.ChatMessageChunk, Tag(tag='ChatMessageChunk')], 
    typing.Annotated[langchain_core.messages.system.SystemMessageChunk, Tag(tag='SystemMessageChunk')], 
    typing.Annotated[langchain_core.messages.function.FunctionMessageChunk, Tag(tag='FunctionMessageChunk')], 
    typing.Annotated[langchain_core.messages.tool.ToolMessageChunk, Tag(tag='ToolMessageChunk')]], 
    FieldInfo(annotation=NoneType, required=True, discriminator=Discriminator(discriminator=<function _get_type at 0x7781d5155090>, custom_error_type=None, custom_error_message=None, custom_error_context=None))]]} partial_variables={'chat_history': []} metadata={'lc_hub_owner': 'hwchase17', 'lc_hub_repo': 'openai-functions-agent', 'lc_hub_commit_hash': 'a1655024b06afbd95d17449f21316291e0726f13dcfaf990cc0d18087ad689a5'} 
    messages=[
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='You are a helpful assistant'), additional_kwargs={}), 
    MessagesPlaceholder(variable_name='chat_history', optional=True), 
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], input_types={}, partial_variables={}, template='{input}'), additional_kwargs={}), 
    MessagesPlaceholder(variable_name='agent_scratchpad')]
    '''
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    @classmethod
    def run(cls, question, verbose=False):
        result = cls.agent_executor.invoke({"input": question})
        if verbose: print(result)
        return result
    
if __name__ == "__main__":

    CASES = ["", "","TavilySearch"] # ReactWikiExplore, ReactTavilySearch, TavilySearch

    if "ReactWikiExplore" in CASES:
        react_wiki_explore = ReactWikiExploreExecutor()    
        question = "What is the first film that Russell Crowe won an Oscar for, and who directed that movie?"
        react_wiki_explore.run(question)

        question = "How old is the president of the United States?"
        react_wiki_explore.run(question)
        
        '''
        Thought: I need to search for the current president of the United States and find their age.
        Action: Search[current president of the United States]
        Observation:
        Thought: Joe Biden is the current president of the United States. I need to find his age.
        Action: Search[Joe Biden age]
        Observation:
        Thought: Joe Biden is 81 years old. So the answer is 81 years old.
        Action: Finish[81 years old]
        '''

    if "ReactTavilySearch" in CASES:
        react_tavily_search = ReactTavilySearch()

        question="what is LangChain?"
        react_tavily_search.run(question, verbose=True)

        question="Who is the owner of Tesla company? Let me know details about owner."
        react_tavily_search.run(question)
        '''
        Invoking: `tavily_search_results_json` with `{'query': 'Who is the owner of Tesla company'}`
        '''

        # Page & Summary
        '''
        Page: Family of Elon Musk
        Summary: Elon Musk's family consists  of several notable individuals, among them his mother Maye Musk, a model and author, his father Errol Musk, a businessman and politician, his siblings Kimbal Musk  and Tosca Musk, and his cousin Lyndon Rive. His ex-wives are Justine Musk and Tallauh Riley.
        '''

    if "TavilySearch" in CASES:
        tavily_search = TavilySearch()
        question="what is LangChain?"
        tavily_search.run(question)

f