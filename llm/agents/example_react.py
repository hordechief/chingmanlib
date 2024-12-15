########## react agents

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAI

try:
    import src.settings, src.envs
except: # run as main file
    # import sys,os
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # import settings, envs
    pass

########################################################
#
#             Wiki Explore
#
########################################################      
class ReactWikiExploreExecutor():
    import langchain
    from langchain import Wikipedia
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    from langchain.agents.react.base import DocstoreExplorer

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

    react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)
    
    @classmethod    
    def run(cls, question, debug=False):
        import langchain
        
        if debug:
            langchain.debug = True
            
        print(cls.react.agent.llm_chain.prompt.template)        

        cls.react.run(question)
        
        langchain.debug = False
    

def test_react_wiki_search(llm):
    from langchain import OpenAI, Wikipedia
    from langchain.agents import initialize_agent, Tool, AgentType
    from langchain.agents.react.base import DocstoreExplorer    
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
    
    question = "How old is the president of the United States?"

    react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)  # deprecated    
    react.run(question)
    print(react.agent.llm_chain.prompt.template)
    
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
    
########################################################
#
#             Tavily Search
#
########################################################      
class ReactTavilySearch():
    # https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/ 
    from langchain import hub
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_openai import OpenAI
    
    tools = [TavilySearchResults(max_results=1)]
    
    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")
    
    # Choose the LLM to use
    llm = OpenAI()

    # Construct the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)    
    
    @classmethod    
    def run(cls, question="what is LangChain?", debug=False):
        cls.agent_executor.invoke({"input": question})

def test_react_tavily_search(llm):
    from langchain.tools.tavily_search import TavilySearchResults
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
    from langchain.agents import load_tools, create_openai_functions_agent

    tools = [
      TavilySearchResults(api_wrapper=TavilySearchAPIWrapper())      
    ]
    tools.extend(load_tools(['wikipedia']))
    
    # llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke({"input": "Who is the owner of Tesla company? Let me know details about owner."})
    
'''
Invoking: `tavily_search_results_json` with `{'query': 'Who is the owner of Tesla company'}`
'''

# Page & Summary
'''
Page: Family of Elon Musk
Summary: Elon Musk's family consists  of several notable individuals, among them his mother Maye Musk, a model and author, his father Errol Musk, a businessman and politician, his siblings Kimbal Musk  and Tosca Musk, and his cousin Lyndon Rive. His ex-wives are Justine Musk and Tallauh Riley.
'''
    
########################################################
#
#             Chain of Thought
#
########################################################    
def test_cot():
    # https://stackoverflow.com/questions/77789886/openai-api-error-the-model-text-davinci-003-has-been-deprecated
    llm = OpenAI(model_name='gpt-3.5-turbo-instruct',
             temperature=0,
             max_tokens = 256)
             
    text = "Explain step by step. How old is the president of the United States?"
    # print(llm(text))
    
    '''
    Step 1: Determine the current year
    The first step in determining the age of the president of the United States is to determine the current year. As of 2021, the current year is 2021.

    Step 2: Identify the president's birth year
    Next, we need to identify the birth year of the current president. As of 2021, the current president is Joe Biden, who was born on November 20, 1942.

    Step 3: Calculate the president's age
    To calculate the president's age, we subtract the birth year from the current year. In this case, 2021 - 1942 = 79. This means that as of 2021, Joe Biden is 79 years old.

    Step 4: Consider the president's birthday
    It's important to note that the president's age may change throughout the year depending on their birthday. In this case, Joe Biden's birthday is in November, so he will turn 80 in 2022.

    Step 5: Verify the information
    To verify the age of the president, we can check reliable sources such as government websites or news articles.

    Step 6: Repeat the process for previous presidents
    To determine the age of previous presidents, we
    '''
    
########################################################
#
#             Few Shots for React
#
########################################################
def test_few_shot():  
    llm = OpenAI(model_name='gpt-3.5-turbo-instruct',
             temperature=0,
             max_tokens = 256)
             
    question = "How old is the president of the United States?"

    manual_react = f"""
    Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
    Thought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
    Action: Search[Colorado orogeny]
    Observation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
    Thought: It does not mention the eastern sector. So I need to look up eastern sector.
    Action: Lookup[eastern sector]
    Observation: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
    Thought: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
    Action: Search[High Plains]
    Observation: High Plains refers to one of two distinct land regions
    Thought: I need to instead search High Plains (United States).
    Action: Search[High Plains (United States)]
    Observation: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
    Thought: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
    Action: Finish[1,800 to 7,000 ft]

    Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
    Thought: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
    Action: Search[Milhouse]
    Observation: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
    Thought: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
    Action: Lookup[named after]
    Observation: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
    Thought: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
    Action: Finish[Richard Nixon]

    Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
    Thought: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
    Action: Search[Adam Clayton Powell]
    Observation: Could not find [Adam Clayton Powell]. Similar: [’Adam Clayton Powell III’, ’Seventh Avenue (Manhattan)’, ’Adam Clayton Powell Jr. State Office Building’, ’Isabel Washington Powell’, ’Adam Powell’, ’Adam Clayton Powell (film)’, ’Giancarlo Esposito’].
    Thought: To find the documentary, I can search Adam Clayton Powell (film).
    Action: Search[Adam Clayton Powell (film)]
    Observation: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg. The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
    Thought: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
    Action: Finish[The Saimaa Gesture]

    Question: What profession does Nicholas Ray and Elia Kazan have in common?
    Thought: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
    Action: Search[Nicholas Ray]
    Observation: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
    Thought: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
    Action: Search[Elia Kazan]
    Observation: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
    Thought: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
    Action: Finish[director, screenwriter, actor]

    Question: Which magazine was started first Arthur’s Magazine or First for Women?
    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started first.
    Action: Search[Arthur’s Magazine]
    Observation: Arthur’s Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women next.
    Action: Search[First for Women]
    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) < 1989 (First for Women), so Arthur’s Magazine was started first.
    Action: Finish[Arthur’s Magazine]

    Question:{question}"""
    
    print(llm(manual_react))
    
if __name__ == "__main__":
    react_wiki_explore = ReactWikiExploreExecutor()    
    question = "What is the first film that Russell Crowe won an Oscar for, and who directed that movie?"
    #react_wiki_explore.run(question)
    
    react_tavily_search = ReactTavilySearch()
    question="what is LangChain?"
    react_tavily_search.run(question)
