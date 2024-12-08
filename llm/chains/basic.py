from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableParallel,RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string

from operator import itemgetter # 从上面的变量中提取

from chingmanlib.llm.prompts.templates import CHATBOT_RAG_QA_TEMPLATE

class BasicChain():
    def __init__(self, llm):
        self.llm = llm

    def check_input_variables(self,template,input_variables):
        '''
        check whether the input variable are in template
        '''
        for variable in input_variables:
            _variable = "{" + variable + "}"
            # print(_variable)
            if not _variable in template:
                print(f"variable {_variable} is not included in template")
                return False
        return True
    
    def create_prompt(self, template, default_template=None, input_variables=None):
        if not template:
            assert default_template
            template = default_template["template"]
            input_variables = default_template["input_variables"]
            prompt = PromptTemplate(template=template, input_variables=input_variables)
        else:
            prompt = PromptTemplate.from_template(template) 
            if input_variables and not self.check_input_variables(template,input_variables):
                raise ValueError("input variable is not consistant")
            prompt = PromptTemplate.from_template(template) 
                        
        return prompt
        
    def create_llm_chain(self, template=None):
        if not template:
            template = """Question: {question}
            Answer: Let's think step by step."""
        
        prompt = PromptTemplate.from_template(template)
        self.llm_chain = prompt | self.llm
        
        return self.llm_chain    
    
    def create_llm_rag_chain(self, rag_template=None, input_variables=None, retriever=None):
        if not retriever:
            retriever = self.retriever

        assert retriever
            
        rag_prompt = self.create_prompt(rag_template, input_variables=input_variables, default_template=CHATBOT_RAG_QA_TEMPLATE)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt 
            | self.llm
            | StrOutputParser()
        )
        
        print(type(rag_chain))
                
        return rag_chain    
    
    def create_llm_conversationn_QA_chain(self,
                question_prompt=None,
                answer_prompt=None,
                retriever=None,
                has_memory=False):
        '''
        language in template is not mandatory, just give an example to add additional paramter
        '''
        print("create_llm_conversationn_QA_chain")

        self.has_memory = has_memory

        # use default retriever
        if not retriever:
            retriever = self.retriever
            
        assert retriever

        if not question_prompt:
            question_prompt = """
            Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

            Chat History:
            {chat_history}

            Follow Up Input: {question}

            Standalone question
            """
            
        if not answer_prompt:
            answer_prompt = """
            Answer the question based only on the following context:
            {context}

            Question: {question}
            """            
        # question prompt 
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(question_prompt)
        print(f"CONDENSE_QUESTION_PROMPT is : \n\n {CONDENSE_QUESTION_PROMPT}")
        
        # answer prompt
        ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_prompt)
        print(f"ANSWER_PROMPT is : \n\n {ANSWER_PROMPT}")
        
        # document prompt
        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}") # page_content 应该是Document的变量
        print(f"DEFAULT_DOCUMENT_PROMPT is : \n\n {DEFAULT_DOCUMENT_PROMPT}")

        def _combine_documents(
            docs, 
            document_prompt=DEFAULT_DOCUMENT_PROMPT, 
            document_separator="\n\n"
        ):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)
        
        # Now we calculate the standalone question        
        if not has_memory:   
            _inputs = RunnableParallel(
                # RunnablePassthrough 会把language变量带进去，chat_history是额外分配的
                # RunnablePassthrough() 是完整参数列表，通过lambda取指定参数
                # language = lambda x: x["language"], 
                standalone_question = RunnablePassthrough.assign(
                    chat_history=lambda x: get_buffer_string(x["chat_history"]),
                )
                | CONDENSE_QUESTION_PROMPT
                | self.llm
                | StrOutputParser(),                
            )
                                
            _context = {
                "context": itemgetter("standalone_question") | retriever | _combine_documents,
                "question": lambda x: x["standalone_question"], # answer prompt里的question跟question prompt里是不一样的
                # only RunnablePassthrough work for language
                "language": RunnablePassthrough(),
                # "language": lambda x: x["language"], # _input中提供的变量，否则这儿无法获取 ####
                # "language": itemgetter("language")
            }  
                                    
            self.conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | self.llm 
            
            # Notes:
            # itemgetter("standalone_question") -> itemgetter("standalone_question")(_inputs)
            # lambda x: x["standalone_question"] -> lambda x: x["standalone_question"](_inputs)
            
        else:
            memory = ConversationBufferMemory(
                return_messages=True, output_key="answer", input_key="question"
            )
            # First we add a step to load memory
            # This adds a "memory" key to the input object
            loaded_memory = RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
            )

            self.memory = memory
            
            standalone_question = {
                "standalone_question": {
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: get_buffer_string(x["chat_history"]),
                    # ALL OK for language
                    "language": RunnablePassthrough(), 
                    # "language": itemgetter("language"), 
                    # "language": lambda x: x["language"],
                }
                | CONDENSE_QUESTION_PROMPT
                | self.llm
                | StrOutputParser(),
            }
            # "standalone_question" is the input of next step
            
            # Now we retrieve the documents
            retrieved_documents = {
                "docs": itemgetter("standalone_question") | retriever,
                "question": lambda x: x["standalone_question"],
            }
            # "docs", "question" are the input for next step

            # Now we construct the inputs for the final prompt
            final_inputs = {
                "context": lambda x: _combine_documents(x["docs"]),
                "question": itemgetter("question"),
                # only RunnablePassthrough work for language
                "language": RunnablePassthrough(),           
                # "language": itemgetter("language"), # KO
            }

            # And finally, we do the part that returns the answers
            answer = {
                "answer": final_inputs | ANSWER_PROMPT | self.llm,
                "docs": itemgetter("docs"),
            }        

            # And now we put it all together!
            self.conversational_qa_chain = loaded_memory | standalone_question | retrieved_documents | answer
                
        return self.conversational_qa_chain    
    
    def run(self, question):
        from langchain_core.messages import AIMessage

        inputs = {"question": question}
        result = self.conversational_qa_chain.invoke(inputs)
        if isinstance(result["answer"], AIMessage):
            response = result["answer"].content
        else:
            response = result["answer"]

        if self.has_memory:
            # Note that the memory does not save automatically
            # This will be improved in the future
            # For now you need to save it yourself
            self.memory.save_context(inputs, {"answer": response})
            self.memory.load_memory_variables({})

        print(response)

    def test_llm_chain(self):
        prompt_value = self.llm_chain.invoke({"question": "ice cream"})
        print(prompt_value)

    def test_llm_conversationn_QA_chain(self):
        response = self.conversational_qa_chain.invoke(
            {
                "question": "What's the key component of agent system?",
                "chat_history": [],
                "language":"中文"
            }
        )
        print(response)

    def test_llm_conversationn_QA_chain_with_memory(self):
        response = self.run("What's the key component of agent system?")
        print(response)  

def prepare_llm_conversation_QA_chain(llm_executor):
    
    embeddings = llm_executor.get_embeddings()

    # Define template
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}

    Follow Up Input: {question}

    Answer in the following language: {language}

    Standalone question
    """

    template = """
    Answer the question based only on the following context using Chinese:
    {context}

    Answer in the following language: {language}

    Question: {question}
    """ 

    # Load Data
    from chingmanlib.llm.pipeline import DataLoaderUtils
    dataloader_util = DataLoaderUtils()
    file_path = "/home/aurora/repos/dl-ex/llm/docs/The_Guide_To_LLM_Evals.pdf"
    data = dataloader_util.pdf_loader(file_path)

    # Split
    from chingmanlib.llm.pipeline import DocSplittingUtils
    doc_splitter_util = DocSplittingUtils(chunk_size=1000,chunk_overlap=200,separator = "\n",is_separator_regex = False)
    texts = doc_splitter_util.split(data)

    # Vector Store
    from langchain.vectorstores import Chroma
    from chingmanlib.llm.pipeline import VectorStoreUtils
    vector_store_util = VectorStoreUtils(Chroma)
    retriever = vector_store_util.from_documents(texts, embeddings)

    return _template, template, retriever
    
if __name__ == "__main__":
    from chingmanlib.llm.models import create_llm
    import os

    BASE_DIR = os.getenv("BASE_DIR", "/home/aurora")
    cache_dir = os.path.join(BASE_DIR,"models","llama")

    llm_executor = create_llm("hf",cache_dir)

    llm = llm_executor.llm

    llm_chain_executor = BasicChain(llm)
    # llm_chain_executor.create_llm_chain()
    # llm_chain_executor.test_llm_chain()

    ###################
    _template, template, retriever = prepare_llm_conversation_QA_chain(llm_executor)
    # llm_chain_executor.create_llm_conversationn_QA_chain(_template, template, retriever)
    # llm_chain_executor.test_llm_conversationn_QA_chain()

    llm_chain_executor.create_llm_conversationn_QA_chain(_template, template, retriever, has_memory=True)
    llm_chain_executor.test_llm_conversationn_QA_chain()
    llm_chain_executor.test_llm_conversationn_QA_chain_with_memory()

    # export PYTHONPATH=/home/aurora/repos/dl-ex/submodules:$PYTHONPATH
    # python -m submodules.chingmanlib.llm.chains.basic
