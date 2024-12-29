from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, format_document, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel,RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
import os
from operator import itemgetter # 从上面的变量中提取
import textwrap

from .chain_interface import BaseChain
from chingmanlib.llm.utils.log import LOG
from chingmanlib.llm.prompts.utils import PromptUtils
from chingmanlib.llm.prompts.templates import LLM_TEMPLATES
LANGUAGE = os.environ.get("TEMPLATE_LANGUAGE", 'en')

# https://python.langchain.com.cn/docs/expression_language/cookbook/memory
# LCEL

class LCECChain(BaseChain):
        
    def create_chain(self, 
                prompt=None, 
                streaming=False,
                verbose=False, 
                **kwargs):
            
        raise NotImplemented
            
    
class LLMChain(LCECChain):

    def create_chain(self, 
                prompt=None, 
                streaming=False,
                verbose=False, 
                **kwargs):
            
        '''
        Basic chain
        '''
        if not prompt:
            template = """Question: {question}
            Answer: Let's think step by step."""
        
            prompt = PromptTemplate.from_template(template)

        self.chain = prompt | self.llm | StrOutputParser()
        
        return self.chain   
    
    def run(self, question, verbose=False):
        prompt_value = self.chain.invoke({"question": question})
        if verbose: print(prompt_value)
        return prompt_value
                    
    def steam(self, question, verbose=False):
        pass

class RetrievalQAChain(LCECChain):
    
    def __create_prompt(self, system_prompt=None):
        # system_prompt = system_prompt or textwrap.dedent()('''
        #     Use the given context to answer the question. 
        #     If you don't know the answer, say you don't know. 
        #     Use three sentence maximum and keep the answer concise. 
        #     Context: {context}
        # ''')

        system_prompt = system_prompt or (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        return prompt
    
    def create_chain(self, 
                prompt=None, 
                streaming=False,
                verbose=False, 
                **kwargs):    
        
        '''
        basic RAG chain
        '''

        # create retriever
        retriever = kwargs.get("retriever", None) or self.retriever
        assert retriever
            
        # create rag prompt
        rag_prompt = prompt or \
            self.create_prompt(
                kwargs.get("template", None), 
                input_variables=kwargs.get("input_variables", None), 
                default_template_name=kwargs.get("default_template_name", "CHATBOT_RAG_TEMPLATE_LLAMA")) or \
            self.__create_prompt()
        
        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt 
            | self.llm
            | StrOutputParser()
        )
                        
        return self.rag_chain    
    
    def run(self, question, verbose=False):
        response = self.rag_chain.invoke(question)
        if verbose: print(response)
        return response

        # This issue can be fixed when change to call with invoke if HuggingFacePipeline
        # resp = PromptUtils.cut_off_text_for_answer(response["result"],"Answer: ")
        # print(resp)

        # # by default, openai return AIMessage, llama return str
        # return resp     
                    
    def steam(self, question, verbose=False):
        pass   
        

class ConversationChain(LCECChain):
    def __create_prompt(self, system_prompt=None):
        prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful chatbot"),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ]
        )
        prompt = ChatPromptTemplate.from_template(textwrap.dedent('''
            You are a helpful chatbot.

            {history}

            {input}
            '''))
        
        return prompt

    def create_chain(self, 
                prompt=None, 
                streaming=False,
                verbose=False, 
                **kwargs):           

        '''
        Basic QA chain
        '''

        prompt = prompt or \
            self.create_prompt(
                kwargs.get("template", None), 
                input_variables=kwargs.get("input_variables", None), 
                default_template_name=kwargs.get("default_template_name", "CHATBOT_CONVERSATION_TEMPLATE_LLAMA")) or \
            self.__create_prompt()

        self.memory = ConversationBufferMemory(
            return_messages=True, # output_key="answer", input_key="question"
            )

        self.conversational_qa_chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history")
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return self.conversational_qa_chain
        
    def run(self, question, verbose=False):

        if verbose: print(self.memory.load_memory_variables({}))

        response = self.conversational_qa_chain.invoke(
            {
                "input": question,
                "history": self.memory,
            }
        )
        self.memory.save_context({"question": question}, {"answer": response})

        if verbose: print(response)

        return response
            
    def stream(self, question, verbose=False):
        response = self.conversational_qa_chain.stream(
            {
                "input": question, 
                "history": self.memory
            }
        )

        # FIXME - This is a Generator
        self.memory.save_context({"question": question}, {"answer": response})

        varaibles = self.memory.load_memory_variables({})
        if verbose: LOG.log(varaibles)

        return response
                
    # 添加 __call__ 方法使得实例可以像函数一样调用
    def __call__(self, question, verbose=False):
        LOG.log("ConversationChain::__call__ is called")
        return self.run(question, verbose)

    # 你也可以根据需要创建一个 Tool，使其更适合于 LangChain 环境
    def to_tool(self):
        return RunnableLambda(self.run)
                    
# FIXME is there a built-in parser?                    
# from langchain_core.output_parsers.base import BaseOutputParser
# class CustomOutputParser(BaseOutputParser):
#     # {'answer': ' 根据提供的Table 24硬件安全，Precision 5470支持的操作系统有:\n\n- Windows 10 Pro\n- Windows 10 Enterprise\n- Windows 10 Education\n- Linux (XenDesktop/XenApp)\n\n因此，Precision 5470支持Windows 10系列和Linux。', 
#     # 'docs': [
#     # Document(metadata={'page': 38, 'source': '/home/aurora/kb/dell/precision-5470-technical-guidebook.pdf'}, 
#     # page_content='Table 51. Precision 5470 specific testing \nTest Name Test procedure Specifications\n● Non-operating\nThermal and acoustic improvements\nThe following table lists the thermal and acoustic improvements of your Precision 5470.\nTable 52. Thermal and acoustic improvements  \n100% dual heat pipe Increase the heat capacity to improve thermal dissipation\nBetter system tuning/setting Get higher performance and good user experience\nPro-OS enhanced thermal setting (Dynamic PL1) Increases boot-up time'), 
#     # Document(metadata={'page': 35, 'source': '/home/aurora/kb/dell/precision-5480-technical-guidebook.pdf'}, 
#     # page_content='c\nomplaint\nDevice complaint with FIPS 140-2 level 3\nr\nequirements\nYes\nFIPS 140-2 level 3\nc\nertified\nDevice certified with FIPS 140-2 level 3\nr\nequirements\nYes\nTrusted Platform Module\nT\nhe following table lists the Trusted Platform Module (TPM) of your Precision 5480.\nTable 47. Trusted Platform Module (TPM) \nTPM: ST/ST33 HTPH2X32AHD8\nSPI interface\nTPM 2.0\nFIPs 140-2 certificate\n36 Engineering specifications'), 
#     # Document(metadata={'page': 39, 'source': '/home/aurora/kb/dell/precision-5480-technical-guidebook.pdf'}, 
#     # page_content='● Complete protection\nagainst contact\n● Non-operating\nThermal and acoustic improvements\nT\nhe following table lists the thermal and acoustic improvements of your Precision 5480.\nTable 51. Thermal and acoustic improvements  \n100% dual heat pipe Increase the heat capacity to improve thermal dissipation\nBetter system tuning/setting Get higher performance and good user experience\nPro-OS enhanced thermal setting (Dynamic PL1) Increases boot-up time'), 
#     # Document(metadata={'page': 19, 'source': '/home/aurora/kb/dell/precision-5470-technical-guidebook.pdf'}, 
#     # page_content='Hardware security\nThe following table lists the hardware security of your Precision 5470.\nTable 24. Hardware security  \nHardware security\nTrusted Platform Module (TPM) 2.0 discrete\nFIPS 140-2 certification for TPM\nTCG Certificatication for TPM (Trusted Computing Group)\nContacted Smart Card and Control Vault 3\nContactless Smart Card, NFC, and ControlVault 3\nSED SSD NVMe, SSD, and HDD (Opal and non-Opal) per SDL\nFinger Print Reader in Power Button\nSED (Opal 2.0 only - PCIe Interface)')]}
#     def parse(self, output: dict):
#         LOG.log("parser of chain: ", output)
#         answer = output.get('answer')
#         return answer
#     #  Input should be a valid string [type=string_type, input_value={'answer': AIMessage(cont... out when the key is')]}, input_type=dict]
                        
class ConversationRetrievalQAChain(LCECChain):

    def create_chain(self, 
                prompt=None, 
                streaming=False,
                verbose=False, 
                **kwargs):        
        
        '''
        This chain leverage 3 prompt (question prompt, answer prompt and document prompt) to compose the chain

        Note: language in template is not mandatory, just give an example to add additional paramter
        '''
        print("create_llm_conversationn_rag_qa_chain")

        self.has_memory =  kwargs.get("has_memory", False)

        # create retriever
        retriever = kwargs.get("retriever", None) or self.retriever
        assert retriever
            
        question_prompt = kwargs.get("question_prompt", None) or \
                textwrap.dedent(LLM_TEMPLATES["CHATBOT_CONVERSATION_RAG_QUESTION_TEMPLATE"]["template"][LANGUAGE])
                    
        answer_prompt = kwargs.get("answer_prompt", None) or \
                textwrap.dedent(LLM_TEMPLATES["CHATBOT_CONVERSATION_RAG_ANSWER_TEMPLATE"]["template"][LANGUAGE])
                        
        # question prompt 
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(question_prompt)
        LOG.log("CONDENSE_QUESTION_PROMPT is :", CONDENSE_QUESTION_PROMPT)
        
        # answer prompt - [ChatPromptTemplate]
        ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_prompt)
        LOG.log("ANSWER_PROMPT is :",ANSWER_PROMPT)
        
        # document prompt
        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}") # page_content 应该是Document的变量
        LOG.log("DEFAULT_DOCUMENT_PROMPT is :", DEFAULT_DOCUMENT_PROMPT)

        def _combine_documents(
            docs, 
            document_prompt=DEFAULT_DOCUMENT_PROMPT, 
            document_separator="\n\n"
        ):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)
        
        # Now we calculate the standalone question        
        if not self.has_memory:   
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
                # "language": RunnablePassthrough(), # OK
                # "language": lambda x: x["language"], # KO, _input中提供的变量，否则这儿无法获取 ####
                # "language": itemgetter("language") # KO
            }  
                                    
            self.conversational_rag_qa_chain = _inputs | _context | ANSWER_PROMPT | self.llm 
            
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
                    # "language": RunnablePassthrough(),  # OK
                    # "language": itemgetter("language"),  # OK
                    # "language": lambda x: x["language"], # OK
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
                # "language": RunnablePassthrough(), # OK
                # "language": itemgetter("language"), # KO
            }

            # And finally, we do the part that returns the answers
            answer = {
                "answer": final_inputs | ANSWER_PROMPT | self.llm, 
                "docs": itemgetter("docs"),
            }        

            # And now we put it all together!
            self.conversational_rag_qa_chain = loaded_memory | standalone_question | retrieved_documents | answer # | CustomOutputParser()
                
        return self.conversational_rag_qa_chain    
    
    def run(self, question, verbose=False):
        '''
        run inference to get the answer
        currently, we need manually save the memory
        '''
        inputs = {"question": question}
        result = self.conversational_rag_qa_chain.invoke(inputs)
        
        if verbose: 
            LOG.log("Chain invoke result is:", "\n<Answer>:", result['answer'], "\n<docs>:",  result['docs'][0])

        if isinstance(result["answer"], AIMessage):
            response = result["answer"].content
        else:
            response = result["answer"]

        if self.has_memory:
            # Note that the memory does not save automatically
            # This will be improved in the future
            # For now you need to save it yourself
            self.memory.save_context(inputs, {"answer": response})
            varaibles = self.memory.load_memory_variables({})
            if verbose: 
                LOG.log("memory varaibles is :", varaibles)

        return response
    
    def stream(self, question, verbose=False):
        inputs = {"question": question}
        response = self.conversational_rag_qa_chain.stream(inputs)

        if self.has_memory:
            self.memory.save_context(inputs, {"answer": response})
            varaibles = self.memory.load_memory_variables({})
            if verbose: LOG.log(varaibles)

        return response
    
    async def astream(self, question, verbose):
        inputs = {"question": question}
        response = self.conversational_rag_qa_chain.astream(inputs)

        if self.has_memory:
            self.memory.save_context(inputs, {"answer": response})
            varaibles = self.memory.load_memory_variables({})
            if verbose: LOG.log(varaibles)

        return response
    

class RouteChain():
    pass


if __name__ == "__main__":
    # os.system("export PYTHONPATH=/home/aurora/repos/dl-ex/submodules:$PYTHONPATH")
    import sys
    sys.path.append("/home/aurora/repos/CTE-LLM")
    from chingmanlib.llm.models import create_llm
    from chingmanlib.llm.db import create_data_pipeline
    from chingmanlib.llm.models.hf import HFEmbeddings
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    env_path = Path(__file__).parent.parent / ".env"
    assert os.path.exists(env_path)
    load_dotenv(dotenv_path=str(env_path), override=True)  # 加载 .env 文件    

    # llm_executor = create_llm("hf",os.environ["CACHE_DIR"],model_name="EntropyYue/chatglm3:6b")
    # llm_executor = create_llm("hf",os.environ["CACHE_DIR"])
    llm_executor = create_llm("ollama",os.environ["CACHE_DIR"])
    # llm_executor = create_llm("openai")

    llm = llm_executor.llm
    embeddings = HFEmbeddings(model_name="jinaai/jina-embeddings-v2-base-zh").create_embeddings()

    retriever = create_data_pipeline("/home/aurora/repos/dl-ex/llm/docs/The_Guide_To_LLM_Evals.pdf", embeddings).retriever

    TEST_CATEGORIES = ['CONVERSATION'] # ['BASE', 'CONVERSATION', 'CONVERSATION_RAG']

    if 'BASE' in TEST_CATEGORIES: 
        llm_chain = LLMChain(llm=llm, retriever=retriever)  

    if 'RAG' in TEST_CATEGORIES:
        llm_chain = RetrievalQAChain(llm=llm, retriever=retriever)  

    if 'CONVERSATION' in TEST_CATEGORIES:
        llm_chain = ConversationChain(llm=llm, retriever=retriever)  

    if 'CONVERSATION_RAG' in TEST_CATEGORIES:      
        llm_chain = ConversationRetrievalQAChain(llm=llm, retriever=retriever) 

    llm_chain.create_chain(has_memory=True)
    while True:
        print("\n>> (Press Ctrl+C to exit.)")
        inputs = input(">> ")
        chat_complet = llm_chain.run(inputs, verbose=True)        

        import types
        if isinstance(chat_complet, types.GeneratorType):
            for chunk in chat_complet:
                if chunk:
                    print(chunk, end="")
        else:
            print(chat_complet)

    # python -m chingmanlib.llm.chains.lcec
