from langchain.chains import LLMChain
from langchain.chains import ConversationChain # child class of LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA # wrapper of load_qa_chain
from langchain.chains import ConversationalRetrievalChain # extension of RetrievalQA
from langchain.chains import StuffDocumentsChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator # wrapper of RetrievalQA
from langchain.memory import ConversationBufferMemory

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import os

from .chain_interface import BaseChain

class LLMChainUtilizer(BaseChain):    
    '''
    Chat + History (Customized)
    '''
    def create_chain(self, 
                prompt=None,
                streaming=False,
                verbose=False, 
                **kwargs):
        
        has_memory = kwargs.get("has_memory", True)        
        # create template
        if not prompt:
            # TBD, use yaml to get
            if not has_memory:
                default_template_name="CHATBOT_TEMPLATE"
            else:
                default_template_name="CHATBOT_CONVERSATION_TEMPLATE_LLAMA"
                            
            template = kwargs.get("template", None)
            input_variables = kwargs.get("input_variables", None)
            
            prompt = self.create_prompt(template, input_variables=input_variables, default_template_name=default_template_name)
        
        # add memory
        chain_kwargs = {}
        if has_memory:
            memory = ConversationBufferMemory(memory_key="chat_history",
                                            #   return_messages=True, output_key="answer", input_key="question"
                                              )
            chain_kwargs["memory"] = memory

            if verbose: print(f"\n==================\nMemory is : \n{memory}")
            
        if streaming:
            self.steaming = True

        # create LLMChain
        self.chain = LLMChain(
            prompt=prompt, 
            llm=self.llm, 
            verbose=verbose, 
            **chain_kwargs)
        
        return self.chain    
    
    def run(self, question, verbose=False):
        response = self.chain.invoke({"question": question})
        if verbose: print(f"LLMChainUtilizer invoke response: \n {response}")
        # {'question': '我的上一个问题是什么', 'text': ' 您的问题是：“我的上一个问题是什么”。'}
        return response['text']
    
    def steam(self, question, verbose=False):
        if self.steaming:
            raise ValueError("streaming function not provided")
    
class ConversationChainUtilizer(BaseChain):    
    '''
    Inherit from LLMChain

    : CHAT + History
   '''
    def create_chain(self,
                prompt=None, 
                streaming=False,
                verbose=False, 
                **kwargs):

        # create prompt
        template = kwargs.get("template", None)
        input_variables = kwargs.get("input_variables", None)        
        PROMPT = prompt or self.create_prompt(template, input_variables=input_variables, default_template_name="CHATBOT_CONVERSATION_TEMPLATE")

        # create ConversationChain
        self.chain = ConversationChain(
            prompt=PROMPT,
            llm=self.llm, # .bind(skip_prompt=True) for huggingface
            verbose=verbose,
            memory=ConversationBufferMemory(ai_prefix="AI Assistant"), # Here it is by default set to "AI", variable is ['history'] 
        )        
        
        return self.chain
    
    def run(self, prompt, verbose=False):
        response = self.chain.predict(input=prompt)
        if verbose: print(f"ConversationChainUtilizer chain invoke response is: \n===================================\n {response}")
        return response
    
    def steam(self, question, verbose=False):
        pass
    
class RetrievalQAUtilizer(BaseChain):
    '''
    CHAT + RAG
    '''
    def create_chain(self,
                prompt=None, 
                streaming=False,
                verbose=False, 
                **kwargs):
        
        # create retriever
        retriever = kwargs.get("retriever", None) or self.retriever
        assert retriever
            
        # create rag prompt
        template = kwargs.get("template", None)
        input_variables = kwargs.get("input_variables", None)             
        rag_prompt = prompt or \
            self.create_prompt(template, input_variables=input_variables, default_template_name="CHATBOT_RAG_TEMPLATE_LLAMA") or \
            self.__create_prompt()

        chain_type_kwargs = {"prompt": rag_prompt}
        chain_type=kwargs.get("chain_type", "stuff")
        # the result provide the source document information
        return_source_documents=kwargs.get("return_source_documents", True)

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type = chain_type,
            return_source_documents=return_source_documents,
            chain_type_kwargs=chain_type_kwargs,
        )
                
        return self.chain            
    
    def __create_prompt(self, system_prompt=None):
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
        
    def create_chain_built(self, 
                prompt=None, 
                streaming=False,
                verbose=False,
                **kwargs):
        '''
        NOT TESTED
        '''
        # create retriever
        retriever = kwargs.get("retriever", None) or self.retriever
        assert retriever

        # create rag prompt
        template = kwargs.get("template", None)
        input_variables = kwargs.get("input_variables", None)           
        prompt = prompt or \
            self.create_prompt(template, input_variables=input_variables, default_template_name="CHATBOT_RAG_TEMPLATE_LLAMA") or \
            self.__create_prompt()

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        self.chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return self.chain
    
    def load_qa_chain_wrapper(self, file_path, query):
        '''
        NOT TESTED
        '''
        # load document
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        chain = load_qa_chain(self.llm, chain_type="map_reduce") # 'stuff'

        response = chain.run(input_documents=documents, question=query)

        return response

    def run(self, prompt, verbose=False):
        response = self.chain.invoke(input=prompt)
        if verbose: print(f"RetrievalQAUtilizer invoke response is: \n===================================\n {response}")
        return response["result"]
        # from chingmanlib.llm.prompts.utils import PromptUtils
        # return PromptUtils.cut_off_text_for_answer(response["result"],f"# Question: {prompt}\n\nAnswer: ")    

    def steam(self, question, verbose=False):
        pass

class ConversationRetrievalQAUtilizer(BaseChain):
    '''
    NOT TESTED

    CHAT + RAG + History
    '''
    def __init__(self, llm, retriever,**kwargs):
        self.chat_history = []

        self.question_prompt = kwargs.get("question_prompt", None) or self.__create_contextualize_prompt()
        self.answer_prompt = kwargs.get("answer_prompt", None) or self.__create_qa_prompt()

        super().__init__(llm, retriever)

    def create_chain(self,
                prompt=None, 
                streaming=False,
                verbose=False, 
                **kwargs):
        '''
        Deprecated since version 0.1.17. It will be removed in None==1.0.
        '''
        retriever = kwargs.get("retriever", None) or self.retriever
        assert retriever

        # self.conversation_rag_qa = ConversationalRetrievalChain.from_llm(self.llm, retriever)

        # This controls how each document will be formatted. Specifically,
        # it will be passed to `format_document` - see that function for more details.
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )
        document_variable_name = "context"
        # The prompt here should take as an input variable the `document_variable_name`
        doc_chat_prompt = PromptTemplate.from_template(
            "Summarize this content: {context}"
        )
        llm_chain = LLMChain(llm=self.llm, prompt=doc_chat_prompt)
        combine_docs_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        )

        # This controls how the standalone question is generated.
        # Should take `chat_history` and `question` as input variables.
        if not self.question_prompt:
            standalone_question_generator_template = (
                "Combine the chat history and follow up question into "
                "a standalone question. Chat History: {chat_history}"
                "Follow up question: {question}"
            )
            standalone_question_generator_prompt = PromptTemplate.from_template(standalone_question_generator_template)
            self.question_prompt = standalone_question_generator_prompt

        question_generator_chain = LLMChain(llm=self.llm, prompt=self.question_prompt)

        self.conversation_rag_qa = ConversationalRetrievalChain(
            combine_docs_chain=combine_docs_chain,
            retriever=retriever,
            question_generator=question_generator_chain,
        ) 
        
        return self.conversation_rag_qa
    
    def __create_contextualize_prompt(self, system_prompt=None):
        # Contextualize question
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )        

        # alternative
        prompt_search_query = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user","{input}"),
            ("user","Given the above conversation, generate a search query to look up to get information relevant to the conversation")
            ])
        
        return contextualize_q_prompt
    
    def __create_qa_prompt(self):
        # Answer question
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know. Use three sentences maximum and keep the answer "
            "concise."
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
            ("system", qa_system_prompt), 
            MessagesPlaceholder("chat_history"), 
            ("human", "{input}"),
            ]
        )

        # alternative
        prompt_get_answer = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\\n\\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user","{input}"),
            ])

        return qa_prompt
        
    def create_chain_built(self, **kwargs):
        # create retriever
        retriever = kwargs.get("retriever", None) or self.retriever
        assert retriever

        contextualize_prompt = self.question_prompt or self.__create_contextualize_prompt()
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_prompt
        )

        # Below we use create_stuff_documents_chain to feed all retrieved context 
        # into the LLM. Note that we can also use StuffDocumentsChain and other 
        # instances of BaseCombineDocumentsChain. 
        qa_prompt = self.answer_prompt or self.__create_qa_prompt()
        question_answer_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=qa_prompt
        )

        self.conversation_rag_qa = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        return self.conversation_rag_qa

    def run(self, question, verbose=False):
        # Usage: chat_history = [] 
        # Collect chat history here (a sequence of messages) 
        response = self.conversation_rag_qa.invoke({
            # "input": question,
            "question": question, # input and question should keep one only
            "chat_history": self.chat_history
        })

        self.chat_history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=response["answer"]),
            ]
        )

        if verbose: print(f"ConversationRetrievalQAUtilizer invoke response: \n {response}")

        return response["answer"]

    def steam(self, question, verbose=False):
        pass

class VectorstoreIndexCreatorUtiliizer(BaseChain):
    '''
    NOT TESTED
    '''
    def create_chain(self, loader, text_splitter, embedding, vectorstore_cls=Chroma, **kwargs):
        self.index = VectorstoreIndexCreator(
            text_splitter=text_splitter,
            embedding=embedding,
            vectorstore_cls=vectorstore_cls
        ).from_loaders([loader])
        return self.index
    
    def run(self, query):
        response = self.index.query(llm=self.llm, question=query, chain_type="stuff")
        return response
    
    def steam(self, question, verbose=False):
        pass

if __name__ == "__main__":
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from chingmanlib.llm.db import create_data_pipeline
    client = Ollama(model="EntropyYue/chatglm3:6b")
    retriever = create_data_pipeline("/home/aurora/repos/dl-ex/llm/data/", HuggingFaceEmbeddings(model_name='jinaai/jina-embeddings-v2-base-zh'))

    CHAIN_TYPE = "CONVERSATION_RETRIEVAL_QA"
    if "LLM_CHAIN" == CHAIN_TYPE:
        chain_utilizer = LLMChainUtilizer(llm=client, retriever=retriever)
        # chain_utilizer.create_chain(has_memory=False)
        chain_utilizer.create_chain(has_memory=True, verbose=True)    
    elif "CONVERSATION_CHAIN" == CHAIN_TYPE:
        chain_utilizer = ConversationChainUtilizer(llm=client, retriever=retriever)
        chain_utilizer.create_chain(verbose=True) 
    elif "RETRIEVAL_QA" == CHAIN_TYPE:
        chain_utilizer = RetrievalQAUtilizer(llm=client, retriever=retriever)
        chain_utilizer.create_chain(verbose=True)
    elif "CONVERSATION_RETRIEVAL_QA" == CHAIN_TYPE:
        chain_utilizer = ConversationRetrievalQAUtilizer(llm=client, retriever=retriever)
        # chain_utilizer.create_chain_built(verbose=True)
        chain_utilizer.create_chain(verbose=True)

    while True:
        print("\n>> (Press Ctrl+C to exit.)")
        chat_complet = chain_utilizer.run(input(">> "), verbose=True)
        # print(type(chat_complet))

        import types
        if isinstance(chat_complet, types.GeneratorType):
            for chunk in chat_complet:
                if chunk:
                    print(chunk, end="")
        else:
            print("\nChat result is: \n")
            print(chat_complet)        

    # export PYTHONPATH="/home/aurora/repos/CTE-LLM/:$PYTHONPATH"