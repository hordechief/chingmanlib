from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from .basic import BasicChain
from chingmanlib.llm.prompts.templates import CHATBOT_TEMPLATE, CHATBOT_CONVERSATION_TEMPLATE, CHATBOT_RAG_TEMPLATE

class LLMChainUtilizer(BasicChain):    
    def create_llm_chain(self, 
                template=None, 
                input_variables=None, 
                has_memory=False,
                verbose=False):
        
        # create template
        prompt = self.create_prompt(template, input_variables=input_variables, default_template=CHATBOT_TEMPLATE)
        
        # add memory
        kwargs = {}
        if has_memory:
            memory = ConversationBufferMemory(memory_key="chat_history")
            kwargs["memory"] = memory
            
            if not "chat_history" in input_variables:
                raise KeyError("variable chat_history should be included in input_variables to support memory")
            
        # create LLMChain
        llm_chain = LLMChain(
            prompt=prompt, 
            llm=self.llm, 
            verbose=verbose, 
            **kwargs)
        
        return llm_chain    
    
# https://python.langchain.com/v0.1/docs/modules/memory/conversational_customization/
class ConversationChainUtilizer(BasicChain):    
    def create_llm_conversation_QA_chain(self,
                template=None, 
                input_variables=None, 
                verbose=False):

        # create prompt
        PROMPT = self.create_prompt(template, input_variables=input_variables, default_template=CHATBOT_CONVERSATION_TEMPLATE)

        # create ConversationChain
        self.conversation_chain = ConversationChain(
            prompt=PROMPT,
            llm=self.llm,
            verbose=verbose,
            memory=ConversationBufferMemory(ai_prefix="AI Assistant"), # Here it is by default set to "AI"
        )        
        
        return self.conversation_chain
    
    def run(self, prompt):
        self.conversation.predict(input=prompt)

class RetrievalQAUtilizer(BasicChain):
    def create_llm_rag_chain(self,rag_template=None,retriever=None,input_variables=None):
        # create retriever
        if not retriever:
            retriever = self.retriever
            
        # create rag prompt
        rag_prompt = self.create_prompt(rag_template, input_variables=input_variables, default_template=CHATBOT_RAG_TEMPLATE)

        chain_type_kwargs = {"prompt": rag_prompt}

        rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
        )
                
        return rag_chain            