from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from .chain_interface import BaseChain
from chingmanlib.llm.prompts.templates import CHATBOT_TEMPLATE, CHATBOT_CN_TEMPLATE, CHATBOT_HISTORY_TEMPLATE, CHATBOT_HISTORY_CN_TEMPLATE, CHATBOT_CONVERSATION_TEMPLATE, CHATBOT_RAG_TEMPLATE

class LLMChainUtilizer(BaseChain):    
    def create_chain(self, 
                template=None, 
                input_variables=None, 
                has_memory=False,
                streaming=False,
                verbose=False, 
                **kwargs):
        
        # create template
        if not has_memory:
            default_template=CHATBOT_CN_TEMPLATE
        else:
            default_template=CHATBOT_HISTORY_CN_TEMPLATE

            if not "chat_history" in default_template["template"]:
                raise KeyError("variable chat_history should be included in template to support memory")
                        
        prompt = self.create_prompt(template, input_variables=input_variables, default_template=default_template)
        
        # add memory
        kwargs = {}
        if has_memory:
            memory = ConversationBufferMemory(memory_key="chat_history",
                                            #   return_messages=True, output_key="answer", input_key="question"
                                              )
            kwargs["memory"] = memory
            if verbose:
                print(f"\n==================\nMemory is : \n{memory}")
            
        # create LLMChain
        self.chain = LLMChain(
            prompt=prompt, 
            llm=self.llm, 
            verbose=verbose, 
            **kwargs)
        
        return self.chain    
    
    def run(self, question, verbose=False):
        response = self.chain.invoke({"question": question})
        if verbose: print(f"chain invoke response is: \n {response}")
        # {'question': '我的上一个问题是什么', 'text': ' 您的问题是：“我的上一个问题是什么”。'}
        return response['text']
    
# https://python.langchain.com/v0.1/docs/modules/memory/conversational_customization/
class ConversationChainUtilizer(BaseChain):    
    def create_chain(self,
                template=None, 
                input_variables=None, 
                has_memory=False,
                streaming=False,
                verbose=False, 
                **kwargs):

        # create prompt
        PROMPT = self.create_prompt(template, input_variables=input_variables, default_template=CHATBOT_CONVERSATION_TEMPLATE)

        # create ConversationChain
        self.chain = ConversationChain(
            prompt=PROMPT,
            llm=self.llm, # .bind(skip_prompt=True) for huggingface
            verbose=verbose,
            memory=ConversationBufferMemory(ai_prefix="AI Assistant"), # Here it is by default set to "AI"
        )        
        
        return self.chain
    
    def run(self, prompt, verbose=False):
        response = self.chain.predict(input=prompt)
        if verbose: print(f"ConversationChainUtilizer chain invoke response is: \n===================================\n {response}")
        return response
    
class RetrievalQAUtilizer(BaseChain):
    def create_chain(self,
                template=None, 
                input_variables=None, 
                has_memory=False,
                streaming=False,
                verbose=False, 
                **kwargs):
        # create retriever
        retriever = kwargs.get("retriever", None)
        if not retriever:
            retriever = self.retriever

        assert retriever
            
        # create rag prompt
        rag_prompt = self.create_prompt(template, input_variables=input_variables, default_template=CHATBOT_RAG_TEMPLATE)

        chain_type_kwargs = {"prompt": rag_prompt}

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
        )
                
        return self.chain            
    
    def run(self, prompt, verbose=False):
        response = self.chain.invoke(input=prompt)
        if verbose: print(f"RetrievalQAUtilizer invoke response is: \n===================================\\n {response}")
        return response["result"]
        # from chingmanlib.llm.prompts.utils import PromptUtils
        # return PromptUtils.cut_off_text_for_answer(response["result"],f"# Question: {prompt}\n\nAnswer: ")    