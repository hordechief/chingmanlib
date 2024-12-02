# https://python.langchain.com.cn/docs/expression_language/get_started
# RunnablePassthrough https://python.langchain.com.cn/docs/expression_language/how_to/passthrough
# RunnableParallel=RunnableMap: https://python.langchain.com.cn/docs/expression_language/how_to/map
# https://python.langchain.com.cn/docs/expression_language/cookbook/retrieval
# https://python.langchain.com/v0.1/docs/expression_language/primitives/assign/
# The RunnablePassthrough.assign(...) static method takes an input value and adds the extra arguments passed to the assign function. This is useful when additively creating a dictionary to use as input to a later step, which is a common LCEL pattern.

# import executor of LLM models
from llm.llm_hf import LLMHFxecutor
from llm.llm_llamacpp import LlamaCppExecutor

# import sys
# sys.path.append(lib_dir)
# from llm_app import LLMExecutorApp
def call_llm_app(cache_dir,device,model="hf",**kwargs):
    
    model_kwargs = kwargs.get("model_kwargs", {'temperature':0})
    base_dir = kwargs.get("base_dir", None)    

    if "hf" == model:
        llm_hf_executor = LLMHFxecutor(
            base_dir=base_dir,
            cache_dir=cache_dir, 
            device=device,
            model_kwargs = model_kwargs
        )
    elif "lamma_cpp" == model:
        llamacpp_executor = LlamaCppExecutor(
            base_dir=base_dir,
            cache_dir=cache_dir, 
            device=device,
            model_name="llama-2-7b-chat.Q4_K_M.gguf",
            model_kwargs = model_kwargs
        )
    else:
        raise ValueError("llm model must be given")
   
    llm_exe_app = LLMExecutorApp(llm = llm_hf_executor.llm)
    
    embeddings_hfe = llm_hf_executor.get_embeddings()

    return llm_exe_app, llm_hf_executor, embeddings_hfe

def call_data_pipeline(**kwargs):
    split_kwargs = kwargs.get("split_kwargs", None)
    if not split_kwargs:
        split_kwargs = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separator": "\n",
            "is_separator_regex": False
        }
    
    from llm.pipeline import DataLoaderUtils
    dlu = DataLoaderUtils()  
    
    from llm.pipeline import DocSplittingUtils
    dsu = DocSplittingUtils(**kwargs)
    
    return dlu, dsu

from llm.pipeline import VectorStoreUtils, DocSplittingUtils

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

from langchain_openai.chat_models import ChatOpenAI

def run_openai_test_completion():
    # https://github.com/openai/openai-quickstart-python
    from openai import OpenAI
    client = OpenAI()

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
        {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
      ]
    )

    response = completion.choices[0].message

    return response

def run_openai_test_human_message():
    from langchain_core.messages import AIMessage, HumanMessage
    
    model = ChatOpenAI()
    message = HumanMessage(
        content="Translate this sentence from English to French. I love programming.")
    model([message])

        
system_prompt = """
The flowchart outlines a troubleshooting process for hardware or configuration issues, particularly focusing on PCIe and related components. The flow is organized into several branches based on the type of issue encountered. Here’s the detailed description:

Start: The process begins with identifying the problem related to the troubleshooting steps outlined in the flowchart.

Step 1: Retrieve detailed error information.

Branch 1: If NDC/Mini Perc reports Bus:Device error.
Step 1.1: Perform compatibility test on the customer's system.
Branch 1.1.1: If the test passes.
Step 1.1.1.1: Follow specific customer escalation process if needed.
Branch 1.1.2: If the test fails.
Step 1.1.2.1: Check support documentation or escalate according to procedure.
Branch 2: If HBA/NIC/NDC only has one port recognized, while others are functioning.
Step 2.1: Isolate the issue to the specific PCIe card.
Step 2.1.1: Follow the specific steps for PCIe card troubleshooting.
Branch 3: If the PCIe Adapter cannot be recognized.
Step 3.1: Isolate the issue to the specific PCIe Adapter card.
Step 3.1.1: Check DIMM slot configurations and compatibility with MB (Motherboard).
Step 3.1.2: Follow support procedures to resolve the issue.
Branch 4: If other errors cannot be resolved.
Step 4.1: Analyze and perform additional tests to narrow down the issue.
Step 4.1.1: If the issue persists with multiple devices.
Step 4.1.1.1: Escalate the issue as it might be complex.
Step 4.1.2: If the issue persists with a single device.
Step 4.1.2.1: Perform compatibility test on the customer's system.
Step 4.1.2.1.1: If the test passes.
Step 4.1.2.1.1.1: Follow specific customer escalation process if needed.
Step 4.1.2.1.2: If the test fails.
Step 4.1.2.1.2.1: Check support documentation or escalate according to procedure.
"""

question = """
please troubleshooting on below issue step by step, and give details of each step.

Question: How to solve the issue of PCIe Adapter cannot be recognized? what's the potential error
"""

#*******************************************
#
#             LLM Model
#
#******************************************

from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableParallel,RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter # 从上面的变量中提取

class LLMExecutorApp():
    def __init__(self,**config):
        # LLM
        self.llm = config.get("llm", None)
        
        # embeddings
        self.embeddings = config.get("embeddings", None)

        # vector db
        vdb = config.get("vdb", None)
        if vdb:
            self.vsu = self.set_vsu(vdb)
            
        # text split
        split_param = config.get("split_param", None)
        if split_param:
            self.dsu = self.set_dsu(**split_param)
    
    #############################################################
    # Vector DB
    def set_vsu(vdb):
        self.vsu = VectorStoreUtils(vdb)
        
    def set_retriever(self,texts):
        self.retriever = self.vsu.from_documents(texts, self.embeddings)
        
    def get_retriever(self):
        return self.retriever

    ##############################################################
    # Text Splitting
    def set_dsu(**split_param):
        self.dsu = DocSplittingUtils(**split_param)
        
    def split_docs(self,data):        
        self.texts = self.dsu.split(data)
    
    def get_split_docs(self):
        return self.texts
    
    #############################################################
    # LLM
    def run_basic_test(self):
        from langchain_core.messages import AIMessage
        response = self.llm("Translate this sentence from English to French. I love programming.")
        return response, AIMessage(response)
        
    def run_test_troubleshooting(self):
        prompt = system_prompt + question
        response = self.llm(prompt)
        return response
    
    def run_test_troubleshooting_with_openai(self):
        if not isinstance(self.llm, ChatOpenAI):
            print("this funciton only valid for OpenAI llm instance")
            return
        
        completion = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "How to solve the issue of PCIe Adapter cannot be recognized?"}
          ]
        )

        response = completion.choices[0].message
        
        return response
    
    ###############################################################
    # LLM Chain
    def check_input_variables(self,template,input_variables):
        for variable in input_variables:
            _variable = "{" + variable + "}"
            # print(_variable)
            if not _variable in template:
                print(f"variable {_variable} is not included in template")
                return False
        return True
        
    # 管道方式
    def llm_chain_pipeline(self,template=None):
        if not template:
            template = """Question: {question}
            Answer: Let's think step by step."""
        
        prompt = PromptTemplate.from_template(template)
        llm_chain = prompt | self.llm
        
        return llm_chain

    def llm_rag_chain_pipeline(self,rag_template=None,retriever=None):
        if not retriever:
            retriever = self.retriever
            
        if not rag_template:
            rag_template="""
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            {context}

            Question: {question}

            Answer:
            """
        else:
            # "context" and "question" are input_variables
            for _variable in ["context","question"]:
                if not _variable in rag_template:
                    return None
            
        rag_prompt = PromptTemplate.from_template(rag_template)
            
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt 
            | self.llm
            | StrOutputParser()
        )
        
        print(type(rag_chain))
                
        return rag_chain
    
    def llm_rag_chain_retrieval_QA_pipeline(self,question_prompt=None,answer_prompt=None,retriever=None,has_memory=False):
        
        # use default retriever
        if not retriever:
            retriever = self.retriever
            
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
        # print(CONDENSE_QUESTION_PROMPT)
        
        # answer prompt
        ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_prompt)
        # print(ANSWER_PROMPT)
        
        # dpcument prompt
        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}") # page_content 应该是Document的变量

        def _combine_documents(
            docs, 
            document_prompt=DEFAULT_DOCUMENT_PROMPT, 
            document_separator="\n\n"
        ):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)
        
        if has_memory:
            self.memory = ConversationBufferMemory(
                return_messages=True, output_key="answer", input_key="question"
            )
            
            # First we add a step to load memory
            # This adds a "memory" key to the input object
            loaded_memory = RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
            )
        else:
            self.memory = None 
        
        # Now we calculate the standalone question        
        if not has_memory:
                  
            _inputs = RunnableParallel(
                standalone_question = RunnablePassthrough.assign(
                    chat_history=lambda x: get_buffer_string(x["chat_history"]),
                )
                | CONDENSE_QUESTION_PROMPT
                | self.llm
                | StrOutputParser(),
                
                # language = lambda x: x["language"], #RunnablePassthrough() 是完整参数列表，通过lambda取指定参数
            )
                                
            _context = {
                "context": itemgetter("standalone_question") | retriever | _combine_documents,
                "question": lambda x: x["standalone_question"], # answer prompt里的question跟question prompt里是不一样的
                # "language": lambda x: x["language"], # _input中提取的变量，否则这儿无法获取
            }  
                                    
            conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | self.llm 
            
            # itemgetter("standalone_question") -> itemgetter("standalone_question")(_inputs)
            # lambda x: x["standalone_question"] -> lambda x: x["standalone_question"](_inputs)
            
        else:
            standalone_question = {
                "standalone_question": {
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: get_buffer_string(x["chat_history"]),
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
            }

            # And finally, we do the part that returns the answers
            answer = {
                "answer": final_inputs | ANSWER_PROMPT | self.llm,
                "docs": itemgetter("docs"),
            }        

            # And now we put it all together!
            conversational_qa_chain = loaded_memory | standalone_question | retrieved_documents | answer
                
        return conversational_qa_chain

    def llm_chain_with_LLMChain(self, template=None, input_variables=None, has_memory=False,verbose=False):
        if not template:
            template = """This is the Chatbot sytem ..."
            Let's work this out in a step by step way to be sure we have the right answer.
            Question: {question}
            Answer: """            

            prompt = PromptTemplate.from_template(template) 
        else:        
            # 如果您不想手动指定 input_variables，也可以使用 from_template 类方法创建 PromptTemplate。
            # LangChain 将根据传递的 template 自动推断 input_variables。
            # https://python.langchain.com.cn/docs/modules/model_io/prompts/prompt_templates
            
            if input_variables:
                if not self.check_input_variables(template,input_variables):
                    print("input variable is not correct")
                    return None

                prompt = PromptTemplate(template=template, input_variables=input_variables)
            else:
                prompt = PromptTemplate.from_template(template) 
        
        kwargs = {
            "prompt":prompt,
            "llm":self.llm,
            "verbose":verbose,
        }
        
        if has_memory:
            memory = ConversationBufferMemory(memory_key="chat_history")
            kwargs["memory"] = memory
            
            if not "chat_history" in input_variables:
                print("variable chat_history should be included in input_variables")
                return None           
            
        # llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        llm_chain = LLMChain(**kwargs)
        
        return llm_chain
    
    def llm_conversation_chain(self):
        # llm_chain = ConversationChain(prompt=prompt, llm=llm) # + chat_history
        pass
    
    def llm_rag_chain_from_RetrievalQA(self,rag_template=None,retriever=None,input_variables=None):
        if not retriever:
            retriever = self.retriever
            
        if not rag_template:
            rag_template = """
            [INST] <<SYS>>
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
            Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
            Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
            If you don't know the answer to a question, please don't share false information.
            <</SYS>>

            # Context: {context}

            # Question: {question}

            # Answer: [/INST]

            """
        else:
            if input_variables:
                if not self.check_input_variables(rag_template,input_variables):
                    print("input variable is not correct")
                    return None

                rag_prompt=PromptTemplate(input_variables=input_variables, template=rag_template)
            else:
                rag_prompt = PromptTemplate.from_template(rag_template)             
        
        from langchain.chains import RetrievalQA

        chain_type_kwargs = {"prompt": rag_prompt}

        rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
        )
                
        return rag_chain
    

#*******************************************
#
#             Prompt Template
#
#******************************************

import json
import textwrap


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""
class PromptUtils():
    def __init__():
        pass

    @classmethod
    def get_prompt(cls, instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
        SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
        prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
        return prompt_template

    @classmethod    
    def cut_off_text(cls, text, prompt):
        cutoff_phrase = prompt
        index = text.find(cutoff_phrase)
        if index != -1:
            return text[:index]
        else:
            return text

    @classmethod
    def cut_off_text_for_answer(cls, text, prompt):
        cutoff_phrase = prompt
        index = text.rfind(cutoff_phrase)
        if index != -1:
            return text[index+len(prompt):]
        else:
            return text

    @classmethod
    def remove_substring(cls, string, substring):
        return string.replace(substring, "")

    @classmethod
    def generate(cls, text):
        prompt = get_prompt(text)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
            outputs = model.generate(**inputs,
                                     max_new_tokens=512,
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.eos_token_id,
                                     )
            final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            final_outputs = cut_off_text(final_outputs, '</s>')
            final_outputs = remove_substring(final_outputs, prompt)

        return final_outputs#, outputs

    @classmethod
    def parse_text(cls, text):
            wrapped_text = textwrap.fill(text, width=100)
            print(wrapped_text +'\n\n')
            # return assistant_text

if __name__ == "__main__":
    print("starting...")
    
    test_troubleshooting()
       
    
# %cd /home/aurora/repos/lightint/chingmanlib/
# !python llm_app.py    