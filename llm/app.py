# https://python.langchain.com.cn/docs/expression_language/get_started
# RunnablePassthrough https://python.langchain.com.cn/docs/expression_language/how_to/passthrough
# RunnableParallel=RunnableMap: https://python.langchain.com.cn/docs/expression_language/how_to/map
# https://python.langchain.com.cn/docs/expression_language/cookbook/retrieval
# https://python.langchain.com/v0.1/docs/expression_language/primitives/assign/
# The RunnablePassthrough.assign(...) static method takes an input value and adds the extra arguments passed to the assign function. This is useful when additively creating a dictionary to use as input to a later step, which is a common LCEL pattern.

# import sys
# sys.path.append(lib_dir)

# import executor of LLM models
# from submodules.chingmanlib.llm.models.hf import LLMHFxecutor
# from submodules.chingmanlib.llm.models.llamacpp import LlamaCppExecutor
# from chingmanlib.llm.models.hf import LLMHFxecutor
# from chingmanlib.llm.models.llamacpp import LlamaCppExecutor
from .models.hf import LLMHFxecutor
from .models.llamacpp import LlamaCppExecutor
from .models.llm_factory import LLMFactory


from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai.chat_models import ChatOpenAI  

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
    def set_vsu(self, vdb):
        self.vsu = VectorStoreUtils(vdb)
        
    def set_retriever(self,texts):
        self.retriever = self.vsu.from_documents(texts, self.embeddings)
        
    def get_retriever(self):
        return self.retriever

    ##############################################################
    # Text Splitting
    def set_dsu(self, **split_param):
        self.dsu = DocSplittingUtils(**split_param)
        
    def split_docs(self,data):        
        self.texts = self.dsu.split(data)
    
    def get_split_docs(self):
        return self.texts
    
    #############################################################
    #
    # LLM run test
    def run_basic_test(self):
        response = self.llm("Translate this sentence from English to French. I love programming.")
        return response, AIMessage(response)
        
    def run_test_troubleshooting(self):
        from .prompts.templates import PC_troubleshooting_promtps
        system_prompt = PC_troubleshooting_promtps["system_prompt"] 
        question = PC_troubleshooting_promtps["question"]

        if isinstance(self.llm, ChatOpenAI):
            completion = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "How to solve the issue of PCIe Adapter cannot be recognized?"}
            ]
            )
            response = completion.choices[0].message
        else:            
            response = self.llm(system_prompt + question)
        return response

if __name__ == "__main__":
    print("starting...")
    
    llm_executor_app = LLMExecutorApp
    llm_executor_app.run_test_troubleshooting()
       
    
# %cd /home/aurora/repos/lightint/chingmanlib/
# !python llm_app.py    
# python -m submodules.chingmanlib.app