# https://dev.to/mohsin_rashid_13537f11a91/rag-with-ollama-1049
# https://ollama.com/
# https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html

from typing_extensions import List, TypedDict
import os
from dotenv import load_dotenv
from langchain import hub
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, format_document, MessagesPlaceholder

# pip install python-dotenv
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件
# from pathlib import Path
# env_path = Path(__file__).parent.parent / ".env"
# assert os.path.exists(env_path)
# load_dotenv(dotenv_path=str(env_path), override=True)  # 加载 .env 文件    

from chingmanlib.llm.models import create_llm
from chingmanlib.llm.db import create_data_pipeline
from chingmanlib.llm.db.vector_store import VectorStoreUtils

# Loading The LLM (Language Model)
llm = Ollama(
    # model="llama3.1", 
    model="EntropyYue/chatglm3:6b",
    base_url="http://127.0.0.1:11434")

llm_executor = create_llm("openai")
llm = llm_executor.llm

# Setting Ollama Embeddings
embeddings = OllamaEmbeddings(
    model="llama3.2",
    # model="EntropyYue/chatglm3:6b",
    # model="llama3",
    base_url='http://127.0.0.1:11434'
)

embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v2-base-zh"
    # sentence-transformers/all-mpnet-base-v2
    ) 

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
    "split_kwargs": split_kwargs
}

kb_data_dir = os.getenv("KB_FOLDER", "/home/aurora/repos/dl-ex/llm/data") 
vsu = create_data_pipeline(kb_data_dir, embeddings, **kwargs)
retriever = vsu.retriever
# retriever = get_retriever(kb_data_dir).retriever

# rag_prompt = hub.pull("rlm/rag-prompt")
# rag_prompt = hub.pull("zbgd/rag-prompt")
'''
你是问答任务的助手。 使用以下检索到的上下文来回答问题。 如果你不知道答案，就说你不知道。 最多使用三个句子并保持答案简洁。
Question: {question} 
Context: {context} 
Answer:
'''

rag_template = hub.pull("yusiwen/rag-prompt-qwen")
'''
<s> [INST] 你是一位专门处理问答任务的助手。请根据以下已检索的上下文信息来回答问题。如果你不知道如何回答，请直接回复“我不知道。” 请控制回答的长度在5句以内，并保持尽量简洁。 [/INST] </s> 

[INST] 问题: {question} 

上下文: {context} 

回答: [/INST]
'''

rag_template = '''
<s> [INST] 你是一位专门处理问答任务的助手。请根据以下已检索的上下文信息来回答问题。如果上下文信息里不包含问题相关信息，那么请基于你的现有知识回答。请控制回答的长度在5句以内，并保持尽量简洁。 [/INST] </s> 

[INST] 问题: {question} 

上下文: {context} 

指示: 如果上下文内容无法回答问题，基于您自身的知识补充完整答案。

回答: [/INST]
'''

rag_prompt = PromptTemplate.from_template(rag_template)

def retrieve(question):
    retrieved_docs = retriever.invoke(question)
    print(f"retrieved documents is {retrieved_docs}")

    retrieved_docs = vsu.run(question) # call retriever.invoke
    print(f"retrieved documents is {retrieved_docs}")

    retrieved_docs = vsu.similarity_search_with_score(prompt, k=3)
    print(f"retrieved documents is {retrieved_docs}") 

    return retrieved_docs


def generate(question, retrieved_docs):
    docs_content = "\n\n".join(doc[0].page_content if isinstance(doc, tuple) else doc.page_content for doc in retrieved_docs)
    messages = rag_prompt.invoke({"question": question, "context": docs_content}) # ChatPromptValue
    return messages.to_string()

def compose_retrieval_chain(llm, retriever, promt=None):
    # Creating a Retrieval Chain
    # chain = create_retrieval_chain(combine_docs_chain=llm, retriever=retriever) # Error?

    # Retrieval-QA Chat Prompt
    # https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    retrieval_qa_chat_prompt = promt or hub.pull("langchain-ai/retrieval-qa-chat")
    # retrieval_qa_chat_prompt_raw = ChatPromptTemplate.from_messages([
    #     ("system", '''
    #         Answer any use questions based solely on the context below:

    #         <context>
    #         {context}
    #         </context>
    #         '''),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("human", "{input}"),
    # ])
    print(retrieval_qa_chat_prompt)

    # Combining Documents
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    # https://python.langchain.com/v0.1/docs/modules/chains/
    # This chain takes a list of documents and formats them all into a prompt, then passes that prompt to an LLM. 
    # It passes ALL documents, so you should make sure it fits within the context window of the LLM you are using.

    # Final Retrieval Chain
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)    
    # This chain takes in a user inquiry, which is then passed to the retriever to fetch relevant documents. 
    # Those documents (and original inputs) are then passed to an LLM to generate a response

    return retrieval_chain


if __name__ == "__main__":
    import sys
    sys.path.append("/home/aurora/repos/CTE-LLM")
    from chingmanlib.llm.utils.log import LOG
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template('''
Given the following question and context, please evaluate their relevance. Only return one of the following three answers:
- Highly Relevant
- Relevant
- Not Relevant

Question: {input}

Context: {context}

Relevance Evaluation:
''',)
    retrieval_chain = compose_retrieval_chain(llm,retriever,retrieval_qa_chat_prompt)
    while True:
        print("\n>> (Press Ctrl+C to exit.)")
        prompt = input(">> ")

        # Invoking the Retrieval DB
        retrieved_docs = retrieve(prompt)
        LOG.log("Retrieved document is:", retrieved_docs)

        # Invoking the Prompt
        # prompt = generate(prompt, retrieved_docs)
        # LOG.log("New Prompt is:", prompt)

        # Invoking the Retrieval Chain
        response = retrieval_chain.invoke({"input": prompt})
        LOG.log("Retrieval chain: ", response['answer'])
