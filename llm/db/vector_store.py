# pip install -q faiss-gpu
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch

from langchain_openai import OpenAIEmbeddings

import os

class VectorStoreUtils():
    def __init__(self,class_type,**kwargs):
        '''
        '''
        self.vectorstore = class_type
        print(f"vector store type is {class_type.__name__}")
        if self.vectorstore.__name__ == Chroma.__name__:
            pass
        elif self.vectorstore.__name__ == FAISS.__name__:
            pass   
        else:
            raise ValueError("Incorrect Vector Store Class")

        # create persist directory for chroma db
        # the default save directory is ~/.chroma
        self.persist_directory = kwargs.get("persist_directory", None)

        print("Vector db initialized with type {class_type.__name__}")

    def create_vectordb(self):
        # vectordb = Chroma(
        #     persist_directory=persist_directory,
        #     embedding_function=OpenAIEmbeddings()
        # )
        pass
            
    def from_documents(self,texts, embeddings):
        '''
        # 鸭子函数
        '''        
        self.docsearch = self.vectorstore.from_documents(
            texts, 
            embeddings,
            persist_directory = self.persist_directory
        )
        
        self.retriever = self.docsearch.as_retriever()

        print(f"vector store {type(self.docsearch.as_retriever())} created... colllection count {self.docsearch._collection.count()}")
        
        return self.retriever
    
        # vectordb.persist() # In newer versions the documents are automatically persisted.
    
    def from_text(self,texts, embeddings):
        '''
        '''
        # text list - e.g.. ["harrison worked at kensho"]
        self.docsearch = self.vectorstore.from_texts(
            texts, 
            embeddings,
            persist_directory = self.persist_directory
        )
        
        self.retriever = self.docsearch.as_retriever()

        print(f"vector store {type(self.docsearch.as_retriever())} created... colllection count {self.docsearch._collection.count()}")
        
        return self.retriever
    
    def test_similarity_search(self, question, k=3):
        assert self.docsearch

        docs = self.docsearch.similarity_search(question, k=k)
        print(f"docs len is {len(docs)}, page_count of first doc is '{docs[0].page_content}'\n")
        for doc in docs:
            print(f"doc metadata is {doc.metadata}") 
            print(f"doc page content is: {doc.page_content}")          
        # self.docsearch.persist()
        
        return docs

    @staticmethod
    def test_retriever_get_relevant_docs(texts, question, embeddings):    

        vectorstore = DocArrayInMemorySearch.from_texts(texts,embedding=embeddings)
        retriever = vectorstore.as_retriever()
        response = retriever.get_relevant_documents(question)
        print(response)


if __name__ == "__main__":

    texts = [
            "harrison worked at kensho", 
            "bears like to eat honey"]
    question = "where did harrison work?"
    embeddings = OpenAIEmbeddings()

    vsu = VectorStoreUtils(Chroma)
    vsu.from_text(texts, embeddings)    
    vsu.test_similarity_search(question, k=1)

    VectorStoreUtils.test_retriever_get_relevant_docs(texts, question, embeddings)    

    # python -m chingmanlib.llm.db.vector_store