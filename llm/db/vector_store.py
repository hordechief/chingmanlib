# pip install -q faiss-gpu
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
import os

class VectorStoreUtils():
    def __init__(self,class_type):
        self.vectorstore = class_type
        print(self.vectorstore.__name__, class_type.__name__)
        if self.vectorstore.__name__ == Chroma.__name__:
            pass
        elif self.vectorstore.__name__ == FAISS.__name__:
            pass   
        else:
            raise ValueError("Incorrect Vector Store Class")

        # create persist directory for chroma db
        # self.persist_directory = os.getenv("CHATBOT_PERSIST_DIRECTORY", './persist_directory/')
        self.persist_directory = os.getenv("CHATBOT_PERSIST_DIRECTORY", None)
        os.makedirs(self.persist_directory, exist_ok=True)

    def create_vectordb(self):
        # vectordb = Chroma(
        #     persist_directory=persist_directory,
        #     embedding_function=OpenAIEmbeddings()
        # )
        pass
            
    def from_documents(self,texts, embeddings):
        # 鸭子函数     
        self.docsearch = self.vectorstore.from_documents(
            texts, 
            embeddings,
            persist_directory = self.persist_directory
        )
        
        self.retriever = self.docsearch.as_retriever()

        print(f"vector store {type(self.docsearch.as_retriever())} created... colllection count {self.docsearch._collection.count()}")
        
        return self.retriever
    
        # vectordb = Chroma.from_documents(
        #     # data, # use raw data from_text
        #     documents=splits, # use splits
        #     embedding=OpenAIEmbeddings(), 
        #     persist_directory = persist_directory)

        # vectordb.persist() # In newer versions the documents are automatically persisted.
    
    def from_text(self,texts, embeddings):
        # text list - e.g.. ["harrison worked at kensho"]
        self.docsearch = self.vectorstore.from_texts(
            texts, 
            embeddings,
            # persist_directory
        )
        
        self.retriever = self.docsearch.as_retriever()

        print(f"vector store {type(self.docsearch.as_retriever())} created... colllection count {self.docsearch._collection.count()}")
        
        return self.retriever
    
    def similarity_search(self, question, k=3):
        docs = self.docsearch.similarity_search(question,k=k)
        print(f"docs len is {len(docs)}, page_count of first doc is {docs[0].page_content}\n\n")
        for doc in docs:
            print(doc.metadata)        
        
        # self.docsearch.persist()
        
        return docs


if __name__ == "__main__":
    pass    