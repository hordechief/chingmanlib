# pip install -q faiss-gpu
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings

import os
from chingmanlib.llm import load_envs

load_envs()

class VectorStoreUtils():
    def __init__(self,class_type,**kwargs):
        '''
        '''
        print(f"vector store type is {class_type.__name__}")
        if class_type.__name__ == Chroma.__name__:
            pass
        elif class_type.__name__ == FAISS.__name__:
            pass   
        # elif class_type.__name__ == InMemoryVectorStore.__name__:
        #     pass 
        else:
            raise ValueError("Incorrect Vector Store Class")
                
        vs_kwargs = {}
        if "embedding_function" in kwargs: 
            vs_kwargs["embedding_function"] = kwargs["embedding_function"]
            self.embeddings = kwargs["embedding_function"]
            self.vectorstore = class_type(**vs_kwargs)
        else:
            self.vectorstore = class_type

        # create persist directory for chroma db
        # the default save directory is ~/.chroma
        self.persist_directory = kwargs.get("persist_directory", None)

        print(f"Vector db initialized with type {class_type.__name__}")

    def create_retriever(self):
        # vectordb = Chroma(
        #     persist_directory=persist_directory,
        #     embedding_function=OpenAIEmbeddings()
        # )
        # self.retriever = VectorStoreRetriever(vectorstore=FAISS(...))

        pass
            
    def from_documents(self,documents, embeddings, **kwargs):
        # print(type(self.vectorstore))
        # assert isinstance(self.vectorstore, Chroma) or isinstance(self.vectorstore, FAISS)

        '''
        # 鸭子函数
        '''        
        self.docsearch = self.vectorstore.from_documents(
            documents, 
            embeddings,
            persist_directory = self.persist_directory
        )
        
        retriever_kwargs = {} # search_type="similarity", search_kwargs={"k": 3}
        if "search_type" in kwargs: retriever_kwargs["search_type"] = kwargs["search_type"]
        if "search_kwargs" in kwargs: retriever_kwargs["search_kwargs"] = kwargs["search_kwargs"]
        self.retriever = self.docsearch.as_retriever(**retriever_kwargs)

        print(f"vector store {type(self.docsearch.as_retriever())} created... colllection count {self.docsearch._collection.count()}")
        
        return self.retriever
    
        # vectordb.persist() # In newer versions the documents are automatically persisted.

        # 可选：关闭连接（虽然不是必须的，方便资源释放）
        # vectordb.close()
    
    def from_text(self,texts, embeddings, **kwargs):
        '''
        '''
        # text list - e.g.. ["harrison worked at kensho"]
        self.docsearch = self.vectorstore.from_texts(
            texts, 
            embeddings,
            persist_directory = self.persist_directory
        )
        
        retriever_kwargs = {} # search_type="similarity", search_kwargs={"k": 3}
        if "search_type" in kwargs: retriever_kwargs["search_type"] = kwargs["search_type"]
        if "search_kwargs" in kwargs: retriever_kwargs["search_kwargs"] = kwargs["search_kwargs"]        
        self.retriever = self.docsearch.as_retriever(**retriever_kwargs)

        print(f"vector store {type(self.docsearch.as_retriever())} created... colllection count {self.docsearch._collection.count()}")
        
        return self.retriever
    
    def run(self, question, verbose=True):
        retrieved_docs = self.retriever.invoke(question)
        if verbose: print(f"retrieved documents is {retrieved_docs}")
        return retrieved_docs
    
    def similarity_search(self, question, k=3, verbose=False):

        assert self.docsearch

        docs = self.docsearch.similarity_search(question, k=k)
        if verbose: 
            print(f"docs len is {len(docs)}, page_count of first doc is '{docs[0].page_content}'\n")
            print(docs[0])
            for doc in docs:
                print(f"doc metadata is {doc.metadata}") 
                print(f"doc page content is: {doc.page_content}")     

        # self.docsearch.persist()
        
        return docs
    
    @staticmethod
    def calculate_cosine_similarity(embeddings, query, docs):
        from langchain_community.utils.math import cosine_similarity
        def _cosine_similarity(vec1, vec2):
            import numpy as np
            return np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        doc_contents = [doc.page_content for doc in docs]
        query_embedding = embeddings.embed_query(query)
        doc_embedding = embeddings.embed_documents(doc_contents)

        similarity = cosine_similarity([query_embedding], doc_embedding)[0]
        most_similar = doc_contents[similarity.argmax()]

        return similarity

    def similarity_search_with_score(self, question, k=3, verbose=False):

        assert self.docsearch
        
        docs = self.docsearch.similarity_search_with_score(question, k=k)
        if verbose: 
            for doc in docs:
                print(doc)
                # print(f"doc metadata is {doc[0].metadata}") 
                # print(f"doc page content is: {doc[0].page_content}")     
                # print(f"doc Cosine similarity_score is: {doc[1]}")      
        
        return docs

class InMemoryVectorStoreUtils(VectorStoreUtils):
    def __init__(self, embeddings, **kwargs):
        self.vectorstore = InMemoryVectorStore(embeddings)
        
    def add_documents(self, documents, **retriever_kwargs):
        self.vectorstore.add_documents(documents=documents)
        self.retriever = self.vectorstore.as_retriever(**retriever_kwargs)

    @staticmethod
    def retrieve_relevant_docs(question, texts, embeddings):    

        vectorstore = DocArrayInMemorySearch.from_texts(texts,embedding=embeddings)
        retriever = vectorstore.as_retriever()
        response = retriever.get_relevant_documents(question)

        print(response)

if __name__ == "__main__":

    texts = [
            "harrison worked at kensho", 
            "bears like to eat honey"]
    question = "where did harrison work?"
    question = "what does bears like to eat?"
    embeddings = OpenAIEmbeddings()

    vsu = VectorStoreUtils(Chroma)
    vsu.from_text(texts, embeddings)    
    vsu.similarity_search_with_score(question, k=1, verbose=True)
    vsu.similarity_search(question, k=1, verbose=True)

    # VectorStoreUtils.test_retriever_get_relevant_docs(texts, question, embeddings)    

    # python -m chingmanlib.llm.db.vector_store