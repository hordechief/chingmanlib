from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, WebBaseLoader

#*******************************************
#
#             Data Loader
#
#******************************************

class DataLoaderUtils():
    def __init__(self):
        pass
    
    @classmethod
    def text_loader(cls,text_dir,**kwargs):
        # Text Loader
        loader = DirectoryLoader(text_dir, **kwargs) # glob="sherlock_homes.txt"
        data = loader.load()
        
        return data
    
    @classmethod
    def web_loader(cls,url):
        loader = WebBaseLoader(url)
        data = loader.load()
        
        return data
    
    @classmethod    
    def pdf_loader(cls, file_path):
        """
        loader.load() return pages list.
        Each page is a Document. A Document contains text (page_content) and metadata.
        page.metadata: {'source': '/binhe/Deeplearning-Notebook-v5.7-comments.pdf', 'page': 0}
        """
        loader = PyPDFLoader(file_path)
        data = loader.load() # return data is pages list. 
        print(f"load pdf file successfull, number of pages: {len(data)}, ")

        # plans = os.listdir("data")
        # doc_titles = [plan.split('.')[0] for plan in plans]
        # docs = {}
        # for doc in doc_titles:
        #     docs[plan]= loader.load_data(file=Path(f"data/{doc}.pdf"))
        
        return data
    
    def hybrid_loader(cls, file_dir):        
        loaders_map = {
            "pdf": PyPDFLoader,
        }
        
        loaders = []
        files = os.listdir(file_dir)
        for file in files:
            loader_func = loaders_map[split('.')[-1]]
            loader = loader_func(Path(f"{file}"))
            loaders.append(loader)
        
        docs = []
        for loader in loaders:
            docs.extend(loader.load())        

    @classmethod
    def directory_loader(cls, file_dir, loader_cls):
        # loader = DirectoryLoader('data/', loader_cls=PyPDFLoader)
        pass

#*******************************************
#
#             Text Splitting
#
#******************************************

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter

class DocSplittingUtils():
    def __init__(self, chunk_size =1000, chunk_overlap = 200, separator = "\n", is_separator_regex = False):

        # text splitter
        self.text_splitter = self.create_text_splitter(chunk_size, chunk_overlap,separator,is_separator_regex)
        
    def create_text_splitter(self, chunk_size, chunk_overlap,separator = "\n",is_separator_regex = False,):
        
        self.text_splitter = CharacterTextSplitter(        
            chunk_size = chunk_size,
            chunk_overlap  = chunk_overlap,
            separator = separator,
            is_separator_regex = is_separator_regex,
            length_function = len,
        )
        
        return self.text_splitter
        
    def split(self, data, verbose=False):
        texts = self.text_splitter.split_documents(data)
        if verbose:
            print(len(texts))
            print(texts[0])
            
        return texts
        
    def test_r_splitter():
        print(r_splitter.split_text(text1))
        # ['abcdefghijklmnopqrstuvwxyz']

    def test_c_splitter(text):
        
        chunk_size =26
        chunk_overlap = 4
        
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        c_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        text1 = 'abcdefghijklmnopqrstuvwxyz'
        print(r_splitter.split_text(text1))
        # ['abcdefghijklmnopqrstuvwxyz']
        
        text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
        print(r_splitter.split_text(text2))
        # ['abcdefghijklmnopqrstuvwxyz', 'wxyzabcdefg']
        
        print(r_splitter.split_text(text))

#*******************************************
#
#             Vector DB
#
#******************************************

# pip install -q faiss-gpu
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

class VectorStoreUtils():
    def __init__(self,class_type):
        self.vectorstore = class_type
        print(self.vectorstore.__name__, class_type.__name__)
        if self.vectorstore.__name__ == Chroma.__name__:
            pass
        elif self.vectorstore.__name__ == FAISS.__name__:
            pass   
        else:
            print("Incorrect Vector Store Class")
            
    def from_documents(self,texts, embeddings):
        # 鸭子函数     
        self.docsearch = self.vectorstore.from_documents(
            texts, 
            embeddings,
            # persist_directory
        )
        
        self.retriever = self.docsearch.as_retriever()

        print(f"vector store {type(self.docsearch.as_retriever())} created... colllection count {self.docsearch._collection.count()}")
        
        return self.retriever
    
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