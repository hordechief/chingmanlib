from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, WebBaseLoader

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
    def directory_loader(cls, file_dir, glob="*.pdf", loader_cls=PyPDFLoader, verbose=False):
        loader = DirectoryLoader(
            file_dir, 
            # glob="poweredge_poweredge-6850_user_guide.pdf",
            glob=glob,
            loader_cls=loader_cls)

        data = loader.load()

        print(f"data page loaded, length is {len(data)}")

        if verbose:
            print("page metadata \n--------------------------")
            for page in data:
                print(page.metadata)

        return data