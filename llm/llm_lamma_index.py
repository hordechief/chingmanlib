# !pip install -q llama-index-embeddings-huggingface llama-index-llms-fireworks
# %pip install llama-index-llms-openai

  
from llama_index.llms.openai import OpenAI as LLAMA_INDEX_OpenAI
    
class LlamaIndexUtils():
    def __init__(self,**kwargs):
        
        pass
    
    ##############################################################################
    # LLM
    def create_LLM(self):
        # https://docs.llamaindex.ai/en/stable/examples/llm/openai/
        self.llm = LLAMA_INDEX_OpenAI()
        return self.llm
    
    def test_LLM(self,llm):
        if not isinstance(llm,LLAMA_INDEX_OpenAI):
            raise ValueError("llm instance is not from llama_index.llms.openai.OpenAI")
        resp = llm.complete("Paul Graham is ")
        print(resp)

    ##############################################################################
    # Embeddings

    def get_embed_model1(self, model_name,model_kwargs,encode_kwargs,cache_folder):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding as LIEH_HuggingFaceEmbedding
        embed_model = LIEH_HuggingFaceEmbedding(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder = cache_folder
        )
        return embed_model   
        # SentenceTransformer TypeError: __init__() got an unexpected keyword argument 'trust_remote_code'
        
    @property
    def embeddings(self):
        return self._embeddings
    
    @embeddings.setter
    def embeddings(self,embeddings):
        if embeddings:
            self._embeddings = embeddings
        else:
            raise ValueError("Value must be non-negative")
    
    #############################################################################
    # data loader
    def load_data(self, directory_path):
        # Data ingestion: load all files from a directory
        from llama_index.core import SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_dir=directory_path)
        self.documents = reader.load_data()

        return self.documents
    
    ########################################################################
    # document split
    def doc_split(self,documents, **kwargs):
        show_progress = kwargs.get("show_progress",True)
        split_kwargs = kwargs.get("split_kwargs",dict(chunk_size=1024, chunk_overlap=200))

        from llama_index.core.node_parser import SentenceSplitter

        # Split the documents into nodes
        self.text_splitter = SentenceSplitter(**split_kwargs)
        self.nodes = self.text_splitter.get_nodes_from_documents(documents, show_progress=show_progress)

        return self.nodes    
    
    ###########################################################################
    # service context
    def get_service_context(self,llm,embed_model):    
        # Create service context
        from llama_index.core import ServiceContext
        self.service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
        return self.service_context
        
    ###########################################################################
    # indexing
    
    def indexing(self,service_context,documents,nodes,persist_dir):

        if not persist_dir:
            persist_dir = "./db/storage_mini"

        from llama_index.core import (
            VectorStoreIndex,
            StorageContext, 
            load_index_from_storage
        )
        
        # Create and persist the vector store index
        vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, service_context=service_context, node_parser=nodes)
        vector_index.storage_context.persist(persist_dir=persist_dir)

        # Load the index from storage
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        
        self.index = load_index_from_storage(storage_context, service_context=service_context)

        return self.index    