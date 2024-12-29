from langchain_chroma import Chroma

from chingmanlib.llm.db.data_loader import DataLoaderUtils
from chingmanlib.llm.db.text_splitter import DocSplittingUtils
from chingmanlib.llm.db.vector_store import VectorStoreUtils
import os

def create_data_pipeline(data_dir_or_path, embeddings, **kwargs):
    '''
    create data pipeline including load, split and embedding
    '''

    # if kwargs.get("use_exist_db", False) == True:
    #     vector_store_kwargs = kwargs.get("vector_store_kwargs", {})
    #     if "VDB_CLS" in vector_store_kwargs:
    #         vdb_cls = vector_store_kwargs.pop("VDB_CLS")
    #     else:
    #         vdb_cls = Chroma         
    #     return VectorStoreUtils(vdb_cls, **vector_store_kwargs)

    # load data from directory
    data_loader_kwargs = kwargs.get("data_loader_kwargs", {"verbose":True})

    if os.path.isdir(data_dir_or_path):
        data = DataLoaderUtils.directory_loader(data_dir_or_path, **data_loader_kwargs)
    elif os.path.isfile(data_dir_or_path):
        if data_dir_or_path.endswith(".pdf"):
            data = DataLoaderUtils.pdf_loader(data_dir_or_path)
        else:
            raise ValueError("Not Implemented")

    # text slitter
    split_kwargs = kwargs.get("split_kwargs", {})
    dsu = DocSplittingUtils(**split_kwargs)
    texts = dsu.r_split(data)
    print(f"text splitted, lenght is {len(texts)}")

    # vector store with Chroma
    vector_store_kwargs = kwargs.get("vector_store_kwargs", {})
    if "VDB_CLS" in vector_store_kwargs:
        vdb_cls = vector_store_kwargs.pop("VDB_CLS")
    else:
        vdb_cls = Chroma
    vsu = VectorStoreUtils(vdb_cls, **vector_store_kwargs)
    retriever = vsu.from_documents(texts, embeddings)
    print(f"vectordb created")

    return vsu

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_vector_store(db_data_dir, embeddings):
    # Load and chunk contents of the blog
    loader = DirectoryLoader(db_data_dir, glob="*.pdf",loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"data page loaded, length is {len(docs)}")
    print("page metadata \n--------------------------")
    for page in docs:
        print(page.metadata)
    # Loading Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # Splitting Text into Chunks
    # chunks = text_splitter.split_text(texts)
    chunks = text_splitter.split_documents(docs)
    # Creating a Vector Store (Chroma) from Text
    # vector_store = Chroma.from_texts(chunks, embeddings)
    vector_store = Chroma.from_documents(chunks, embeddings)

    # Creating a Retriever
    # retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # return retriever

    return vector_store