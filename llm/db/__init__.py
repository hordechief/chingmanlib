from langchain_chroma import Chroma

from chingmanlib.llm.db.data_loader import DataLoaderUtils
from chingmanlib.llm.db.text_splitter import DocSplittingUtils
from chingmanlib.llm.db.vector_store import VectorStoreUtils

def create_data_pipeline(data_dir, embeddings, **kwargs):
    '''
    create data pipeline including load, split and embedding
    '''

    # load data from directory
    data_loader_kwargs = kwargs.get("data_loader_kwargs", {"verbose":True})
    data = DataLoaderUtils.directory_loader(data_dir, **data_loader_kwargs)


    # text slitter
    split_kwargs = kwargs.get("split_kwargs", {})
    dsu = DocSplittingUtils(**split_kwargs)
    texts = dsu.r_split(data)
    print(f"text splitted, lenght is {len(texts)}")

    # vector store with Chroma
    vector_store_kwargs = kwargs.get("vector_store_kwargs", {})
    vsu = VectorStoreUtils(Chroma, **vector_store_kwargs)
    retriever = vsu.from_documents(texts, embeddings)
    print(f"vectordb created")

    return retriever