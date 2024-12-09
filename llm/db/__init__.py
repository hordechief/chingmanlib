from langchain_chroma import Chroma

from chingmanlib.llm.db.data_loader import DataLoaderUtils
from chingmanlib.llm.db.text_splitter import DocSplittingUtils
from chingmanlib.llm.db.vector_store import VectorStoreUtils

def create_data_pipeline(data_dir, embeddings, **kwargs):

    split_kwargs = kwargs.get("split_kwargs", {})

    data = DataLoaderUtils.directory_loader(data_dir)

    dsu = DocSplittingUtils(**split_kwargs)
    texts = dsu.r_split(data)
    print(f"text splitted, lenght is {len(texts)}")

    vsu = VectorStoreUtils(Chroma)
    retriever = vsu.from_documents(texts, embeddings)
    print(f"vectordb created")

    return retriever