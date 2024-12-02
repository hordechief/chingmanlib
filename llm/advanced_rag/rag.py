try:
    from llama_index import (
        SimpleDirectoryReader, 
        Document,
        ServiceContext,
        VectorStoreIndex, 
        StorageContext, 
        load_index_from_storage
    )
    from llama_index.llms import OpenAI
    from llama_index.indices.postprocessor import SentenceTransformerRerank,MetadataReplacementPostProcessor
    from llama_index.retrievers import AutoMergingRetriever
    from llama_index.query_engine import RetrieverQueryEngine
    from llama_index.response.notebook_utils import display_response
    from llama_index.schema import TextNode, NodeWithScore
    from llama_index import QueryBundle
    from llama_index.node_parser import (
        HierarchicalNodeParser,
        SentenceWindowNodeParser,
        get_leaf_nodes
    )
except: # lamma_index == 0.11.10
    from llama_index.core import (
        SimpleDirectoryReader, 
        Document,
        ServiceContext,
        VectorStoreIndex, 
        StorageContext, 
        load_index_from_storage
    )
    from llama_index.llms.openai import OpenAI
    from llama_index.core.indices.postprocessor import SentenceTransformerRerank,MetadataReplacementPostProcessor
    from llama_index.core.retrievers import AutoMergingRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response.notebook_utils import display_response
    from llama_index.core.schema import TextNode, NodeWithScore
    from llama_index.core import QueryBundle
    from llama_index.core.node_parser import (
        HierarchicalNodeParser,
        SentenceWindowNodeParser,
        get_leaf_nodes
    )
    
import utils    

# pip install trulens-eval==0.18.1 lamma_index==0.11.10 matplotlib==3.9.2 python-dotenv==1.0.1

# docker run -it -p 9000:9000 --name py310 -v "${pwd}:/app" -w /app python:3.10 /bin/bash

# pip install trulens-eval==0.18.1 --use-deprecated=legacy-resolver

'''
ERROR: pip's legacy dependency resolver does not consider dependency conflicts when selecting packages. This behaviour is the source of the following dependency conflicts.
sqlalchemy 2.0.35 requires typing-extensions>=4.6.0, but you'll have typing-extensions 4.5.0 which is incompatible.
ipython 8.27.0 requires typing-extensions>=4.6; python_version < "3.12", but you'll have typing-extensions 4.5.0 which is incompatible.
llama-index-core 0.11.10 requires pydantic<3.0.0,>=2.7.0, but you'll have pydantic 1.10.16 which is incompatible.
openai 1.45.1 requires typing-extensions<5,>=4.11, but you'll have typing-extensions 4.5.0 which is incompatible.
pydantic-core 2.23.3 requires typing-extensions!=4.7.0,>=4.6.0, but you'll have typing-extensions 4.5.0 which is incompatible.
langchain-core 0.3.0 requires pydantic<3.0.0,>=2.5.2; python_full_version < "3.12.4", but you'll have pydantic 1.10.16 which is incompatible.
langchain-core 0.3.0 requires typing-extensions>=4.7, but you'll have typing-extensions 4.5.0 which is incompatible.
langchain 0.3.0 requires pydantic<3.0.0,>=2.7.4, but you'll have pydantic 1.10.16 which is incompatible.
altair 5.4.1 requires typing-extensions>=4.10.0; python_version < "3.13", but you'll have typing-extensions 4.5.0 which is incompatible.
streamlit-aggrid 1.0.5 requires altair<5, but you'll have altair 5.4.1 which is incompatible.
transformers 4.44.2 requires tokenizers<0.20,>=0.19, but you'll have tokenizers 0.20.0 which is incompatible.
'''
    
if __name__ == "__main__":
    import os
    import openai
    openai.api_key = utils.get_openai_api_key()
    
    documents = SimpleDirectoryReader(
        input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
    ).load_data()
    
    print(type(documents), "\n")
    print(len(documents), "\n")
    print(type(documents[0]))
    print(documents[0])