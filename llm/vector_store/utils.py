def call_data_pipeline(**kwargs):
    split_kwargs = kwargs.get("split_kwargs", None)
    if not split_kwargs:
        split_kwargs = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separator": "\n",
            "is_separator_regex": False
        }
    
    from submodules.chingmanlib.llm.pipeline import DataLoaderUtils
    dlu = DataLoaderUtils()  
    
    from llm.pipeline import DocSplittingUtils
    dsu = DocSplittingUtils(**kwargs)
    
    return dlu, dsu