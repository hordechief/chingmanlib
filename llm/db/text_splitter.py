from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter

class DocSplittingUtils():
    def __init__(self, **kwargs):

        default_kwargs = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separator": "\n",
            "is_separator_regex": False,
        }

        chunk_size = kwargs.get("chunk_size", 1000)
        chunk_overlap = kwargs.get("chunk_overlap", 200)
        separator = kwargs.get("separator", '\n')
        is_separator_regex = kwargs.get("is_separator_regex", False)

        # text splitter
        self.c_text_splitter = CharacterTextSplitter(        
            chunk_size = chunk_size,
            chunk_overlap  = chunk_overlap,
            separator = separator,
            is_separator_regex = is_separator_regex,
            length_function = len,
        )

        self.r_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )         
        
    def c_split(self, data, verbose=False):
        texts = self.c_text_splitter.split_documents(data)
        if verbose:
            print(len(texts))
            print(texts[0])
            
        return texts

    def r_split(self, data, verbose=False):
        texts = self.r_text_splitter.split_documents(data)        
        
        if verbose:
            print(len(texts))
            print(texts[0])
            
        return texts
                         

if __name__ == "__main__":
    dsu = DocSplittingUtils(chunk_size =26, chunk_overlap = 4)

    text1 = 'abcdefghijklmnopqrstuvwxyz'
    print(dsu.r_splitter.split_text(text1))
    # ['abcdefghijklmnopqrstuvwxyz']
    
    text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
    print(dsu.r_splitter.split_text(text2))
    # ['abcdefghijklmnopqrstuvwxyz', 'wxyzabcdefg']    

    print(dsu.r_splitter.split_text(text1))
    # ['abcdefghijklmnopqrstuvwxyz']    