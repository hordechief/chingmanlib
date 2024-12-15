import abc

from langchain.prompts import PromptTemplate

class ChainInterface(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def create_chain(self, 
                template=None, 
                input_variables=None, 
                has_memory=False,
                streaming=False,
                verbose=False, 
                **kwargs):
        raise NotImplementedError


class BaseChain(ChainInterface):
    def __init__(self, llm, retriever):
        '''
        3 components for qa: A chat model, An embedding model, and a vector store
        '''
        self.llm = llm
        self.retriever = retriever

    def check_input_variables(self,template,input_variables):
        '''
        check whether the input variable are in template
        '''
        for variable in input_variables:
            _variable = "{" + variable + "}"
            # print(_variable)
            if not _variable in template:
                print(f"variable {_variable} is not included in template")
                return False
        return True    
    
    def create_prompt(self, template, default_template=None, input_variables=None):        
        '''
        default template is a dict including template and input_variables
        '''
        if not template:
            assert default_template
            # Using PromptTemplate to initialize require parameter input_variables
            template = default_template["template"]
            input_variables = default_template["input_variables"]
            prompt = PromptTemplate(template=template, input_variables=input_variables)
        else:
            prompt = PromptTemplate.from_template(template) 
            if input_variables and not self.check_input_variables(template,input_variables):
                raise ValueError("input variable is not consistant")
            prompt = PromptTemplate.from_template(template) 
                        
        return prompt
    
