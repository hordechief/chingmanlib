import abc
import os
from langchain.prompts import PromptTemplate

LANGUAGE = os.environ.get("TEMPLATE_LANGUAGE", 'cn')
from chingmanlib.llm.prompts.templates import LLM_TEMPLATES

class ChainInterface(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def create_chain(self, 
                prompt=None,
                streaming=False,
                verbose=False, 
                **kwargs):
        
        template = kwargs.get("template", None)
        input_variables = kwargs.get("input_variables", None)

        raise NotImplementedError


class BaseChain(ChainInterface):
    def __init__(self, llm, retriever, *args, **kwargs):
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
    
    def create_prompt(self, template, input_variables=None, default_template_name=None):        
        '''
        default template is a dict including template and input_variables
        '''
        if not template:
            assert default_template_name in LLM_TEMPLATES.keys()
            # Using PromptTemplate to initialize require parameter input_variables
            template = default_template=LLM_TEMPLATES[default_template_name]["template"][LANGUAGE] 
            input_variables = LLM_TEMPLATES[default_template_name]["input_variables"]
            prompt = PromptTemplate(template=template, input_variables=input_variables)
        else:
            prompt = PromptTemplate.from_template(template) 
            if input_variables and not self.check_input_variables(template,input_variables):
                raise ValueError("input variable is not consistant")
            prompt = PromptTemplate.from_template(template) 
                        
        return prompt
    
    def run(self, question, verbose=False):
        raise NotImplemented

    def steam(self, question, verbose=False):
        raise NotImplemented
