import json
import textwrap
from .templates import *

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""
class PromptUtils():
    def __init__():
        pass

    @classmethod
    def get_prompt(cls, instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
        '''
        this methond combines system prompt and instruction to generate the prompt
        '''
        SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
        prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
        return prompt_template

    @classmethod    
    def cut_off_text(cls, text, prompt):
        '''
        cut off the text after the prompt
        '''
        cutoff_phrase = prompt
        index = text.find(cutoff_phrase)
        if index != -1:
            return text[:index]
        else:
            return text

    @classmethod
    def cut_off_text_for_answer(cls, text, prompt):
        cutoff_phrase = prompt
        index = text.rfind(cutoff_phrase)
        if index != -1:
            return text[index+len(prompt):]
        else:
            return text

    @classmethod
    def remove_substring(cls, string, substring):
        return string.replace(substring, "")

    @classmethod
    def parse_text(cls, text):
            wrapped_text = textwrap.fill(text, width=100)
            print(wrapped_text +'\n\n')
            # return assistant_text

