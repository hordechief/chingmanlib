from langchain_core.prompts import ChatPromptTemplate
# from langchain.prompts import (
#     PromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     ChatPromptTemplate,
# )

class PromptBasic():
    def __init__(self):
        pass

    @classmethod
    def test_prompt_invoke(cls):
        prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
        prompt_value = prompt.invoke({"topic": "ice cream"})
        print(prompt_value.to_messages())
        print(prompt_value.to_string())

if __name__ == "__main__":
    PromptBasic.test_prompt_invoke()

    # export PYTHONPATH=/home/aurora/repos/dl-ex/submodules:$PYTHONPATH
    # python -m submodules.chingmanlib.llm.prompts.basic        