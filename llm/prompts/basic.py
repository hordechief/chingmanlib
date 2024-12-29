from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    format_document,
    MessagesPlaceholder
)

# from langchain.prompts import (
#     PromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     ChatPromptTemplate,
# )
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     AIMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage
# )

import textwrap

class PromptBasic():
    def __init__(self):
        pass

    @classmethod
    def test_prompt_invoke(cls):
        prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
        prompt_value = prompt.invoke({"topic": "ice cream"})
        print(prompt_value.to_messages())
        # [HumanMessage(content='tell me a short joke about ice cream', additional_kwargs={}, response_metadata={})]
        print(prompt_value.to_string())
        # Human: tell me a short joke about ice cream

if __name__ == "__main__":
    PromptBasic.test_prompt_invoke()

    ## TEST PROMPT
    prompt = PromptTemplate.from_template("tell me a joke about {topic}")
    # print(prompt)
    # input_variables=['topic'] input_types={} partial_variables={} 
    # template='tell me a joke about {topic}'

    prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
    # print(prompt)
    # input_variables=['topic'] input_types={} partial_variables={} 
    # message =[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['topic'], input_types={}, partial_variables={}, 
    # template='tell me a joke about {topic}'), additional_kwargs={})]

    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful chatbot"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
    )
    # print(prompt)
    # input_variables=['history', 'input'] 
    # input_types={'history': list[typing.Annotated[typing.Union[typing.Annotated[langchain_core.messages.ai.AIMessage, 
    # Tag(tag='ai')], 
    # typing.Annotated[langchain_core.messages.human.HumanMessage, Tag(tag='human')], 
    # typing.Annotated[langchain_core.messages.chat.ChatMessage, Tag(tag='chat')], 
    # typing.Annotated[langchain_core.messages.system.SystemMessage, Tag(tag='system')], 
    # typing.Annotated[langchain_core.messages.function.FunctionMessage, Tag(tag='function')], 
    # typing.Annotated[langchain_core.messages.tool.ToolMessage, Tag(tag='tool')], 
    # typing.Annotated[langchain_core.messages.ai.AIMessageChunk, Tag(tag='AIMessageChunk')], 
    # typing.Annotated[langchain_core.messages.human.HumanMessageChunk, Tag(tag='HumanMessageChunk')], 
    # typing.Annotated[langchain_core.messages.chat.ChatMessageChunk, Tag(tag='ChatMessageChunk')], 
    # typing.Annotated[langchain_core.messages.system.SystemMessageChunk, Tag(tag='SystemMessageChunk')], 
    # typing.Annotated[langchain_core.messages.function.FunctionMessageChunk, Tag(tag='FunctionMessageChunk')], 
    # typing.Annotated[langchain_core.messages.tool.ToolMessageChunk, Tag(tag='ToolMessageChunk')]], 
    # FieldInfo(annotation=NoneType, required=True, discriminator=Discriminator(discriminator=<function _get_type at 0x707f618584c0>, custom_error_type=None, custom_error_message=None, custom_error_context=None))]]} partial_variables={} 
    # messages=[
    # SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='You are a helpful chatbot'), additional_kwargs={}), 
    # MessagesPlaceholder(variable_name='history'), 
    # HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], input_types={}, partial_variables={}, template='{input}'), additional_kwargs={})]
    
    prompt = ChatPromptTemplate.from_template(textwrap.dedent(
        """Your job is to use patient ...

        {context}

        {question}
        """))
    # input_variables=['context', 'question'] input_types={} partial_variables={} 
    # messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, 
    # template="Your job is to use patient ...\n        {context}\n\n        {question}\n"), additional_kwargs={})]
    
    # print(prompt)

    prompt = ChatPromptTemplate.from_template(textwrap.dedent(
        """
        [INST] <<SYS>>
            You are a helpful ...
            <</SYS>>

            # Chat History:
            {chat_history} 

            # Question: {question}

            # Answer: [/INST]
        """))
    # print(prompt)

    # input_variables=['chat_history', 'question'] input_types={} partial_variables={} 
    # messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['chat_history', 'question'], input_types={}, partial_variables={}, 
    # template='\n[INST] <<SYS>>\n    You are a helpful ...\n    <</SYS>>\n\n    # Chat History:\n    {chat_history} \n\n    # Question: {question}\n\n    # Answer: [/INST]\n'), additional_kwargs={})]

    
    # export PYTHONPATH=/home/aurora/repos/dl-ex/submodules:$PYTHONPATH
    # python -m chingmanlib.llm.prompts.basic        