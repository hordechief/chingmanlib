PC_troubleshooting_promtps = {
    "system_prompt": """
The flowchart outlines a troubleshooting process for hardware or configuration issues, particularly focusing on PCIe and related components. The flow is organized into several branches based on the type of issue encountered. Here’s the detailed description:

Start: The process begins with identifying the problem related to the troubleshooting steps outlined in the flowchart.

Step 1: Retrieve detailed error information.

Branch 1: If NDC/Mini Perc reports Bus:Device error.
Step 1.1: Perform compatibility test on the customer's system.
Branch 1.1.1: If the test passes.
Step 1.1.1.1: Follow specific customer escalation process if needed.
Branch 1.1.2: If the test fails.
Step 1.1.2.1: Check support documentation or escalate according to procedure.
Branch 2: If HBA/NIC/NDC only has one port recognized, while others are functioning.
Step 2.1: Isolate the issue to the specific PCIe card.
Step 2.1.1: Follow the specific steps for PCIe card troubleshooting.
Branch 3: If the PCIe Adapter cannot be recognized.
Step 3.1: Isolate the issue to the specific PCIe Adapter card.
Step 3.1.1: Check DIMM slot configurations and compatibility with MB (Motherboard).
Step 3.1.2: Follow support procedures to resolve the issue.
Branch 4: If other errors cannot be resolved.
Step 4.1: Analyze and perform additional tests to narrow down the issue.
Step 4.1.1: If the issue persists with multiple devices.
Step 4.1.1.1: Escalate the issue as it might be complex.
Step 4.1.2: If the issue persists with a single device.
Step 4.1.2.1: Perform compatibility test on the customer's system.
Step 4.1.2.1.1: If the test passes.
Step 4.1.2.1.1.1: Follow specific customer escalation process if needed.
Step 4.1.2.1.2: If the test fails.
Step 4.1.2.1.2.1: Check support documentation or escalate according to procedure.
""",
    "question": """
please troubleshooting on below issue step by step, and give details of each step.

Question: How to solve the issue of PCIe Adapter cannot be recognized? what's the potential error
"""
}

# 如果您不想手动指定 input_variables，也可以使用 from_template 类方法创建 PromptTemplate。
# LangChain 将根据传递的 template 自动推断 input_variables。
# https://python.langchain.com.cn/docs/modules/model_io/prompts/prompt_templates

LLM_TEMPLATES ={
    # default chat template
    "CHATBOT_TEMPLATE" :{
        "template":{
            "en": """
                This is the Chatbot sytem ..."
                Let's work this out in a step by step way to be sure we have the right answer.
                Question: {question}
                Answer: """, 
                            "cn": """
                这是一个聊天机器人系统……
                让我们一步一步地解决这个问题，以确保我们得到正确的答案。
                问题：{question}
                答案： """, 
        },
        "input_variables":["question"]
    },
    # chat history template with [INST] [SYS] template
    "CHATBOT_CONVERSATION_TEMPLATE_LLAMA" :{
        "template":{
            "en": """
                [INST] <<SYS>>
                You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
                Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
                Please ensure that your responses are socially unbiased and positive in nature.
                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
                If you don't know the answer to a question, please don't share false information.
                You always only answer for the assistant then you stop, don't add greet and ask back question, read the chat history to get context.
                <</SYS>>

                # Chat History:
                {history} 

                # Question: {input}

                # Answer: [/INST]
                """, 
            "cn": """
                [INST]<<SYS>>
                你是一个乐于助人的助手，你总是只为助手提供答案，然后停止，不要添加问候语或反问问题，阅读聊天记录以获取上下文信息。
                <</SYS>>

                # 聊天记录：
                {history}

                # 问题：{input}

                # 回答[/INST]""", 
        },
        "input_variables":["chat_history", "question"]
    },    

    #################################################
    # chat history template with RAG
    "CHATBOT_CONVERSATION_TEMPLATE" :{
        "template":{
            "en": """
                The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
                If the AI does not know the answer to a question, it truthfully says it does not know.

                Current conversation:
                {history}

                Human: {input}

                AI Assistant:""", 
                            "cn": """
                以下是人类和人工智能之间的友好对话。人工智能非常健谈，并提供许多具体的上下文细节。
                如果人工智能不知道问题的答案，它会如实地说自己不知道。

                当前对话：
                {history}

                人类：{input}

                AI Assistant:"
                """, 
        },
        "input_variables":["history", "input"]
    },        
    # #################################################
    "CHATBOT_RAG_TEMPLATE_LLAMA" :{
        "template":{
            "en": """
                [INST] <<SYS>>
                You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
                Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
                Please ensure that your responses are socially unbiased and positive in nature.

                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
                If you don't know the answer to a question, please don't share false information.
                <</SYS>>

                # Context: {context}

                # Question: {question}

                # Answer: [/INST]
                """, 
            "cn": """
                [INST] <<SYS>>
                你是一个乐于助人、尊重他人且诚实的助手。始终尽可能提供有帮助的回答，同时保持安全。
                你的回答不应包含任何有害、不道德、种族主义、性别歧视、毒性、危险或非法的内容。
                请确保你的回答在社会上没有偏见，并且具有积极性。

                如果问题没有意义，或不符合事实逻辑，请解释原因，而不是回答一个错误的内容。
                如果你不知道问题的答案，请不要提供虚假的信息。
                <</SYS>>

                # 上下文: {context}

                # 问题: {question}

                # 回答: [/INST]
                """, 
        },
        "input_variables":["context", "question"]
    },         
    ########################################################
    # RAG template
    "CHATBOT_RAG_TEMPLATE" :{
        "template":{
            "en": """
                You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

                {context}

                Question: {question}

                Answer:
                """, 
                            "cn": """

                你是一个问答任务的助手。
                使用以下检索到的上下文来回答问题。
                如果你不知道答案，就说你不知道。回答最多三句话，保持简洁。

                {context}

                问题：{question}

                回答：
                """, 
        },
        "input_variables":["context", "question"]
    },     

    ########################################################
    # CONVERSATION RAG 
    "CHATBOT_CONVERSATION_RAG_QUESTION_LANG_TEMPLATE" :{
        "template":{
            "en": """
                Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

                    Chat History:
                    {chat_history}

                    Follow Up Input: {question}

                    Answer in the following language: {language}

                    Standalone question
                """, 
            "cn": """
                给定以下对话历史和后续问题，请将后续问题重新表述为一个独立的问题，并保持原语言。

                对话历史： {chat_history}

                后续问题输入：{question}

                请用以下语言回答：{language}

                独立问题
                """, 
        },
        "input_variables":["context", "question","chat_history"]
    }, 

    "CHATBOT_CONVERSATION_RAG_ANSWER_LANG_TEMPLATE" :{
        "template":{
            "en": """
                Answer the question based only on the following context using Chinese:
                {context}

                Answer in the following language: {language}

                Question: {question}
            """, 
            "cn": """
                仅根据以下内容回答问题，使用中文： {context}

                请用以下语言回答：{language}

                问题：{question}
                """, 
        },
        "input_variables":["context", "question","chat_history"]
    },       


    ########################################################
    # CONVERSATION RAG 
    "CHATBOT_CONVERSATION_RAG_QUESTION_TEMPLATE" :{
        "template":{
            "en": """
                Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

                Chat History:
                {chat_history}

                Follow Up Input: {question}

                Standalone question""", 
            "cn": """  
                给定以下对话历史和后续问题，请将后续问题重新表述为一个独立的问题，并保持原语言。

                对话历史： {chat_history}

                后续问题输入：{question}

                独立问题
                """, 
        },
        "input_variables":["context", "question","chat_history"]
    }, 
    
    "CHATBOT_CONVERSATION_RAG_ANSWER_TEMPLATE" :{
        "template":{
            "en": """
                Answer the question based only on the following context:
                {context}

                Question: {question}""", 
            "cn": """ 
                仅根据以下内容回答问题： {context}

                问题：{question}
                """, 
        },
        "input_variables":["context", "question","chat_history"]
    },               
}
