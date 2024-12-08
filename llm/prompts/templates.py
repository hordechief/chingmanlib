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

CHATBOT_TEMPLATE = {
    "template": """This is the Chatbot sytem ..."
Let's work this out in a step by step way to be sure we have the right answer.
Question: {question}
Answer: """, 
    "input_variables":["Question"]
}

CHATBOT_CONVERSATION_TEMPLATE = {
    "template": """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:""",
    "input_variables":["history", "input"]
}

CHATBOT_RAG_TEMPLATE = {
    "template": """
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
    "input_variables":["context", "question"]
}

CHATBOT_RAG_QA_TEMPLATE = {
    "template": """
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            {context}

            Question: {question}

            Answer:
            """,
    "input_variables":["context", "question"]
}