import os
from litellm import completion

from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent.parent / "llm" / ".env"
load_dotenv(dotenv_path=str(env_path), override=True)  # 加载 .env 文件    

# [OPTIONAL] set env var
# os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key"

messages = [{ "content": "There's a llama in my garden 😱 What should I do?","role": "user"}]

# e.g. Call 'https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct' from Serverless Inference API
response = completion(
    # model="huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",
    model="huggingface/NousResearch/Llama-2-7b-chat-hf",
    messages=[{ "content": "Hello, how are you?","role": "user"}],
    stream=True
)

for res in response:
    print(res)

# python -m chingmanlib.llm.models.litellm