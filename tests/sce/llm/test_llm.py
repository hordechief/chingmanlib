import unittest
from unittest import TestCase
from unittest.mock import patch

import sys
import os

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain_core.messages import AIMessage, HumanMessage

llm_dir = os.path.dirname(os.path.abspath(__file__))
scettest_dir = os.path.dirname(llm_dir)
tests_dir = os.path.dirname(scettest_dir)
chingmanlib_dir = os.path.dirname(tests_dir)
project_dir = os.path.dirname(chingmanlib_dir)
sys.path.append(project_dir)

from submodules.chingmanlib.llm.models.hf import LLMHFxecutor
from chingmanlib.tests.settings import BASE_DIR, CACHE_DIR

def get_device():
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.") # pylint 
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

########################## Function Test ################################
# 功能测试用于验证 LLM 是否能够根据输入生成预期的输出。


import pytest

@pytest.fixture(scope="module")
def setup_llm_hf():
    """Fixture to set up the LLM model using Hugging Face pipeline."""
    
    config = {
        "base_dir": BASE_DIR,
        "cache_dir": CACHE_DIR,
        "device": "cuda",
        "model_kwargs": {'temperature':0.7,'max_length': 10, 'top_k' :50},
    }

    generator = LLMHFxecutor(**config)
    return generator
        
def test_llm_generate_text(setup_llm_hf):
    llm = setup_llm_hf
    
    prompt = "Once upon a time"
    expected_output_part = "Once upon a time"
    generated_text = llm.run(prompt)
    print(generated_text)
    
    assert expected_output_part in generated_text

    
############################### Scenario Test ########################################
# 情景测试用于模拟用户与模型的交互，以验证模型在特定上下文中的表现。


@pytest.mark.parametrize("prompt, expected_response", [
    ("What is the capital of France? Answer in english", "Paris"),
    ("Translate 'Hello' to Spanish.", "Hola"),
])
def test_llm_scenario(setup_llm_hf, prompt, expected_response):
    llm = setup_llm_hf
    
    response = llm.run(prompt)
    assert expected_response in response


############################### Performance Test ####################################
# 性能测试用于评估模型在不同输入条件下的响应时间和效率。
import time

def test_llm_response_time(setup_llm_hf):
    llm = setup_llm_hf
    
    prompt = "Generate a story about a brave knight."
    start_time = time.time()
    llm.run(prompt)
    elapsed_time = time.time() - start_time
    
    assert elapsed_time < 20  # 假设我们希望模型在20秒内完成生成

############################## Integration Test #####################################
# 如果 LLM 是在更大的系统中使用，例如与 API 交互或前端应用，集成测试可以确保系统的各个部分正常工作。

@pytest.mark.skip(reason="This test is skipped for demonstration purposes.")
def test_llm_api_integration(client):
    response = client.post("/api/generate", json={"prompt": "Tell me a joke."})
    assert response.status_code == 200
    assert "joke" in response.json()  # 假设响应中包含 "joke" 字段

# @patch('your_llm_module.external_api_call')
# def test_llm_with_mock(mock_api):
#     mock_api.return_value = "Mock response"
#     prompt = "What is the weather today?"
#     response = llm.generate(prompt)
    
#     assert response == "Mock response"

    
############################## Exception Test #######################################
# 测试模型在处理边界情况或无效输入时的表现。
def test_llm_invalid_input(setup_llm_hf):
    llm = setup_llm_hf
    
    with pytest.raises(ValueError):  # 假设无效输入会抛出 ValueError
        llm.run("")  # 传递一个空字符串


#  pytest test_llm.py::test_llm_generate_text -v -s    
    
    
    
    
    
    
    