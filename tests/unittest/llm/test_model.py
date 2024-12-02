import unittest
from unittest import TestCase
from unittest.mock import patch

import sys
import os

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain_core.messages import AIMessage, HumanMessage

llm_dir = os.path.dirname(os.path.abspath(__file__))
unittest_dir = os.path.dirname(llm_dir)
tests_dir = os.path.dirname(unittest_dir)
chingmanlib_dir = os.path.dirname(tests_dir)
project_dir = os.path.dirname(chingmanlib_dir)
sys.path.append(project_dir)

from chingmanlib.llm.llm_hf import LLMHFxecutor
from chingmanlib.tests.settings import BASE_DIR, CACHE_DIR

def get_device():
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.") # pylint 
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")


class TestLLMHFxecutor(unittest.TestCase):
    
    def setUp(self):
        config = {
            "base_dir": BASE_DIR,
            "cache_dir": CACHE_DIR,
            "device": "cuda",
        }

        self.llm = LLMHFxecutor(**config)

    def tearDown(self):
        pass
    
    @unittest.skip("skip this test")
    def test_skip(self):
        pass
    
    def test_device(self):
        self.assertTrue(get_device)
        
    def test_initialization(self):
        self.assertEqual(self.llm.device,'cuda')
        # self.assertIs(self.llm,HuggingFacePipeline)
        
    def test_get_embeddings(self):
        # self.assertIs(LLMHFxecutor.get_embeddings(),HuggingFaceEmbeddings)
        pass
    
#     @pytest.mark.parametrize("s3_url, expected_bucket, expected_key, expected_filename",
#                          [("s3://klue/test.txt", "klue", "test.txt", "test.txt"),
#                           ("s3://klue/klue/test.txt", "klue", "klue/test.txt", "test.txt"),])
    @patch('chingmanlib.llm.llm_hf.LLMHFxecutor.run', return_value="run result with string")
    def test_run_to_message(self,mock_run):
        prompt = "fake prompt"
        ret = self.llm.run_to_message(prompt)
        # self.assertIs(ret,AIMessage(content='run result with string'))

###################################################
#
#               pytest
#
###################################################
import pytest
from transformers import pipeline

@pytest.fixture
def setup_llm():
    """Fixture to set up the LLM model using Hugging Face pipeline."""
    # Use transformer in Hugging Face to load a language generation model
    
    generator = pipeline("text-generation", model="gpt2")

    return generator

# parametrize test: test different input and output
@pytest.mark.parametrize("input_text, min_length, expected_word", [
    ("Hello, my name is", 10, "name"),   # input should contain "name"
    ("In the year 2023,", 15, "2023"),  # input should contain "2023"
])
def test_llm_response_contains_word(setup_llm, input_text, min_length, expected_word):
    """ test whether the generated result contain the expected word"""
    generator = setup_llm

    # generate text
    result = generator(input_text, max_length=min_length, num_return_sequences=1)
    print(f"Generator result is {result}")
    
    # extract result from the generated text
    generated_text = result[0]['generated_text']

    # assert the generated text contain the expected word
    assert expected_word in generated_text, f"Expected word '{expected_word}' not in generated text"


def test_llm_output_length(setup_llm):
    """test whether the generated text length meet the expectation"""
    generator = setup_llm

    input_text = "The quick brown fox"
    min_length = 20

    # generate text
    result = generator(input_text, max_length=min_length, num_return_sequences=1)
    print(f"Generator result is {result}")

    # extract result from the generated text
    generated_text = result[0]['generated_text']

    # asset the generated length contain at least length of min_length
    assert len(generated_text) >= min_length, f"Generated text is shorter than {min_length} characters"
       
@patch('transformers.pipeline')
def test_llm_with_mock(mock_pipeline):
    """Use mock to test the LLM response"""
    
    # simulate the LLM generated result
    mock_pipeline.return_value = lambda input_text, **kwargs: [{'generated_text': 'Hello, world!'}]

    generator = mock_pipeline()
    
    input_text = "Hello"
    result = generator(input_text)

    generated_text = result[0]['generated_text']

    # validate whether the returned text meet expectation
    assert generated_text == "Hello, world!"
        
# pytest.mark.scenario 不是 pytest 的内置功能，而是一个用户定义的标记（marker）。在 pytest 中，用户可以创建自定义标记来标识和分类测试用例。       
@pytest.mark.scenario
def test_llm_scenario_customer_support(setup_llm):
    """Scenario test for customer support conversation."""
    generator = setup_llm

    # Round 1: User asks for return policy
    user_input_1 = "Hi, I want to return a product. What's your return policy?"
    response_1 = generator(user_input_1, max_length=50, num_return_sequences=1)
    generated_text_1 = response_1[0]['generated_text']

    # Assert the response contains key phrases
    assert "return policy" in generated_text_1 or "return" in generated_text_1, \
        "The response should mention return policy"

    # Round 2: User follows up with a request for time period for returns
    user_input_2 = "How many days do I have to return the product?"
    response_2 = generator(user_input_2, max_length=50, num_return_sequences=1)
    generated_text_2 = response_2[0]['generated_text']

    # Assert the response mentions a reasonable number of days for return
    assert any(str(day) in generated_text_2 for day in range(2,30)), \
        "The response should mention a valid return period"

    # Round 3: User asks about the return process
    user_input_3 = "Can I return it online or do I have to visit a store?"
    response_3 = generator(user_input_3, max_length=50, num_return_sequences=1)
    generated_text_3 = response_3[0]['generated_text']

    # Assert the response provides an appropriate return process
    assert "online" in generated_text_3 or "store" in generated_text_3, \
        "The response should explain whether returns can be done online or in store"

    # Round 4: User asks for return shipping costs
    user_input_4 = "Do I have to pay for return shipping? "
    response_4 = generator(user_input_4, max_length=50, num_return_sequences=1)
    print(f"response is {response_4}")
    generated_text_4 = response_4[0]['generated_text']

    # Assert the response contains a valid shipping cost policy
    assert any(phrase in generated_text_4 for phrase in ["free", "no cost", "you will be charged"]), \
        "The response should mention return shipping policy"
    
@pytest.fixture
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
        
@pytest.mark.parametrize("prompt, expected_word", [
    ("Hello, my name is", "name"), 
])
def test_llm_hf_response_contains_word(setup_llm_hf, prompt, expected_word):
    """Use mock to test the LLM response"""
    
    generator = setup_llm_hf
    
    generated_text = generator.run(prompt)
    print(f"Generator result from {prompt} is {generated_text}")

    # validate whether the returned text meet expectation
    assert expected_word in generated_text, f"Expected word '{expected_word}' not in generated text"
    
@patch('chingmanlib.llm.llm_hf.LLMHFxecutor.run', return_value="Hello, my name is Alex")
@pytest.mark.parametrize("prompt, expected_word", [
    ("Hello, my name is", "Hello, my name is Alex"), 
])
def test_llm_hf_with_mock(mock_pipeline, setup_llm_hf, prompt, expected_word):
    """Use mock to test the LLM response, only mock run function"""
    
    # generator = mock_pipeline()
    generator = setup_llm_hf
    
    generated_text = generator.run(prompt)

    # validate whether the returned text meet expectation
    assert generated_text == expected_word
        
@patch('chingmanlib.llm.llm_hf.LLMHFxecutor.run')
@pytest.mark.parametrize("prompt, expected_word", [
    ("Hello, my name is", "Hello, my name is Alex"), 
])
def test_llm_hf_with_mock(mock_pipeline, prompt, expected_word):
    """Use mock to test the LLM response"""
    
    mock_pipeline.return_value = "Hello, my name is Alex"
    
    generated_text = mock_pipeline()
    
    # validate whether the returned text meet expectation
    assert generated_text == expected_word        

# pytest test_model.py::test_llm_response_contains_word -v -s

###################################################
#
#               unittest testsuite
#
###################################################        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestLLMHFxecutor('test_device'))
    suite.addTest(TestLLMHFxecutor('test_initialization'))
    return suite

if __name__ == '__main__':
    # full testset
    unittest.main()
    
    # test suite
    # runner = unittest.TextTestRunner()
    # runner.run(suite())    