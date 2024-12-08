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

from submodules.chingmanlib.llm.models.hf import LLMHFxecutor
from chingmanlib.tests.settings import BASE_DIR, CACHE_DIR

def setup_function():
    print("\n测试前的准备工作")

def teardown_function():
    print("测试后的清理工作")