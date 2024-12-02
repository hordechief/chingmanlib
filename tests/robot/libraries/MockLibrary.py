from robot.api.deco import keyword
from unittest.mock import patch
import json

from robot.api import logger

class MockLibrary:

    def __init__(self):
        self.mock_responses = {}

    @keyword
    def my_log_message_tests(self, message):
        logger.info(message)  # 记录信息级别的日志
        
    @keyword
    def create_mock_server(self):
        """Initialize the mock server."""
        self.mock_responses = {}
        print("Mock server started.")

    @keyword
    def mock_post_request(self, url, data, mock_response):
        """Mock a POST request to the specified URL."""
        # Store the mock response for the given URL and data
        self.mock_responses[(url, json.dumps(data))] = mock_response
        return self._create_mock_response(mock_response)

    def _create_mock_response(self, mock_response):
        """Create a mock HTTP response."""
        response = MockResponse()
        response.status_code = 200
        response._content = json.dumps(mock_response).encode('utf-8')
        return response

class MockResponse:
    """Mock class for HTTP response."""
    def __init__(self):
        self.status_code = None
        self._content = None

    @property
    def content(self):
        return self._content

    @property
    def json(self):
        return json.loads(self._content.decode('utf-8'))
