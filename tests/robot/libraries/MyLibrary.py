from robot.api.deco import keyword
from unittest.mock import patch
import json

from robot.api import logger

# if the filename is same as the class name, robot will search the class function
# otherwise, it will search the function outside the class

class MyLibrary:
    
    ROBOT_LIBRARY_SCOPE = "GLOBAL"

    @keyword
    def my_log_message_cls(self, message):
        logger.info(message)  # 记录信息级别的日志

