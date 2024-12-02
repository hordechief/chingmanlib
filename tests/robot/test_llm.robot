*** Settings ***
Library           RequestsLibrary

*** Variables ***
${BASE_URL}      http://localhost:8000/api/generate  # LLM API 的基本 URL
${TIMEOUT}       10

*** Test Cases ***
Test LLM with Valid Input
    [Documentation]    测试 LLM 对有效输入的响应
    ${response} =      Post Request    ${BASE_URL}    json={"prompt": "What is the capital of France?"}    timeout=${TIMEOUT}
    Should Be Equal As Strings    ${response.status_code}    200
    ${json_response} =    To Json    ${response.content}
    Should Contain    ${json_response['response']}    Paris

Test LLM with Invalid Input
    [Documentation]    测试 LLM 对无效输入的响应
    ${response} =      Post Request    ${BASE_URL}    json={"prompt": ""}    timeout=${TIMEOUT}
    Should Be Equal As Strings    ${response.status_code}    400
    ${json_response} =    To Json    ${response.content}
    Should Contain    ${json_response['error']}    "Invalid prompt"

Test LLM with Special Characters
    [Documentation]    测试 LLM 对特殊字符的响应
    ${response} =      Post Request    ${BASE_URL}    json={"prompt": "Translate 'Hello' to Spanish."}    timeout=${TIMEOUT}
    Should Be Equal As Strings    ${response.status_code}    200
    ${json_response} =    To Json    ${response.content}
    Should Contain    ${json_response['response']}    Hola
