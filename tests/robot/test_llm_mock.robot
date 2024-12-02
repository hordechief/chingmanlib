*** Settings ***
Library           RequestsLibrary
Library           OperatingSystem
Library           Collections
Library           ./libraries/MyLibrary.py
Library           ./libraries/my_mock.py
Library           ./libraries/MockLibrary.py
#Library           ./libraries/

*** Variables ***
${BASE_URL}      http://your-llm-api-endpoint

*** Test Cases ***
Test Log Message
    ${message}=    My Log Message Func    World
    Log    ${message}
    
Test LLM Response With Mock
    [Documentation]    Test if LLM returns expected response for given input using mock.
    My Log Message Tests    This is a message from MyLibrary.
    Kw Create Mock Server
    ${response}=       Mock Post Request    ${BASE_URL}/generate    {"prompt": "Hello, how are you?"}    {"text": "Hello, I'm fine."}
    Should Be Equal As Strings    ${response.status_code}    200
    ${json_response}=  To Json    ${response.content}
    Should Contain    ${json_response}    "Hello, I'm fine."    

Test LLM With Different Prompts Using Mock
    [Documentation]    Test LLM response with various prompts using mock.
    Kw Create Mock Server
    ${prompts}=       Create List    "Tell me a joke."    "What's the weather like?"
    FOR    ${prompt}    IN    @{prompts}
        ${response}=    Mock Post Request    ${BASE_URL}/generate    {"prompt": ${prompt}}    {"text": "Mocked response for ${prompt}"}
        Should Be Equal As Strings    ${response.status_code}    200
        ${json_response}=    To Json    ${response.content}
        Log    ${json_response}
    END
    
*** Keywords ***
Kw Create Mock Server
    ${mock_server}=    Create Mock Server
#     ${mock_server}.start()
    
# Mock Post Request
#     [Arguments]    ${url}    ${data}    ${mock_response}
#     ${response}=    Create Mock Response    ${mock_response}    200
#     Return    ${response}    