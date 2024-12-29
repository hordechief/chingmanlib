# https://stackoverflow.com/questions/77789886/openai-api-error-the-model-text-davinci-003-has-been-deprecated

from langchain_openai import OpenAI

class COT():
    def __init__(self, llm):
        self.llm = llm

    def zero_shot_cot(self,text):
        print(f"Question is {text}")
        text = f"Explain step by step. {text}"
        response = self.llm(text)
        print(response)
        return response


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).parent.parent / ".env"
    assert os.path.exists(env_path)
    load_dotenv(dotenv_path=str(env_path), override=True)  
        
    llm = OpenAI(model_name='gpt-3.5-turbo-instruct',
             temperature=0,
             max_tokens = 256)
    cot = COT(llm)
             
    text = "Explain step by step. How old is the president of the United States?"
    print(cot.zero_shot_cot(text))
    
    '''
    Step 1: Determine the current year
    The first step in determining the age of the president of the United States is to determine the current year. As of 2021, the current year is 2021.

    Step 2: Identify the president's birth year
    Next, we need to identify the birth year of the current president. As of 2021, the current president is Joe Biden, who was born on November 20, 1942.

    Step 3: Calculate the president's age
    To calculate the president's age, we subtract the birth year from the current year. In this case, 2021 - 1942 = 79. This means that as of 2021, Joe Biden is 79 years old.

    Step 4: Consider the president's birthday
    It's important to note that the president's age may change throughout the year depending on their birthday. In this case, Joe Biden's birthday is in November, so he will turn 80 in 2022.

    Step 5: Verify the information
    To verify the age of the president, we can check reliable sources such as government websites or news articles.

    Step 6: Repeat the process for previous presidents
    To determine the age of previous presidents, we
    '''
    
    text = "Does Dell Precision 5480 support GPU V1000?"
    print(cot.zero_shot_cot(text))
        
    '''
    Step 1: Check the specifications of Dell Precision 5480
    The first step is to check the specifications of Dell Precision 5480 to see if it supports the GPU V1000. This can be done by visiting the Dell website or checking the product manual.

    Step 2: Check the compatibility of GPU V1000
    Next, check the compatibility of GPU V1000 with Dell Precision 5480. This can be done by checking the system requirements of the GPU on the manufacturer's website.

    Step 3: Check the available ports on Dell Precision 5480
    Check the available ports on Dell Precision 5480 to see if it has the necessary ports to connect the GPU V1000. The GPU V1000 requires a PCIe x16 slot for installation.

    Step 4: Check the power supply
    The GPU V1000 requires a power supply of at least 300W. Check the power supply of Dell Precision 5480 to ensure it meets the minimum requirement.

    Step 5: Install the GPU V1000
    If all the above steps are met, then the Dell Precision 5480 should support the GPU V1000. Install the GPU V1000 in the PCIe x16 slot and connect the necessary power cables.
    '''