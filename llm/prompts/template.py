from langchain.prompts import ChatPromptTemplate

review_template_str = """Your job is to use patient 
reviews to answer questions about their experience at a hospital.
Use the following context to answer questions. Be as detailed
as possible, but don't make up any information that's not
from the context. If you don't know an answer, say you don't know.

{context}

{question}
"""

review_template = ChatPromptTemplate.from_template(review_template_str)
print(review_template)

context = "I had a great stay!"
question = "Did anyone have a positive experience?"

review_template.format(context=context, question=question)
print(review_template)
