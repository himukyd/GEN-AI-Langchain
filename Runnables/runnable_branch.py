from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description = "Give me the sentijment of the feedback")

parser_2 = PydanticOutputParser(pydantic_object=Feedback)

prompt_1 = PromptTemplate(
    template="Classified the sentiment from feedback in Positive and Negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser_2.get_format_instructions()}
)

class_chain = prompt_1 | model | parser_2

prompt_2 = PromptTemplate(
    template="Write an appopriate response on this positive feedback \n {feedback}",
    input_variables=['feedback']
)
prompt_3 = PromptTemplate(
    template="Write an appopriate response on this negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x :x.sentiment == 'positive', prompt_2 | model | parser),
    (lambda x :x.sentiment == 'negative', prompt_3 | model | parser),
    RunnableLambda(lambda x : "Could not find sentiment")
)

chain = class_chain | branch_chain 

text = """
    phone is beautiful buit i recive3d in different colour
"""

result = chain.invoke({"feedback":text})
print(result)

chain.get_graph().print_ascii()