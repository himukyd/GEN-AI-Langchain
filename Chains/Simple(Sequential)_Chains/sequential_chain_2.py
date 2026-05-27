from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

parser = StrOutputParser()

## Dynamic prompt From user 
prompt_1 = PromptTemplate(
    template="Write a summary on {topic}, and it is understandable to {std} student.",
    input_variables=['topic','std']
)

prompt_2 = PromptTemplate(
    template="Erite a 5 line summary on {text}",
    input_variables=["text"]
)

## LLM 1 --> Parser ---> LLM2 ---> parser ---> result 
chain = prompt_1 | model | parser | prompt_2 | model | parser
result= chain.invoke({"topic":"Cricket","std":"10th"}) ## Do Not need to define what is the input of prompt_2 it automatically extracts

print(result)

## Chain Visulation 
chain.get_graph().print_ascii()
