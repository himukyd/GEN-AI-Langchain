from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate # Core is used as these are very useful and pepole use very frequent
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 
import time
start = time.time()
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash'
)

class Person(BaseModel):
    name : str =Field(description="Name of the person")
    age : int = Field(gt=18, description="Age of the person")
    city : str = Field(description="Name of the city that belong to the person")


parser = PydanticOutputParser(pydantic_object=Person) # 2nd one see this 

template = PromptTemplate(
    template="Generate the name age and city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()} ## See this one thing 
)
chain = template | model | parser
result = chain.invoke({'place':'Indian'})
print(result)
time_taken = time.time()-start
print(time_taken)