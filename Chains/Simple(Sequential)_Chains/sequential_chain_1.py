from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
st = time.time()
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash'
)

## Enter a Promt by User 

templet_1 = PromptTemplate(
    template="Write a 5 line summary on {topic} as you explain to a {person}.",
    input_variables=['topic','person']
)
parser = StrOutputParser()

## Sent that pro9mt to llm 

## Print the LLM result
chain = templet_1 | model | parser 

result = chain.invoke({'topic':"Attention all You Need Ppaer", 'person':'Student'})
print(result)
print(time.time()-st)

## To visualize a chain 
chain.get_graph().print_ascii()