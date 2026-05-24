from langchain_openai import OpenAI
from dotenv import load_dotenv ### dotenv load secrate api key in this current code 

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("What is the capital of India?") ## invoke is a method to communicate with different models in this case we communicate with gpt3
print(result)