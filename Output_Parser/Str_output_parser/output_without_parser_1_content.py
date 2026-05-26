from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
model = GoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)
## Ist prompt detailed 
template_1 = PromptTemplate(
    template="write a summary on the {topic}.",
    input_variables=['topic']
)
# 2nd prompt short ie 5 lines only
template_2 = PromptTemplate(
    template="Write a 5 line summary on {text}.",
    input_variables=['text']
)

prompt_1 = template_1.invoke({'topic':'Black Hole'})

result = model.invoke(prompt_1)

prompt_2 = template_2.invoke({'text':'result.content'})

result_1 = model.invoke(prompt_2)

print(result_1.content)
