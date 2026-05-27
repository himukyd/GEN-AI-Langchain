from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

prompt_1 = PromptTemplate(
    template="Write a jock in one line on {topic}",
    input_variables=['topic']
)
prompt_2 = PromptTemplate(
    template="Explain the {text}",
    input_variables=['text']
)
parser = StrOutputParser()

chain = prompt_1 | model | parser | prompt_2 | model | parser

result = chain.invoke({"topic":'A.I.'})
print(result)
chain.get_graph().print_ascii()