from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
## Crete a Dynamic Prompt so we have to import 
from langchain_core.prompts import PromptTemplate
## Load Output Parser 
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(
    model  = 'gemini-2.5-flash'
)

## Dynamic tamplet pass To LLM 
templet_1 = PromptTemplate(
    template="Explain me in Deatil {topic}.",
    input_variables=['topic']
)

## Dynamic tamplet pass to LLM
templet_2 = PromptTemplate(
    template="Give me 5 points summary of {text}",
    input_variables=['text']
)
parser = StrOutputParser() ## This Extract a content poart from an LLM Output
## Create a Chain
chain = templet_1 | model | parser | templet_2 | model | parser

## Fianl Output 
result = chain.invoke({'topic':'Black Hole'})
print(result)

