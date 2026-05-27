from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)
parser = StrOutputParser()

loader = TextLoader('cricket.txt',encoding='utf-8')

prompt_1 = PromptTemplate(
    template="Write a summary on this {poem}",
    input_variables=['poem']
)

docs = loader.load()
# So docs give output in List Format and tht list have 2 things 1. page_content and 2. metadata

print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt_1 | model | parser 
result = chain.invoke({'poem':docs[0].page_content})
print(result)