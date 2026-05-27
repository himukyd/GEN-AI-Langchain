from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnableParallel
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader 

from dotenv import load_dotenv 

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

loader = DirectoryLoader(
    path= 'Content',
    glob= '*.pdf', # from this folder i want to extract all pdf files 
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)



for documents in docs :
    print(documents.metadata)