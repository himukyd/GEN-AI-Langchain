from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

loader = PyPDFLoader('linear_regression_notes.pdf')
docs = loader.load() ## we can see set of documents 

# Extract text from all pages
text = "\n".join([doc.page_content for doc in docs])

parser = StrOutputParser()

                                                    # ---> Passthrough  
## Chain_1 --> Load Doc --> Model -->> Parser       # ---> Summary of notes
                                                    # ---> Que Ans Generate 
prompt_1 = PromptTemplate(
    template="Write a 3 line summary in points of the {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Make a short 5 questins and answers based on {topic}",
    input_variables=['topic']
)


chain_1 = RunnableParallel({
    # Original text
    'original_text': RunnablePassthrough(),
    # Summary
    'summary':({'topic': RunnablePassthrough()} | prompt_1 | model | parser ),
    # Q&A
    'qa':({'topic': RunnablePassthrough()} | prompt_2 | model | parser)
})


# Invoke
result = chain_1.invoke(text)

print(result)

chain_1.get_graph().print_ascii()