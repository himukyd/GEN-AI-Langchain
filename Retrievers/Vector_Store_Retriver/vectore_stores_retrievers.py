from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAIEmbeddings(
    model = 'gemini-embedding-001'
)

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

vector_store = Chroma.from_documents(
    documents = documents,
    embedding= model,
    collection_name="my_collection"
)

# Step 4: Convert vectorstore into a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2}) ## this is a way to create a retriver 

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)