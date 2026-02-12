from langchain_huggingface import HuggingFaceEndpointEmbeddings
## HuggingFaceEmbeding is for local machine and HuggingFaceEndpointEmbeddings is for HuggingFace Endpoint
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

text = "What is the capital of India?"
vector = embedding.embed_query(text)

print(vector)
print("Vector length:", len(vector))