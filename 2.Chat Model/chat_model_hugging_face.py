from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
## HuggingFaceEndpoint is used to talk with api and if we dowanlo9ad model in our machine then we have to use HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task="text-generation",
    temperature=0.5, # by setting high value of temprature it won't through any error as such
    max_new_tokens=100 ## by applying max_new_token that is not necessary that it will complete a sentance it left the sentance in between also
)

model = ChatHuggingFace(llm = llm)
result = model.invoke("What is the capital of India?")
print(result.content)