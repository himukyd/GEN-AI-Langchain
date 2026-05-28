from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled 
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

load_dotenv()

model_1 = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)
model_2 = GoogleGenerativeAIEmbeddings(
    model = 'gemini-embedding-001'
)

## Step  1(a) we hit youtube api and load that's video transcript 


## we first have to pick that video id from link 
video_id = "Gfr50f6ZBvo"

try:
    #  New API — create instance first, then call .fetch()
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)
    
    transcript = " ".join(chunk.text for chunk in transcript_list)
    print(transcript[:500]) # first 500 characters only

except TranscriptsDisabled:
    print("No captions available for this video.")

## Step. 1(b)

# we use text splitter to split that text 
splitter = RecursiveCharacterTextSplitter(chunk_size = 3000 , chunk_overlap = 200)
chunks = splitter.create_documents([transcript])
print(len(chunks)) ## Total no of chunks 


## Step 1(c)

# we have chunk and now we create embedding for each chunk and store in vector stores 

vector_store = Chroma(
    collection_name="data_yt",
    embedding_function=model_2,
)

vector_store.add_documents(chunks)

ids = vector_store.get()['ids']
print(f"Total chunks stored: {len(ids)}")
print(ids)

## To see chunk by its id 
print(vector_store.get_by_ids(['b3ff2ac1-ceae-4ae0-a169-4a49881f5a65']))




## Step 2(a) 
## We used vector store retriver 
retriever = vector_store.as_retriever(
    search_type = 'similarity',
    search_kwargs = {"k":3}
)

## Step 2(b)
query = "what is deep mind ?"
print(retriever.invoke(query))



## Step 3(a)
## now we will move on the Augmentation part 
prompt = PromptTemplate(
    template="""You are a helpful assistant.Answer ONLY from the provided transcript context.If the context is insufficient, just say you don't know.{context} Question: {question}""",
    input_variables = ['context', 'question']
)

question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question) ## This will give 3 


context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
context_text 


final_prompt = prompt.invoke({"context": context_text, "question": question}) ## Final Prompt

print(final_prompt)


# Step 4(a) Generation sent to LLM 
answer = model_1.invoke(final_prompt)
print("Answer is : ----- -------   ------   --------- :  ",answer.content)

