from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
## In this chat model there is no context of history
#  that is no history of previous chats.

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-5",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)

while True:
    user_input = input('You: ')
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    print("AI: ",result.content)