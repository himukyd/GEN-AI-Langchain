from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-5",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)

chat_history = []
while True:
    user_input = input('You: ')
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI :",result.content)

print(chat_history)


## In this there is also an problem that you can see in chat_history there is no track which massage is sended by humann
## and which one is ai generated so it is tipical to understand for a llm. 
