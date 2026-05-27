from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)


prompt_1 = PromptTemplate(
    template="Write a linked in post on topic {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Write a twitter post on {topic} ",
    input_variables=['topic']
)

parser = StrOutputParser()

# We have to built a parallel Chain 

chain = RunnableParallel({
    'chain_1' : prompt_1 | model | parser , 
    'chain_2' : prompt_2 | model | parser
}
)

result = chain.invoke({'topic':"Jio fiber not installed till now i did payment before 15 days."})
print(result)
print(chain.get_graph().draw_ascii())