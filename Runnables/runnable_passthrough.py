from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash'
)

parser = StrOutputParser()
passthrough = RunnablePassthrough()

prompt_1 = PromptTemplate(
    template="Write a one line joke on {topic}.",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Give me a summary on this {text}",
    input_variables=["text"]
)

## what to do is mnake a chain that generate a joke 
## And make a parallel chain that print jock and that meaning 

chain_1 = prompt_1 | model | parser 

chain_2 = RunnableParallel(
    { 'chain_2_1' : passthrough ,
      'chain_2_2' : ({'text': passthrough } |prompt_2 | model | parser ) 
    }
)

chain = chain_1 | chain_2

result = chain.invoke({'topic':'AI'})
print(result)
print(chain.get_graph().draw_ascii())