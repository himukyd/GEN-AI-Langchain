# This Runnable lambda conver a python function into a Runnable.
### What is the benifit into conver into a runnable function 
## Benifit-1. The first Benifit is that after connvert into a Runnable Function it aslo connct to langchain Runnable Functions


                                    #Seq                                             #Parallel          
                                                                                    #---->> Passthrough()
## Task --->> Prompt(generate a joke) ---> LLM ---> Parser(Joke)\\ Sequential       #---->> Wite a explanation of joke 
                                                                                    #---->> Runnable Lambda for count words in a joke 

# Define a function that count words 

def word_counter(text):
    return len(text.split())

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(

    model = 'gemini-3.5-flash'
)

prompt_1 = PromptTemplate(
    template="Write a one line joke on {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Write a 2 line explanation on {text}",
    input_variables=['text']
)

parser = StrOutputParser()

runnable_word_counter = RunnableLambda(word_counter) ## Convert that function into a Runnable 
## So this thing is runnable that means it have invoke and all essential functions

## Seq chain 

chain_1 = prompt_1 | model | parser 

## Parallel Chain 
chain_2 = RunnableParallel({
    'chain_2_1': RunnablePassthrough(),
    'chain_2_2': ({'text':RunnablePassthrough()} | prompt_2 | model | parser ),
    'chain_2_3': runnable_word_counter 
})

chain = chain_1 | chain_2 

result = chain.invoke({'topic':'student'})
print(result)
print(chain.get_graph().draw_ascii())