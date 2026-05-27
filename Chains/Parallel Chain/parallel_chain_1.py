from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash'
)
 ## Will give you a text and give me notes model 1 
prompt_1 = PromptTemplate(
    template="based on {text} generate noites on based of that topic.",
    input_variables=['text']
)
# From notes generate 5 short que and ans model 2 
prompt_2 = PromptTemplate(
    template="Generate 5 Short Question and answer from the following {text}.",
    input_variables=['text']
)
# Merge in single document
prompt_3 = PromptTemplate(
    template="Merge the provided notes and quize into a sible documents {notes} and {quiz}",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

## we develop our chain in 2 parts 1. models and 2. Merge doc  and finnllay we have to merge 2 chains

parallel_chain = RunnableParallel({ ## In Runnable all API request are in parallel 
    'notes' : prompt_1 | model | parser, # Chain Name is notes that shoud be same as prompt_3 requres 
    'quiz' : prompt_2 | model | parser
})

merge_chain = prompt_3 | model | parser # Document merging notes and Quiz 

chain = parallel_chain | merge_chain

text = """
    Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({'text':text})
print(result)

#Chain diagram 
chain.get_graph().print_ascii()