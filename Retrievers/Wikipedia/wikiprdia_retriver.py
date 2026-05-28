### Wikipedia Retriver is a type of retriver that take user query and on based of whole wekipedia contenet it do key/word search and give relevent content

from langchain_community.retrievers import WikipediaRetriever

retriver = WikipediaRetriever(
    top_k_results=2,
    lang='en'
)

query = "what is linear regression ?"

docs = retriver.invoke(query)

for document in docs:
    print(document.page_content)