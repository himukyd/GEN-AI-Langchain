from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0, ## By chunk_overlab i mean that how many comman character in 2 chunks
    separator=''
)

## Go thjrough this link to batter vizulkations https://chunkviz.up.railway.app/


result = splitter.split_documents(docs)

print(result[0])