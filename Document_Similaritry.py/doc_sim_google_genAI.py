from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    output_dimensionality=300
)
documents = [
    """Virat Kohli is one of the greatest batsmen in modern cricket. He is known for his aggressive batting style and consistency across all formats. 
    Kohli has scored many centuries for India and has led the team successfully. His fitness and dedication inspire young cricketers around the world.""",

    """MS Dhoni is regarded as one of the best captains in cricket history. He is famous for his calm nature and excellent finishing ability in matches. 
    Dhoni led India to win the T20 World Cup, ODI World Cup, and Champions Trophy. Fans admire his leadership and wicketkeeping skills.""",

    """Sachin Tendulkar is called the God of Cricket because of his incredible achievements. He scored 100 international centuries and inspired millions of cricket fans. 
    Tendulkar played for India for more than two decades and set many records. His humble personality made him respected worldwide.""",

    """Rohit Sharma is known for his elegant batting and ability to score big hundreds. He holds the record for the
    highest individual score in ODI cricket. Rohit has captained India in multiple tournaments and performed consistently.
    His timing and pull shots are admired by cricket lovers.""",

    """Jasprit Bumrah is one of the best fast bowlers in world cricket today. He is famous for his unique bowling action and deadly yorkers. 
    Bumrah has helped India win many important matches with his pace and accuracy. He performs exceptionally well in all formats of the game."""
]
query = "who is best bolwer?"

doc_embeddings = []

for doc in documents:
    embedding = embeddings.embed_query(doc)
    doc_embeddings.append(embedding)

# Query embedding
query_embedding = embeddings.embed_query(query)

# Similarity calculation
scores = cosine_similarity([query_embedding],doc_embeddings)
print(scores)
# Get index of highest similarity score
highest_score_index = np.argmax(scores[0])

# Print best matching document
print("Highest Similarity Score:")
print(scores[0][highest_score_index])

print(query)
print("\nMost Relevant Document:")
print(documents[highest_score_index])