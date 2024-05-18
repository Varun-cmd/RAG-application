from vector_store import VectorStore
import numpy as np
from gensim.models import Word2Vec

# Create a VectorStore instance
vector_store = VectorStore()

# # Define your sentences
# sentences = [
#     "I eat mango",
#     "mango is my favorite fruit",
#     "mango, apple, oranges are fruits",
#     "fruits are good for health",
#     "some fruits can be juiced",
#     "fruits are good for health",
#     "Strawberries are red color and packed with flavor",
#     "Strawberries are bitter in taste",
#     "Kiwis are fuzzy on outside",
#     "Avocados have creamy texture",
#     "Avocados are green in color"
# ]

sentences = [
    "India's startup ecosystem is flourishing, fueled by innovation and entrepreneurial spirit.",
    "From Bangalore to Mumbai, startups are revolutionizing industries with disruptive technologies.",
    "Investors are increasingly drawn to India's startup scene, recognizing its potential for high returns.",
    "Government initiatives like Startup India are providing support and incentives for aspiring entrepreneurs",
    "Indian startups are not only focusing on local markets but also expanding globally",
    "The rise of fintech startups is transforming the way financial services are accessed and utilized in India.",
    "Startups in sectors like healthcare and education are addressing critical needs",
    "Incubators and accelerators are playing a crucial role in nurturing and mentoring startups",
    "Collaborations between startups and established companies are fostering innovation",
    "Despite challenges, the resilience and creativity of Indian startups continue to drive forward"

]




# Train Word2Vec model
word_embeddings_model = Word2Vec([sentence.split() for sentence in sentences], vector_size=100, window=5, min_count=1, workers=4)

# Encoding sentences using Word2Vec embeddings
sentence_embeddings = {}
for sentence in sentences:
    tokens = sentence.lower().split()
    embedding = np.zeros(word_embeddings_model.vector_size)
    count = 0
    for token in tokens:
        if token in word_embeddings_model.wv:
            embedding += word_embeddings_model.wv[token]
            count += 1
    if count != 0:
        embedding /= count
    sentence_embeddings[sentence] = embedding

# Storing in VectorStore
for sentence, embedding in sentence_embeddings.items():
    vector_store.add_vector(sentence, embedding)

# Searching for Similarity
query_sentence = "Which startups are addressing critical needs?"
query_embedding = np.zeros(word_embeddings_model.vector_size)
query_tokens = query_sentence.lower().split()
count = 0
for token in query_tokens:
    if token in word_embeddings_model.wv:
        query_embedding += word_embeddings_model.wv[token]
        count += 1
if count != 0:
    query_embedding /= count

similar_sentences_with_embeddings = vector_store.find_similar_vectors_with_embeddings(query_embedding, num_results=2)

# Print similar sentences with embeddings
print("\n\nWord2vect Embeddings\n")
print("Query Sentence:", query_sentence)
print("Similar Sentences:")
for sentence, similarity, embedding in similar_sentences_with_embeddings:
    print(f"{sentence}: Similarity = {similarity:.4f}")
    # print("Embedding:", embedding)

print("\n\n")
# print(vector_store.vector_index)
