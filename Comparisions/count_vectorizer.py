from vector_store import VectorStore
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Create a VectorStore instance
vector_store = VectorStore()

# Define your sentences
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







# Initialize CountVectorized
vectorizer = CountVectorizer()

# Tokenization and Vocabulary Creation
vectorizer.fit(sentences)

# Vocabulary
vocabulary = vectorizer.get_feature_names_out()

# Vectorization
sentence_vectors = {}
for sentence in sentences:
    vector = vectorizer.transform([sentence]).toarray().flatten()
    sentence_vectors[sentence] = vector

# Storing in VectorStore
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# Searching for Similarity
print("\nCount-Vectorizer\n")
query_sentence = "Which startups are addressing critical needs? "
query_vector = vectorizer.transform([query_sentence]).toarray().flatten()

similar_sentences_with_embeddings = vector_store.find_similar_vectors_with_embeddings(query_vector, num_results=2)

# Print similar sentences with embeddings
print("Query Sentence:", query_sentence)
print("Similar Sentences:")
for sentence, similarity, embedding in similar_sentences_with_embeddings:
    print(f"{sentence}: Similarity = {similarity:.4f}")
    # print("Embedding:", embedding)

# print("\n\n Vector DB\n \n ")
# print(vector_store.vector_index)

