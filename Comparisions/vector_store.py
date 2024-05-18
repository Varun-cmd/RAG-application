import numpy as np


class VectorStore:
    def __init__(self):
        self.vector_data = {}  # A dictionary to store vectors
        self.vector_index = {}  # An indexing structure for retrieval

    def add_vector(self, vector_id, vector):
        """
        Add a vector to the store.
        """
        self.vector_data[vector_id] = vector
        self._update_index(vector_id, vector)

    def get_vector(self, vector_id):
        """
        Retrieve a vector from the store.
        """
        return self.vector_data.get(vector_id)

    def _update_index(self, vector_id, vector):
        """
        Update the index with the new vector.

        """
        # In this simple example, we use brute-force cosine similarity for indexing
        for existing_id, existing_vector in self.vector_data.items():
            similarity = np.dot(vector, existing_vector) / (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity

    def find_similar_vectors(self, query_vector, num_results=3):
        """
        Find similar vectors to the query vector using brute-force search.
        Returns:
            list: A list of (vector_id, similarity_score) tuples for the most similar vectors.
        """
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity))

        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results
        return results[:num_results]

    def find_similar_vectors_with_embeddings(self, query_vector, num_results=5):
        """
        Find similar vectors to the query vector using brute-force search.
        Returns:
            list: A list of (vector_id, similarity_score, embedding) tuples for the most similar vectors.
        """
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity, vector))

        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results
        return results[:num_results]
