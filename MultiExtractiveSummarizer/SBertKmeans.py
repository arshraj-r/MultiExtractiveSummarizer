# quicksumm/summarizer.py

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans

class ExtractiveSummarizer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def summarize(self, text, num_clusters=3):
        sentences = text.split('. ')
        embeddings = self.model.encode(sentences)

        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(embeddings)
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        closest_sentences = []
        for i in range(num_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            centroid = cluster_centers[i]
            closest_sentence_index = cluster_indices[np.argmin(np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1))]
            closest_sentences.append(sentences[closest_sentence_index])

        summary = '. '.join(closest_sentences)
        return summary
