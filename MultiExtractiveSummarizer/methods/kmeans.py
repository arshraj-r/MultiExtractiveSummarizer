from sklearn.cluster import KMeans
import numpy as np

class KMeansSummarizer:
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters

    def summarize(self, sentences, embeddings):
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(embeddings)
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        closest_sentences = []
        for i in range(self.num_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            centroid = cluster_centers[i]
            closest_sentence_index = cluster_indices[np.argmin(np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1))]
            closest_sentences.append(sentences[closest_sentence_index])

        summary = '. '.join(closest_sentences)
        return summary
