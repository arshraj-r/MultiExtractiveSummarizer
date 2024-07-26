from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

class KMeansSummarizer2:
    def __init__(self, elbow_method=None,num_sentences=2):
        self.elbow_method = elbow_method
        self.num_sentences = num_sentences

    def elbow_method(self, embeddings):
        wcss = []
        for i in range(1, 4):
            kmeans = KMeans(n_clusters=i, random_state=0)
            kmeans.fit(embeddings)
            wcss.append(kmeans.inertia_)
        # Plot the Elbow graph to manually determine the optimal number of clusters
        plt.plot(range(1, 4), wcss)
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        plt.show()

        optimal_clusters = np.argmin(np.diff(np.diff(wcss))) + 2
        return optimal_clusters

    def summarize(self, sentences, embeddings,**kwargs):

        if 'num_sentences' in kwargs:
            self.num_sentences = kwargs['num_sentences']

        # Determine the number of clusters based on the number of sentences
        if len(sentences) < 10:
            num_clusters  = 2
        else:
            num_clusters  = 3

        kmeans = KMeans(n_clusters=num_clusters , random_state=34)
        kmeans.fit(embeddings)
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        # Calculate distances from each sentence to its cluster centroid
        distances = []
        for i in range(len(sentences)):
            cluster_idx = cluster_labels[i]
            centroid = cluster_centers[cluster_idx]
            distance = np.linalg.norm(embeddings[i] - centroid)
            distances.append((distance, i))

        # Sort sentences by distance to cluster centroid
        distances.sort()

        # Select the top `num_sentences` closest sentences
        top_sentence_indices = [idx for _, idx in distances[:self.num_sentences]]

        # Sort the indices to maintain the original order of sentences
        top_sentence_indices.sort()

        # Extract the sentences in the original order
        summary = ' '.join([sentences[idx] for idx in top_sentence_indices])

        return summary
