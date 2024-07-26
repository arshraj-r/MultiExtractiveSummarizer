from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

class KMeansSummarizer:
    def __init__(self, elbow_method=None,num_centroids=2):
        self.elbow_method = elbow_method
        self.num_centroids=num_centroids

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
            self.num_centroids = kwargs['num_sentences']

        kmeans = KMeans(n_clusters=self.num_centroids,random_state=34)
        kmeans.fit(embeddings)
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        closest_sentences_indices  = []
        for i in range(self.num_centroids):
            cluster_indices = np.where(cluster_labels == i)[0]
            centroid = cluster_centers[i]
            closest_sentence_index = cluster_indices[np.argmin(np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1))]
            # closest_sentences.append(sentences[closest_sentence_index])
            closest_sentences_indices.append(closest_sentence_index)
        
        # Sort the indices to maintain the original order of sentences
        closest_sentences_indices.sort()

        closest_sentences = [sentences[idx] for idx in closest_sentences_indices]

        summary = ' '.join(closest_sentences)
        return summary
