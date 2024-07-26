from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

class KMeansSummarizer2:
    def __init__(self, elbow_method=None,ratio=None, num_sentences=None):
        self.elbow_method = elbow_method
        self.ratio = ratio
        self.num_sentences=num_sentences

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
    
    def calculate_num_sentences(self, total_sentences,ratio=None,num_sentences=None):
        if ratio is not None:
            return max(1, int(total_sentences * ratio))  # Ensure at least one sentence is included
        return num_sentences if num_sentences is not None else max(1, int(total_sentences * 0.5))  # Default to ratio=50%  if not provided

    def summarize(self, sentences, embeddings,num_sentences=None, ratio=None):

        self.num_sentences = self.calculate_num_sentences(len(sentences), ratio=ratio,num_sentences=num_sentences )

        # Determine the number of clusters based on the number of sentences
        if len(sentences)<=3:
            num_clusters=1
        elif len(sentences)>3 and len(sentences) <= 10:
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
