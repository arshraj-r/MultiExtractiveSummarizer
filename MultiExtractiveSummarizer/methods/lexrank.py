import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import softmax
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class LexRankSummarizer:
    def __init__(self, threshold=None, increase_power=True):
        self.threshold = threshold
        self.increase_power = increase_power

    def degree_centrality_scores(self, similarity_matrix):
        if not (self.threshold is None or isinstance(self.threshold, float) and 0 <= self.threshold < 1):
            raise ValueError("'threshold' should be a floating-point number from the interval [0, 1) or None")

        if self.threshold is None:
            markov_matrix = self.create_markov_matrix(similarity_matrix)
        else:
            markov_matrix = self.create_markov_matrix_discrete(similarity_matrix)

        scores = self.stationary_distribution(markov_matrix)

        return scores

    def _power_method(self, transition_matrix, max_iter=10000):
        eigenvector = np.ones(len(transition_matrix))

        if len(eigenvector) == 1:
            return eigenvector

        transition = transition_matrix.transpose()

        for _ in range(max_iter):
            eigenvector_next = np.dot(transition, eigenvector)

            if np.allclose(eigenvector_next, eigenvector):
                return eigenvector_next

            eigenvector = eigenvector_next

            if self.increase_power:
                transition = np.dot(transition, transition)

        logger.warning("Maximum number of iterations for power method exceeded without convergence!")
        return eigenvector_next

    def connected_nodes(self, matrix):
        _, labels = connected_components(matrix)

        groups = []

        for tag in np.unique(labels):
            group = np.where(labels == tag)[0]
            groups.append(group)

        return groups

    def create_markov_matrix(self, weights_matrix):
        n_1, n_2 = weights_matrix.shape
        if n_1 != n_2:
            raise ValueError("'weights_matrix' should be square")

        row_sum = weights_matrix.sum(axis=1, keepdims=True)

        # normalize probability distribution differently if we have negative transition values
        if np.min(weights_matrix) <= 0:
            return softmax(weights_matrix, axis=1)

        return weights_matrix / row_sum

    def create_markov_matrix_discrete(self, weights_matrix):
        discrete_weights_matrix = np.zeros(weights_matrix.shape)
        ixs = np.where(weights_matrix >= self.threshold)
        discrete_weights_matrix[ixs] = 1

        return self.create_markov_matrix(discrete_weights_matrix)

    def stationary_distribution(self, transition_matrix):
        n_1, _ = transition_matrix.shape
        distribution = np.zeros(n_1)

        grouped_indices = self.connected_nodes(transition_matrix)

        for group in grouped_indices:
            t_matrix = transition_matrix[np.ix_(group, group)]
            eigenvector = self._power_method(t_matrix)
            distribution[group] = eigenvector

        distribution /= n_1

        return distribution


    def summarize(self,sentences, embeddings):
        similarity_scores = cosine_similarity(embeddings)
        centrality_scores = self.degree_centrality_scores(similarity_scores)
        most_central_sentence_indices = np.argsort(-centrality_scores)
        most_central_sentence_indices_sorted=sorted(most_central_sentence_indices)
        summary_indicies=most_central_sentence_indices_sorted[0:2] #adjust the length here
        summary_sentences=[sentences[idx].strip() for idx in summary_indicies]
        return " ".join(summary_sentences)
