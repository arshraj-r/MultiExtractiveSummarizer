from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import gensim.downloader as api

class Embeddings:
    def __init__(self, method='sbert'):
        self.method = method
        if method == 'sbert':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        elif method == 'glove':
            self.model = api.load("glove-wiki-gigaword-100")
        elif method == 'doc2vec':
            self.model = Doc2Vec(vector_size=100, min_count=2, epochs=40)
        elif method == 'tfidf':
            self.model = TfidfVectorizer()

    def fit_transform(self, texts):
        if self.method == 'sbert':
            return self.model.encode(texts)
        elif self.method == 'glove':
            return np.array([np.mean([self.model[word] for word in text.split() if word in self.model] or [np.zeros(100)], axis=0) for text in texts])
        elif self.method == 'doc2vec':
            tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]
            self.model.build_vocab(tagged_data)
            self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            return np.array([self.model.dv[str(i)] for i in range(len(texts))])
        elif self.method == 'tfidf':
            return self.model.fit_transform(texts).toarray()
