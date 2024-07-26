from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

class Embeddings:
    def __init__(self, method):
        self.method = method
        if method == 'sbert':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        elif method == 'tfidf':
            self.model = TfidfVectorizer()
        else:
            raise ValueError("Model not found")
        
    def fit_transform(self, texts):
        if self.method == 'sbert':
            return self.model.encode(texts)
        elif self.method == 'tfidf':
            return self.model.fit_transform(texts).toarray()
