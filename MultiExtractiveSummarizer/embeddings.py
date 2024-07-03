from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self, method='sbert'):
        self.method = method
        if method == 'sbert':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')


    def fit_transform(self, texts):
        if self.method == 'sbert':
            return self.model.encode(texts)
