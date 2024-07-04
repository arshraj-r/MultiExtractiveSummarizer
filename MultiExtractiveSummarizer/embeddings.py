from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self, method):
        self.method = method
        if method == 'sbert':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            raise ValueError("Model not found")


    def fit_transform(self, texts):
        if self.method == 'sbert':
            return self.model.encode(texts)
