# text_summarizer/summarizer.py

from sentence_transformers import SentenceTransformer, util
import numpy as np

class ExtractiveSummarizer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def summarize(self, text, top_k=3):
        sentences = text.split('. ')
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        query_embedding = self.model.encode([text], convert_to_tensor=True)

        scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_results = np.argpartition(-scores, range(top_k))[:top_k]

        summary = '. '.join([sentences[idx] for idx in top_results])
        return summary
