from .embeddings import Embeddings
from .methods.lexrank import LexRankSummarizer
import nltk

class MultiExtractiveSummarizer:
    def __init__(self, embedding_method, summarization_method):
        self.embedding_method = embedding_method
        self.summarization_method = summarization_method
        self.embeddings = Embeddings(method=embedding_method)
        
        summarization_methods = {
            'lexrank': LexRankSummarizer,
        }
        
        if summarization_method in summarization_methods:
            self.summarizer = summarization_methods[summarization_method]()
        else:
            raise ValueError(f"Summarization method {summarization_method} not supported.")
    
    def summarize(self, text, **kwargs):
        # sentences = text.split('. ')
        sentences = nltk.sent_tokenize(text)
        embeddings = self.embeddings.fit_transform(sentences)
        
        return self.summarizer.summarize(sentences, embeddings, **kwargs)
