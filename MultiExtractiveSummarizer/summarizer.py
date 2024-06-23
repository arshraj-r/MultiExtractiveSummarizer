# quicksumm/summarizer.py

from .embeddings import Embeddings
from .methods.kmeans import KMeansSummarizer
from .methods.textrank import TextRankSummarizer
# from .methods.sumy import SumySummarizer
from .methods.lexrank import LexRankSummarizer
# from .methods.lsa import LSASummarizer
# from .methods.luhn import LuhnSummarizer
# from .methods.edmundson import EdmundsonSummarizer
# from .methods.textteaser import TextTeaserSummarizer
# from .methods.sumbasic import SumBasicSummarizer
# from .methods.klsum import KLSumSummarizer
# from .methods.submodular import SubmodularSummarizer

class MultiExtractiveSummarizer:
    def __init__(self, embedding_method='sbert', summarization_method='kmeans'):
        self.embedding_method = embedding_method
        self.summarization_method = summarization_method
        self.embeddings = Embeddings(method=embedding_method)
        
        summarization_methods = {
            'kmeans': KMeansSummarizer,
            'textrank': TextRankSummarizer,
            'sumy': SumySummarizer,
            'lexrank': LexRankSummarizer,
            'lsa': LSASummarizer,
            'luhn': LuhnSummarizer,
            'edmundson': EdmundsonSummarizer,
            'textteaser': TextTeaserSummarizer,
            'sumbasic': SumBasicSummarizer,
            'klsum': KLSumSummarizer,
            'submodular': SubmodularSummarizer
        }
        
        if summarization_method in summarization_methods:
            self.summarizer = summarization_methods[summarization_method]()
        else:
            raise ValueError(f"Summarization method {summarization_method} not supported.")
    
    def summarize(self, text, **kwargs):
        sentences = text.split('. ')
        embeddings = self.embeddings.fit_transform(sentences)
        
        return self.summarizer.summarize(sentences, embeddings, **kwargs)
