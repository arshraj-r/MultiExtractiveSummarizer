# MultiExtractiveSummarizer

MultiExtractiveSummarizer is a Python package for extractive text summarization using various embeddings and summarization methods. It provides flexibility to choose different word embeddings (e.g., TF-IDF, Doc2Vec, GloVe, Sentence-BERT) and summarization algorithms (e.g., TextRank, LexRank, LSA, K-Means clustering) to generate concise summaries from text documents.

## Features

- Support for multiple word embeddings from sentence-bert.
- Implementations of various extractive summarization algorithms such as TextRank, LexRank, LSA, and K-Means clustering.
- Customizable summarization parameters like number of sentences, ratio of summary length, and clustering parameters.

## Installation

You can install MultiExtractiveSummarizer from PyPI using pip:

```bash
pip install MultiExtractiveSummarizer==0.2.0
```

## Usage

```python 
from MultiExtractiveSummarizer import MESummarizer

text="your imput text"
summarizer = MESummarizer(embedding_method='sbert', summarization_method='lexrank')
summary = summarizer.summarize(text, ratio=.5)
print(summary)

text="your input text"
summarizer = MESummarizer(embedding_method='sbert', summarization_method='lexrank')
summary = summarizer.summarize(text, num_sentences=.5)
```
