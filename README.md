# MultiExtractiveSummarizer

`MultiExtractiveSummarizer` is a Python package designed for extractive text summarization. It leverages advanced embedding techniques and sentence ranking algorithms to provide high-quality summaries of text documents. This package currently includes embedding methods from SBERT and TF-IDF, and sentence ranking using LexRank and K-means clustering. Future updates will include additional embedding methods like Word2Vec, GloVe, and BERT embeddings, as well as other sentence ranking algorithms such as TextRank and KLA.

## Table of Contents

- [Installation](#installation)
- [Description](#description)
- [Features](#features)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Advanced Usage](#advanced-usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Installation

You can install MultiExtractiveSummarizer from PyPI using pip:

```bash
pip install MultiExtractiveSummarizer
```

## Description

### Extractive Summarization

Extractive summarization involves selecting sentences from a document to create a summary that retains the most important information. Unlike abstractive summarization, which generates new sentences, extractive summarization works by identifying and extracting existing sentences.

### Embedding Methods

1. **SBERT (Sentence-BERT)**: SBERT is a modification of BERT that uses Siamese and triplet networks to derive semantically meaningful sentence embeddings.
2. **TF-IDF (Term Frequency-Inverse Document Frequency)**: TF-IDF is a numerical statistic that reflects the importance of a word in a document relative to a corpus.

### Sentence Ranking Methods

1. **LexRank**: LexRank is a graph-based algorithm for computing sentence importance based on eigenvector centrality in a similarity graph.
2. **K-means Clustering**: K-means is a clustering algorithm that partitions sentences into k clusters, and representative sentences from each cluster are selected for the summary.
3. **K-means Clustering-2** : KMeans clustering algorithm to identify key sentences in a document. The process involves the following steps:
-Cluster Determination: Based on the number of sentences in the document, the number of clusters is set to 2 if there are fewer than 10 sentences, and 3 if there are 10 or more sentences.
-Clustering: The KMeans algorithm is applied to the sentence embeddings, grouping the sentences into clusters based on their similarity.
-Distance Calculation: For each sentence, the Euclidean distance to its respective cluster centroid is calculated.
-Sentence Selection: The sentences are then ranked based on their proximity to the cluster centroids. The top num_sentences closest sentences are selected for the summary.
-Original Order: The selected sentences are sorted to maintain their original order in the document, ensuring a coherent summary.

## Features

- **Flexible Embedding Methods**: Choose between SBERT and TF-IDF for embedding sentences.
- **Multiple Sentence Ranking Algorithms**: Use LexRank or K-means clustering to rank sentences and create summaries.
- **Modular and Extensible**: Designed to easily incorporate new embedding methods and ranking algorithms.

## Usage

### Basic Usage

Here's an example of how to use the `MultiExtractiveSummarizer` package to create a summary of a text document.

```python
from MultiExtractiveSummarizer import MultiExtractiveSummarizer

# Initialize the summarizer
summarizer = MultiExtractiveSummarizer(embedding_method='sbert', summarization_method='lexrank')

# Example text document
text = """
Your text document goes here...
"""

# Generate the summary with number of sentences
summary = summarizer.summarize(text, num_sentences=5)

print("Summary:")
print(summary)

# Generate the summary ratio of text
summary = summarizer.summarize(text, ratio=0.5)

print("Summary:")
print(summary)
```
### Advanced Usage

For advanced usage, you can specify different parameters for embedding methods and sentence ranking algorithms.

```python
from MultiExtractiveSummarizer import MultiExtractiveSummarizer

# Initialize the summarizer with TF-IDF and K-means
summarizer = MultiExtractiveSummarizer(embedding_method='tfidf', summarization_method='kmeans')

# Example text document
text = """
Your long text document goes here...
"""

# Generate the summary
summary = summarizer.summarize(text, num_sentences=5)

print("Summary:")
print(summary)
```
## Future Work

I plan to expand the capabilities of the `MultiExtractiveSummarizer` package by including:

- Additional embedding methods: Word2Vec, GloVe, and BERT embeddings.
- New sentence ranking algorithms: TextRank, KLA, and others.

Stay tuned for updates and new features!

## Contributing

We welcome contributions from the community. If you have suggestions or would like to contribute, please fork the repository and create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

