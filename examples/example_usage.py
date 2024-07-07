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
