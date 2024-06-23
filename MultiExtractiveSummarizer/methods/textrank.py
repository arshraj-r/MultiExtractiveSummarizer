from gensim.summarization import summarize

class TextRankSummarizer:
    def summarize(self, text, ratio=0.2):
        return summarize(text, ratio=ratio)
