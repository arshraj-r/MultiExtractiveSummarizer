from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer as SumyLexRankSummarizer

class LexRankSummarizer:
    def __init__(self):
        self.parser = PlaintextParser.from_string('', Tokenizer('english'))
        self.summarizer = SumyLexRankSummarizer()

    def summarize(self, text, num_sentences=3):
        self.parser = PlaintextParser.from_string(text, Tokenizer('english'))
        summary = self.summarizer(self.parser.document, num_sentences)
        return ' '.join([str(sentence) for sentence in summary])
