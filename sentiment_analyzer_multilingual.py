import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess_text import preprocess_text, preprocess_text_multilingual

valid_language_labels = frozenset(pd.read_csv(
    './resources/Sentiment-Lexicons-for-81-Languages/correctedMetadata.csv',
    encoding='utf-8', header=0)['Wikipedia.Language.Code'].astype(str)).union(['en'])

def is_valid_language_label(language_label):
    return language_label in valid_language_labels


def load_lexicon_file(language_label):
    assert is_valid_language_label(language_label), 'invalid language label'
    if language_label == 'en':
        columns = ['word', 'sentiment']
        words = pd.read_csv(
            './resources/Bing_NRC_Afinn_Lexicons/Bing.csv',
            encoding='utf-8', header=0, names=columns)
        positive_only = words[words['sentiment'] == 'positive']
        negative_only = words[words['sentiment'] == 'negative']
        positive_set = frozenset(positive_only['word'].astype(str))
        negative_set = frozenset(negative_only['word'].astype(str))
        return positive_set, negative_set
    else:
        columns = ['word']
        positive_only = pd.read_csv(
            f'./resources/Sentiment-Lexicons-for-81-Languages/sentiment-lexicons/positive_words_{language_label}.txt',
            encoding='utf-8', header=None, names=columns)
        negative_only = pd.read_csv(
            f'./resources/Sentiment-Lexicons-for-81-Languages/sentiment-lexicons/negative_words_{language_label}.txt',
            encoding='utf-8', header=None, names=columns)

        positive_set = frozenset(positive_only['word'].astype(str).map(preprocess_text_multilingual))
        negative_set = frozenset(negative_only['word'].astype(str).map(preprocess_text_multilingual))
        return positive_set, negative_set


class SentimentAnalyzerMultilingual:

    def __init__(self):
        self._clf = None
        self._lexicons = {}

    def load_language_lexicons(self, language_label):
        assert is_valid_language_label(language_label), 'invalid language label'
        positive_set, negative_set = load_lexicon_file(language_label)
        self._lexicons[language_label] = {}
        self._lexicons[language_label]['positive_set'] = positive_set
        self._lexicons[language_label]['negative_set'] = negative_set

    def sentiment_of(self, text, language_label):
        assert is_valid_language_label(language_label), 'invalid language label'

        if language_label not in self._lexicons:
            self.load_language_lexicons(language_label)

        words = preprocess_text_multilingual(text).split(' ')
        word_set = frozenset(words)
        positive_matches = word_set.intersection(self._lexicons[language_label]['positive_set'])
        negative_matches = word_set.intersection(self._lexicons[language_label]['negative_set'])
        positive_match_count = len(positive_matches)
        negative_match_count = len(negative_matches)

        if positive_match_count > negative_match_count:
            return 'pos'
        elif negative_match_count > positive_match_count:
            return 'neg'
        return 'neu'
