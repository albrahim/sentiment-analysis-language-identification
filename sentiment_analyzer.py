import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess_text import preprocess_text


class SentimentAnalyzer:
    def __init__(self):
        self._clf = None

        columns = ['sentiment', 'text']
        positive_tweets = pd.read_csv(
            './resources/Arabic_Sentiment_Twitter_Corpus/train_Arabic_tweets_positive_20190413.tsv', sep='\t',
            encoding='utf-8', header=None, names=columns)

        negative_tweets = pd.read_csv(
            './resources/Arabic_Sentiment_Twitter_Corpus/train_Arabic_tweets_negative_20190413.tsv', sep='\t',
            encoding='utf-8', header=None, names=columns)

        tweets = pd.concat([positive_tweets, negative_tweets]).sample(frac=1, random_state=0).reset_index(drop=True)

        X_train = tweets['text'].astype(str)
        y_train = tweets['sentiment'].astype(str)

        text_clf = Pipeline([
            ('vect', CountVectorizer(
                analyzer='word',
                preprocessor=preprocess_text,
            )),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ])
        text_clf.fit(X_train, y_train)
        self._clf = text_clf

    def sentiment_of(self, text):
        docs_new = [text]
        prediction = self._clf.predict(docs_new)[0]
        return prediction

    def test(self):
        columns = ['sentiment', 'text']
        positive_tweets = pd.read_csv(
            './resources/Arabic_Sentiment_Twitter_Corpus/test_Arabic_tweets_positive_20190413.tsv', sep='\t',
            encoding='utf-8', header=None, names=columns)
        negative_tweets = pd.read_csv(
            './resources/Arabic_Sentiment_Twitter_Corpus/test_Arabic_tweets_negative_20190413.tsv', sep='\t',
            encoding='utf-8', header=None, names=columns)
        tweets = pd.concat([positive_tweets, negative_tweets]).sample(frac=1, random_state=0).reset_index(drop=True)

        X_test = tweets['text'].astype(str)
        y_true = tweets['sentiment'].astype(str)
        y_pred = self._clf.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Sentiment Accuracy Score: {accuracy:.2f}')
