import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from is_arabic_dialect import is_arabic_dialect


class LanguageIdentifier:
    def __init__(self):
        self._clf = None

        train_data = pd.read_csv(
            './resources/language-identification/train.csv',
            encoding='utf-8')

        X_train = train_data['text'].astype(str)
        y_train = train_data['labels'].astype(str)

        text_clf = Pipeline([
            ('vect_lang', CountVectorizer(
                analyzer='word',
            )),
            ('tfidf_lang', TfidfTransformer()),
            ('clf_lang', MultinomialNB())
        ])
        text_clf.fit(X_train, y_train)

        self._clf = text_clf

    def language_of(self, text):
        docs_new = [text]
        proba = self._clf.predict_proba(docs_new)[0]
        valid_prediction = max(proba) != min(proba)
        if not valid_prediction:
            if is_arabic_dialect(text):
                return 'ar'
            return None
        return self._clf.predict(docs_new)[0]

    def test(self):
        test_data = pd.read_csv(
            'resources/language-identification/test.csv',
            encoding='utf-8')

        X_test = test_data['text'].astype(str)
        y_true = test_data['labels'].astype(str)
        y_pred = self._clf.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Language Identification Accuracy Score: {accuracy:.2f}')
