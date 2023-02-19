import pandas as pd

dialect_speech = frozenset(pd.read_table('./resources/multi-dialect-arabic-stop-words/Stop-words/stop_list_1177.txt',
                                         header=None, encoding='utf-8')[0])

def is_arabic_dialect(text):
    words = frozenset(text.split(' '))
    intersection = words.intersection(dialect_speech)
    intersection_count = len(intersection)
    is_dialect = intersection_count > 0
    return is_dialect
