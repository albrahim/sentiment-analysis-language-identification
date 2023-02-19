from info import language_flags, language_names, sentiment_emojis, sentiment_names
from language_identifier import LanguageIdentifier
from sentiment_analyzer import SentimentAnalyzer
import PySimpleGUIWx as sg

from sentiment_analyzer_multilingual import SentimentAnalyzerMultilingual, is_valid_language_label

EMPTY_TEXT = ''


def main():
    language_identifier = LanguageIdentifier()
    language_identifier.test()
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.test()
    sentiment_analyzer_multilingual = SentimentAnalyzerMultilingual()

    layout = [
        [sg.Text("Enter a text", font='Any 15')],
        [sg.Multiline(font='Any 15', size=(30, 5)), sg.Button('Predict', font='Any 15', size=(3, 5))],
        [sg.Text(EMPTY_TEXT, key='predicted-language-flag', size=(20, 2), font='Any 40', justification='c'),
         sg.Text(EMPTY_TEXT, key='predicted-sentiment-emoji', size=(20, 2), font='Any 40', justification='c')],
        [sg.Text(EMPTY_TEXT, key='predicted-language', size=(20, 1), font='Any 15', justification='c'),
         sg.Text(EMPTY_TEXT, key='predicted-sentiment', size=(20, 1), font='Any 15', justification='c')]
    ]

    window = sg.Window('Sentiment Analyzer & Language Identifier', layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'Predict':
            text = values[0]

            language_label = language_identifier.language_of(text)
            if language_label:
                window['predicted-language-flag'].update(language_flags[language_label])
                window['predicted-language'].update(language_names[language_label])
            else:
                window['predicted-language-flag'].update(EMPTY_TEXT)
                window['predicted-language'].update(EMPTY_TEXT)

            if language_label == 'ar':
                sentiment_label = sentiment_analyzer.sentiment_of(text)
                window['predicted-sentiment-emoji'].update(sentiment_emojis[sentiment_label])
                window['predicted-sentiment'].update(sentiment_names[sentiment_label])
            elif is_valid_language_label(language_label):
                sentiment_label = sentiment_analyzer_multilingual.sentiment_of(text, language_label)
                window['predicted-sentiment-emoji'].update(sentiment_emojis[sentiment_label])
                window['predicted-sentiment'].update(sentiment_names[sentiment_label])
            else:
                window['predicted-sentiment-emoji'].update(EMPTY_TEXT)
                window['predicted-sentiment'].update(EMPTY_TEXT)

    window.close()


if __name__ == '__main__':
    main()
