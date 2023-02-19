import re
import pandas as pd

r = re.compile(r'[\W_]+', re.U)
rd = re.compile(r'[\d]+', re.U)
stop_list = frozenset(pd.read_table('./resources/multi-dialect-arabic-stop-words/Stop-words/stop_list_1177.txt',
                          header=None, encoding='utf-8')[0])


FATHAH = "ً"
KASRA = "ِ"
THAMMAH = "ُ"
TANWEEN_FATH = "ً"
TANWEEN_KASR = "َ"
TANWEEN_THAMM = "ٌ"
SHADDAH = "ّ"
SUKOON = "ْ"
HAMZA = "ء"
HAMZA_ALIF = "أ"
HAMZA_YAA = "ئ"
HAMZA_WAAW = "ؤ"
HAMZA_ALIF_TAHAT = "إ"
ALIF_MADD = "آ"
TAA_MARBUTA = "ة"
ALIF_MAQSURA = "ى"
TATWEEL = "ـ"
ALIF = "ا"
YAA = "ي"
WAAW = "و"
HAA = "ه"
EMPTY_STR = ""

PREFIX_YAA = "يا"
PREFIX_LIL = "لل"
PREFIX_LI = "ل"
PREFIX_MA = "ما"
PREFIX_LA = "لا"
PREFIX_BI = "ب"
PREFIX_AL = "ال"
PREFIX_HAL = "هال"
PREFIX_NUUN = "ن"
PREFIX_TAA = "ت"
SUFFIX_IIN = "ين"
SUFFIX_UUN = "ون"
SUFFIX_AAT = "ات"
SUFFIX_NAA = "نا"
SUFFIX_UK = "ك"
SUFFIX_UU1 = "وا"
SUFFIX_UU2 = "و"
SUFFIX_UU3 = "وو"
SUFFIX_KUM = "كم"
SUFFIX_HAA = "ها"

def extract_token_word(word):
    return word.removeprefix(WAAW).removeprefix(PREFIX_YAA).removeprefix(PREFIX_LIL).removeprefix(PREFIX_LIL).removeprefix(YAA).removeprefix(PREFIX_MA).removeprefix(PREFIX_LA).removeprefix(PREFIX_BI).removeprefix(PREFIX_NUUN).removeprefix(PREFIX_TAA).removeprefix(PREFIX_AL).removeprefix(PREFIX_HAL).removesuffix(YAA).removesuffix(SUFFIX_IIN).removesuffix(SUFFIX_UUN).removesuffix(SUFFIX_AAT).removesuffix(SUFFIX_NAA).removesuffix(SUFFIX_UK).removesuffix(SUFFIX_KUM).removesuffix(SUFFIX_UU1).removesuffix(SUFFIX_UU2).removesuffix(SUFFIX_UU3).removesuffix(SUFFIX_HAA)

def preprocess_text(text):

    words0 = [e for e in text.split(' ') if e not in stop_list]
    text0 = ' '.join(words0)
    text1 = text0\
        .replace(FATHAH, EMPTY_STR) \
        .replace(KASRA, EMPTY_STR) \
        .replace(THAMMAH, EMPTY_STR) \
        .replace(TANWEEN_FATH, EMPTY_STR) \
        .replace(TANWEEN_KASR, EMPTY_STR) \
        .replace(TANWEEN_THAMM, EMPTY_STR) \
        .replace(SHADDAH, EMPTY_STR) \
        .replace(SUKOON, EMPTY_STR) \
        .replace(HAMZA, EMPTY_STR) \
        .replace(HAMZA_ALIF, ALIF) \
        .replace(HAMZA_YAA, YAA) \
        .replace(HAMZA_WAAW, WAAW) \
        .replace(HAMZA_ALIF_TAHAT, ALIF) \
        .replace(ALIF_MADD, ALIF) \
        .replace(TAA_MARBUTA, HAA) \
        .replace(ALIF_MAQSURA, YAA) \
        .replace(TATWEEL, EMPTY_STR)
    text2 = rd.sub('', text1)
    text3 = r.sub(' ', text2) \
        .strip(' ')

    words1 = [extract_token_word(e) for e in text3.split(' ') if e not in stop_list]
    words2 = [e for e in words1 if e not in stop_list]
    words3 = [e for e in words2 if e != '']
    text4 = ' '.join(
        words3
    )
    return text4

def preprocess_text_multilingual(text):
    text1 = str.lower(text)
    text2 = rd.sub('', text1)
    text3 = r.sub(' ', text2) \
        .strip(' ')
    return text3