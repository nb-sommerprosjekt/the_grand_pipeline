import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import snowball

def remove_punctuation(text):
    text = re.sub('[^a-zA-ZæøåÆØÅ]+', ' ', text)
    text = text.replace("  ", " ")
    return text

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in set(stopwords.words('norwegian'))]

    text = ' '.join(filtered_words)
    return text

def stem_text(text,norStem):
    tokens = word_tokenize(text)


    stemmed_words = list()
    for word in tokens:
        stemmed_words.append(norStem.stem(word))

    text = ' '.join(stemmed_words)
    return text