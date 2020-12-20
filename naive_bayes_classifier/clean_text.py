import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# find the category of word for further lemmatization
# reference: internet sources
def word_type(word):
    if word.startswith('J'):
        return wordnet.ADJ
    elif word.startswith('V'):
        return wordnet.VERB
    elif word.startswith('N'):
        return wordnet.NOUN
    elif word.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# lemmatize the text into base words
def text_lemmatizer(word):
    # initialize the word lemmatizer
    lemmatizer = WordNetLemmatizer().lemmatize
    word = word_tokenize(str(word))
    word_pos = pos_tag(word)
    lemmatized_words = [lemmatizer(w[0], word_type(w[1])) for w in word_pos]
    return ' '.join(lemmatized_words)


# clean the tweet data
# remove @mention of tweeter handles
# remove '#' tags
# remove retweets
# remove hyperlink
# remove numbers in tweet
# remove extra white spaces
# lemmatize the words
# rejoin the clean words in string format again
def cleanText(tweet):
    word_list = tweet.lower().split()
    stop_list = set(stopwords.words("english"))
    important_words = [w for w in word_list if not w in stop_list]

    clean_word_list = []
    for word in important_words:
        word = re.sub('@[A-Za-z0–9]+', '', word)  # Removing @mentions
        word = re.sub('#', '', word)  # Removing '#' hash tag
        word = re.sub('RT[\s]+', '', word)  # Removing RT
        word = re.sub('https?:\/\/\S+', '', word)  # Removing hyperlink
        word = re.sub('\d+', '', word)  # remove number
        word = re.sub(r'[^a-zA-Z]', '', word)
        word = re.sub(r'\s+', ' ', word)  # remove white space
        word = text_lemmatizer(word)
        clean_word_list.append(word)

    return " ".join(clean_word_list)
