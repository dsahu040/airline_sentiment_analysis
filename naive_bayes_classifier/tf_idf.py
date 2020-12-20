import pandas as pd
import math


# calculate the IDF values for each unique word
def idf(unique_words, bagOfWords):
    n = len(bagOfWords)
    idf_val = {}
    for word in unique_words:
        doc_containing_word = 0
        for bagOfWord in bagOfWords:
            if word in bagOfWord:
                doc_containing_word += 1
        idf_val[word] = math.log(n / doc_containing_word)
    return idf_val


# calculate the TF-IDF for all the tweets
# calculate the TF for words and multiply it with corresponding IDF values
def tf_idf(words, bagOfWords, idf_val):
    tf_idf_list = []
    for bagOfWord in bagOfWords:
        tf_idt_dict = dict.fromkeys(words, 0.0)
        size_of_document = len(bagOfWord)
        my_dict = {i:bagOfWord.count(i) for i in bagOfWord}
        for word, count in my_dict.items():
            tf_val = 0.5 + 0.5*(float(count)/size_of_document)
            tf_idt_dict[word] = tf_val * idf_val[word]
        tf_idf_list.append(tf_idt_dict)
    return tf_idf_list


# vectorize all the tweets using TF-IDF
# convert the list of TF-IDF values in data frame for further classification
def vectorizer(documents):
    # create bag of words
    bagOfWords = []
    # set of all the unique words in the corpus
    unique = set()
    for document in documents:
        bagOfWord = document.split(' ')
        bagOfWords.append(bagOfWord)
        for word in bagOfWord:
            unique.add(word)

    idfs = idf(unique, bagOfWords)
    tf_idfs = tf_idf(idfs.keys(), bagOfWords, idfs)
    print("number of rows: " + str(len(tf_idfs)) + "\nnumer of cols: " + str(len(tf_idfs[0])))

    # convert list to data frame
    feature_names = list(unique)
    df = pd.DataFrame(tf_idfs, columns=feature_names)
    return df

