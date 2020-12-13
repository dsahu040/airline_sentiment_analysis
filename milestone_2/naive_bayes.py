import operator
import math
import numpy as np


# training method
# calculate mean and standard deviation for all features
# calculate class prior probability for each sentiment class
def fit(X_train):
    n = len(X_train)

    # segregate input rows of vectors based on sentiment classes
    sentiment_dict = {}
    for i, row in X_train.iterrows():
        if row[-1] not in sentiment_dict:
            sentiment_dict[row[-1]] = []
        sentiment_dict[row[-1]].append(row[:-1])

    # find prior probability
    # find mean and standard deviation
    prior_prob = {}
    model = {}
    for sentiment_key, rows in sentiment_dict.items():
        prior_prob[sentiment_key] = len(rows) / float(n)
        model[sentiment_key] = [(np.mean(column), np.std(column)) for column in zip(*rows)]
    return model, prior_prob


# predicting the sentiment classes for input
def predict(model, prior_prob, X_test):
    y_pred = []
    for index, row in X_test.iterrows():
        probabilities = {}

        for sentiment_key, rows in model.items():
            probabilities[sentiment_key] = prior_prob[sentiment_key]

            for i in range(len(rows)):
                mean, std_dev = rows[i]
                # probability density calculation
                probability = (1 / (math.sqrt(2 * math.pi) * std_dev)) * math.exp(
                    -((row[i] - mean) ** 2 / (2 * std_dev ** 2)))
                probabilities[sentiment_key] *= probability

        # find the best class value by comparing the posterior probabilities
        y_pred.append(max(probabilities.items(), key=operator.itemgetter(1))[0])

    return y_pred
