import pandas as pd
from sklearn.decomposition import PCA

from milestone_2 import clean_text
from milestone_2 import tf_idf
from milestone_2 import naive_bayes
from milestone_2 import results


# main method to load data and call all other methods for classification
if __name__ == '__main__':
    # read dataset from file
    # remove all unnecessary columns
    # print dataframe head
    df = pd.read_csv('Tweets.csv')
    df = df.loc[:, ['airline_sentiment', 'text']]
    print(df.head(5))

    # clean all the tweets for further processing
    df['text'] = df['text'].apply(lambda x: clean_text.cleanText(x))
    # change the resulting classes from words to integers for ease of classification
    df['label_nb'] = df['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

    # transform text into a meaningful representation of numbers using TF-IDF
    vector_df = tf_idf.vectorizer(df['text'])

    # find important principal components
    # reduce the dimensionality of data and then converting it to dataframe
    pca = PCA(n_components=50)
    principalComponents = pca.fit_transform(vector_df)

    columns = ['pca_%i' % i for i in range(50)]
    principalDf = pd.DataFrame(data=principalComponents, columns=columns)
    principalDf['label_nb'] = df['label_nb']

    # split training and testing data
    X_train = principalDf.sample(frac=0.8, random_state=0)
    X_test = principalDf.drop(X_train.index)
    y_test = X_test['label_nb']
    del X_test['label_nb']
    print(X_train.shape, X_test.shape)

    # call naive bayes classifier
    model, prior = naive_bayes.fit(X_train)
    y_pred = naive_bayes.predict(model, prior, X_test)
    predictions = pd.DataFrame(y_pred, columns=['target'])

    # calculate confusion matrix and result
    results.print_result(y_pred, y_test.tolist())
