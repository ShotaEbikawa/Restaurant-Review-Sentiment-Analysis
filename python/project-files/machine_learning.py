# -*- coding: utf-8
import pandas as pd
import matplotlib as lib
import numpy as np

# imports csv file of restaurant-review-labeled-data.csv
restaurant_labeled = pd.read_csv('restaurant-review-labeled-data.csv', dtype=str)

# line 9 and 10 removes
# the row with nan, or the row with null values
# in the dataset
restaurant_labeled = restaurant_labeled[pd.notnull(restaurant_labeled['sentences__text'])]
restaurant_labeled = restaurant_labeled[pd.notnull(restaurant_labeled['sentences__Opinions__Opinion__polarity'])]
# we will ignore all the row that contains 'neutral' in the positive/negative column
# because we don't need additional label in our dataset
restaurant_labeled = restaurant_labeled[restaurant_labeled['sentences__Opinions__Opinion__polarity'] != 'neutral']

# X stores the text review column of the dataset
# Y stores the positive/negative label of the dataset
X = restaurant_labeled.iloc[1:, 3:4]
Y = restaurant_labeled.iloc[1:, 11:12]

# LabelEncoder is used for converting string label into binary number
# In this case, we're converting positive to 1
# and we're also converting negative to 0
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
Y = encoder.fit_transform(Y.values)

# nltk is natural language processing library in python
# I initialized store_words(which is a list) that stores
# all of the stopwords that the library has
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
stop_words = nltk.corpus.stopwords.words('english')
# real_token(which is a list) stores tokens with non-stop words
real_token = []
# new_X(which is a list) stores real_token
new_X = []
# iterates through a column containing text reviews
lemmatize = WordNetLemmatizer()
stemmer = PorterStemmer()
for review in X['sentences__text']:
    # make real_token empty every time new iteration starts
    real_token = []
    # tokens stores the tokenized version
    # of the given text review(which is a list)
    tokens = nltk.word_tokenize(review)
    # iterate through the tokenized text review
    for t in tokens:
        # real_token appends t
        # ONLY IF t is not a stop word
        if t not in stop_words:
            # t = stemmer.stem(t)
            t = lemmatize.lemmatize(t)
            real_token.append(t.lower())
    # the text review will be reinitialized to real_token
    review = real_token
    # new_X stores reinitialized review
    new_X.append(review)
# reinitialize X to new_X
X = new_X




# featuring is done in this section of code.
# initialized BOW_Vector, which is CountVectorizer
# initialized BOW_Matrix, which stores fit and transformed X
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
BOW_Vector = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda doc: doc, lowercase=False, min_df = 0., max_df = 1.)
BOW_Matrix = BOW_Vector.fit_transform(X)
features = BOW_Vector.get_feature_names()
BOW_Matrix = BOW_Matrix.toarray()
BOW_df = pd.DataFrame(BOW_Matrix, columns = features)
#tfidf = TfidfVectorizer()
#X_tfidf = tfidf.fit_transform(X_train)
#features = tfidf.get_feature_names()
#X_tfidf = X_tfidf.toarray()
#X_df = pd.DataFrame(X_tfidf, columns = features)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(BOW_df, Y.ravel(), test_size = 0.25, random_state = 0)
#X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y.ravel(), test_size = 0.25, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
multinomialNB = MultinomialNB()
multinomialNB.fit(X_train, Y_train)
y_pred = multinomialNB.predict(X_test)
#randomForest = RandomForestClassifier(n_estimators = 300)
#randomForest.fit(X_train, Y_train)
#y_pred = randomForest.predict(X_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, Y_test))


