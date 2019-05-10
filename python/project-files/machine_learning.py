# -*- coding: utf-8
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import re
#from copy import deepcopy
#from string import punctuation
#from random import shuffle
#from gensim.models.word2vec import Word2Vec
#from tqdm import tqdm 
from sklearn import preprocessing
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import LabelBinarizer
import nltk
import scipy.sparse
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
#from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
#from gensim.models import word2vec
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from Extractors import AverageWordLengthExtractor, ExclamationPointCountExtractor, UsefulValueExtractor, ColumnSelector

lemmatizer = WordNetLemmatizer()


def preprocessing(X):
    real_token = []
    stop_words = nltk.corpus.stopwords.words('english')
    punctuation =  [',','-','+','.','/','\\','\'','"','?','!','$','(',')','...',
                    '..', '--', '---',':','~','=','`','{','}','\n', ',']
    # new_X(which is a list) stores real_token
    new_X = []
    y = ''
    lemmatize = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    # iterates through a column containing text reviews
    for review in X.values:
        # make real_token empty every time new iteration starts
        real_token = []
        # tokens stores the tokenized version
        # of the given text review(which is a list)
        tokens = nltk.word_tokenize(str(review))
        # iterate through the tokenized text review
        for t in tokens:
            y = ''
            for ch in t:
                if ch in stop_words or ch in punctuation:
                    continue
                else:
                    y += ch;
            # real_token appends t
            # ONLY IF t is not a stop word
            if y not in stop_words and y not in punctuation:
                #t = stemmer.stem(t)
                t = lemmatize.lemmatize(y)
                real_token.append(t.lower())
        # the text review will be reinitialized to real_token
        review = real_token
        # new_X stores reinitialized review
        new_X.append(review)
    # reinitialize X to new_X
    X = new_X
    return X


def pre_process(s):
    pass
    

def featuring(X, feature_name):
    if (feature_name == 'BOW'):
        BOW_Vector = CountVectorizer(ngram_range=(1, 3), tokenizer=lambda doc: doc,
                                     lowercase=False, min_df = 0., max_df = 1., max_features = 5581)
        BOW_Matrix = BOW_Vector.fit_transform(X)
        features = BOW_Vector.get_feature_names()
        BOW_Matrix = BOW_Matrix.toarray()
        BOW_df = pd.DataFrame(BOW_Matrix, columns = features)
        return BOW_df
    if (feature_name == 'TFIDF'): 
        tfidf = TfidfVectorizer(ngram_range=(1,2), lowercase=False, 
                                analyzer = 'word', tokenizer=lambda doc: doc, 
                                min_df=0., max_df=1.)
        X_tfidf = tfidf.fit_transform(X)
        features = tfidf.get_feature_names()
        X_tfidf = X_tfidf.toarray()
        X_df = pd.DataFrame(X_tfidf, columns = features)
        return X_df

        
def crossValidate(classifier, X, Y):
    k_fold = KFold(n_splits=10, shuffle=True)
    if (classifier == 'LinearSVC'):
        clf = svm.LinearSVC(multi_class='ovr')
        cross_v_tdidf = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1)
#        cross_v_bow = cross_val_score(multinomialNB, BOW_df, Y, cv=k_fold, n_jobs=1)
        num = 0
        for i in cross_v_tdidf:
            num +=i
        print(num/10)
        
        num = 0
#        for i in cross_v_bow:
#            num +=i
#        print(num/10)
        return cross_v_tdidf
    
def produceSentiment(column_name, test_Y):
#    file_name = pd.read_json("src\dummy-review.json")
    i = 0
    j = 1
    k = 2
    pos_count = 0
    neu_count = 0
    neg_count = 0
    test_Y_list = []
    while (i < len(test_Y)):
        if (test_Y[i] and test_Y[i] == 'positive'):
            pos_count += 1
        elif (test_Y[i] and test_Y[i] == 'neutral'):
            neu_count += 1
        else:
            neg_count += 1
        if (test_Y[j] and test_Y[j] == 'positive'):
            pos_count += 1
        elif (test_Y[j] and test_Y[j] == 'neutral'):
            neu_count += 1
        else:
            neg_count += 1
        if (test_Y[k] and test_Y[k] == 'positive'):
            pos_count += 1
        elif (test_Y[k] and test_Y[k] == 'neutral'):
            neu_count += 1
        else:
            neg_count += 1
            
        cmax = max(pos_count, neu_count, neg_count)
        if cmax == pos_count:
            test_Y_list.append('positive')
        elif cmax == neu_count:
            test_Y_list.append('neutral')
        else:
            test_Y_list.append('negative')
        i += 3
        j += 3
        k += 3
        pos_count = 0
        neu_count = 0
        neg_count = 0
    data[column_name] = test_Y_list
# imports csv file of restaurant-review-labeled-data.csv
    
def labelEncoder(Y):
    encoder = preprocessing.LabelEncoder()
    Y = encoder.fit_transform(Y)
    return Y
    
    
def evaluate(y_pred, Y_test):
    print('Accuracy:\n', accuracy_score(Y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    
    
def get_majority_classifier(Y):
    classifiers = {}
    
    for values in Y.values:
        for value in values:
            if value in classifiers:
                classifiers[value] += 1
            else:
                classifiers[value] = 1
            
    max_label = max(classifiers, key=classifiers.get)
    max_value = max(classifiers.values())

    total_values = sum(classifiers.values())
    percentage = max_value / total_values
    
    return max_label, max_value, total_values, percentage


def get_class_distribution(Y):
    classifiers = {}
    
    for values in Y.values:
        for value in values:
            if value in classifiers:
                classifiers[value] += 1
            else:
                classifiers[value] = 1
    
    total_values = sum(classifiers.values())
    
    class_dist = {}
    for key, value in classifiers.items():
        class_dist[key] = value / total_values
    
    return class_dist


def k_fold_cross_validation(k, pipeline, X_train, Y_train):
    scores = cross_val_score(
             pipeline,
             X_train,
             Y_train,
             cv=k,
             scoring='f1_micro'
            )
    print(f'{k}-fold cross validation scores:')
    print(scores)
    return scores


data = pd.read_json('src/restaurant-review.json')
restaurant_labeled = pd.read_csv('restaurant-review-labeled-data.csv', dtype=str)

yelp_data = pd.read_json('yelp_training_set_review.json', lines=True)


# line 9 and 10 removes
# the row with nan, or the row with null values
# in the dataset
restaurant_labeled = restaurant_labeled[pd.notnull(restaurant_labeled['sentences__text'])]
restaurant_labeled = restaurant_labeled[pd.notnull(restaurant_labeled['sentences__Opinions__Opinion__polarity'])]
restaurant_labeled = restaurant_labeled[pd.notnull(restaurant_labeled['sentences__text'])]
restaurant_labeled = restaurant_labeled[pd.notnull(restaurant_labeled['sentences__Opinions__Opinion__polarity'])]
#restaurant_labeled = restaurant_labeled[restaurant_labeled['sentences__Opinions__Opinion__category'].str.contains('SERVICE')]
# we will ignore all the row that contains 'neutral' in the positive/negative column
# because we don't need additional label in our dataset
restaurant_labeled = restaurant_labeled[restaurant_labeled['sentences__Opinions__Opinion__polarity'] != 'neutral']

# X stores the text review column of the dataset
# Y stores the positive/negative label of the dataset
#X = restaurant_labeled.iloc[1:, 3:4]
#Y = restaurant_labeled.iloc[1:, 11:12]



#========================================================================
# THIS IS WHERE I INCLUDED KMEANS PIPELINE
#========================================================================

yelp_data1 = yelp_data[(yelp_data['stars'] == 1)]
yelp_data3 = yelp_data[(yelp_data['stars'] == 3)]
yelp_data5 = yelp_data[(yelp_data['stars'] == 5)]

num_samples = 100

X1 = yelp_data1.iloc[1:, 4:5]
X1 = X1.sample(n=num_samples)
X3 = yelp_data3.iloc[1:, 4:5]
X3 = X3.sample(n=num_samples)
X5 = yelp_data5.iloc[1:, 4:5]
X5 = X5.sample(n=num_samples)
X = pd.concat([X1, X3, X5])

Y1 = yelp_data1.iloc[1:, 3:4]
Y1 = Y1.sample(n=num_samples)
Y3 = yelp_data3.iloc[1:, 3:4]
Y3 = Y3.sample(n=num_samples)
Y5 = yelp_data5.iloc[1:, 3:4]
Y5 = Y5.sample(n=num_samples)
Y = pd.concat([Y1, Y3, Y5])

majority_classifier = get_majority_classifier(Y)
class_distribution = get_class_distribution(Y)

# Features for pipeline
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english',
                        min_df=0., max_df=1.)
avg_word_length = AverageWordLengthExtractor()
exclamation_count = ExclamationPointCountExtractor()
useful_value = UsefulValueExtractor()

# Classifiers for pipeline
svc = svm.LinearSVC(multi_class='crammer_singer')

# Pipeline that selects the text column and extracts tfidf score of terms with TfidfVectorizer
tfidf_pipeline = Pipeline([
        ('selector', ColumnSelector(key='text')),
        ('tfidf', tfidf)
])

# Pipeline that selects the text column and computes average word length for each text review
avg_word_length_pipeline = Pipeline([
        ('selector', ColumnSelector(key='text')),
        ('ave', avg_word_length)
])  
        
# Pipeline that selects the text column and extracts the number of exclamation points for each text review
exclamation_pipeline = Pipeline([
        ('selector', ColumnSelector(key='text')),
        ('exclamation', exclamation_count)
]) 
        
# Pipeline that selects the votes column and extracts the value at the 'useful' key
useful_value_pipeline = Pipeline([
        ('selector', ColumnSelector(key='votes')),
        ('useful-value', useful_value)
]) 
        
# FeatureUnion all individual feature pipelines
feature_pipeline = FeatureUnion([
        ('tfidf', tfidf_pipeline),
        ('avg-word-length', avg_word_length_pipeline),
        ('exclamation', exclamation_pipeline),
#        ('useful-value', useful_value_pipeline)
])

pipeline = Pipeline([
    ('features', feature_pipeline),
    ('classifier', svc)
])
        
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Instead of fitting the pipeline on the training data, 
# do cross-validation too so that you know if it's overfitting.
# This returns an array of values, each having the score for an individual run.
# k=10 implies 10-fold cross validation
scores = k_fold_cross_validation(10, pipeline, X_train, Y_train)

pipeline.fit(X_train, Y_train)

y_pred = pipeline.predict(X_test)

evaluate(y_pred, Y_test)

#test_data = pd.read_csv('src/restaurant-review.csv')
#test_X = test_data.iloc[:,-1]
#test_X = preprocessing(test_X)
#test_X = featuring(test_X, 'TFIDF')
#test_Y = list(clf.predict(test_X))
#produceSentiment('overall_senti', test_Y)

