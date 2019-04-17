# -*- coding: utf-8
import pandas as pd
#import matplotlib as lib
#import numpy as np
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
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
#from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from gensim.models import word2vec
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix




def preprocessing(X):
    real_token = []
    stop_words = nltk.corpus.stopwords.words('english')
    punctuation =  [',','-','+','.','/','\\','\'','"','?','!','$','(',')','...',
                    '..', '--', '---',':','~','=','`','{','}','\n']
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
    

def featuring(X, feature_name):
    if (feature_name == 'BOW'):
        BOW_Vector = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda doc: doc,
                                     lowercase=False, min_df = 0., max_df = 1., max_features = 5581)
        BOW_Matrix = BOW_Vector.fit_transform(X)
        features = BOW_Vector.get_feature_names()
        BOW_Matrix = BOW_Matrix.toarray()
        BOW_df = pd.DataFrame(BOW_Matrix, columns = features)
        return BOW_df
    if (feature_name == 'TFIDF'): 
        tfidf = TfidfVectorizer(ngram_range=(1,2), lowercase = False, 
                                analyzer = 'word',tokenizer= lambda doc: doc, 
                                min_df = 0, max_df = 0.8,max_features = 475)
        X_tfidf = tfidf.fit_transform(X)
        features = tfidf.get_feature_names()
        X_tfidf = X_tfidf.toarray()
        X_df = pd.DataFrame(X_tfidf, columns = features)
        return X_df
    
    
def printAccuracy(pred, test):
    print(accuracy_score(y_pred, Y_test))


def classifier(classifier_name, X_train, Y_train):
    if (classifier_name == 'MultinomialNB'):
        multinomialNB = MultinomialNB(alpha = 0.9,fit_prior = True, class_prior=None)
        multinomialNB.fit(X_train, Y_train)
        y_pred = multinomialNB.predict(X_test)
        return multinomialNB
    
    elif (classifier_name == 'RandomForestClassifier'):
        randomForest = RandomForestClassifier(n_estimators = 1)
        randomForest.fit(X_train, Y_train)
        y_pred = randomForest.predict(X_test)
        return randomForest
    
    elif (classifier_name == 'LogicticRegression'):
        Logistic = LogisticRegression()
        Logistic.fit(X_train, Y_train)
        return Logistic
    
    elif (classifier_name == 'LinearSVC'):
        clf = svm.LinearSVC(multi_class='ovr')
        clf.fit(X_train, Y_train) 
        return clf
        
def crossValidate(classifier, X, Y):
    k_fold = KFold(n_splits=10, shuffle= True)
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


def evaluateCM(y_pred, Y_test):
    cm = confusion_matrix(y_pred, Y_test)
    t_pos_neg = cm[0][0]
    t_neg_neg = cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2]
    f_pos_neg = cm[1][0]+cm[2][0]
    f_neg_neg = cm[0][1]+cm[0][2]
    
    
    t_pos_neu = cm[1][1]
    t_neg_neu = cm[0][0]+cm[0][2]+cm[2][0]+cm[2][2]
    f_pos_neu = cm[0][1]+cm[2][1]
    f_neg_neu = cm[1][0]+cm[1][2]
    
    t_pos_pos = cm[2][2]
    t_neg_pos = cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
    f_pos_pos = cm[0][2]+cm[1][2]
    f_neg_pos = cm[2][0]+cm[2][1]
    
    accuracy = (t_pos_neg + t_pos_neu + t_pos_pos) / (t_pos_neg + t_pos_neu + t_pos_pos + f_neg_neg + f_neg_neu + f_neg_pos)
    precision_pos = t_pos_pos / (t_pos_pos + f_pos_pos)
    precision_neg = t_pos_neg / (t_pos_neg + f_pos_neg)
    precision_neu = t_pos_neu / (t_pos_neu + f_pos_neu)
    
    recall_pos = t_pos_pos / (t_pos_pos + f_neg_pos)
    recall_neg = t_pos_neg / (t_pos_neg + f_neg_neg)
    recall_neu = t_pos_neu / (t_pos_neu + f_neg_neu)
    
    avg_precision = (precision_pos + precision_neg + precision_neu) / 3
    avg_recall = (recall_pos + recall_neg + recall_neu) / 3
    
    f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)
    
    print("The accuracy is: ", accuracy*100, "%")
    print("The precision is: ", avg_precision*100, "%")
    print("The recall is: ", avg_recall*100, "%")
    print("The f1 score is: ", f1*100, "%")
    
def class_distribution(Y):
    class_dist = {}
    
    for values in Y.values:
        for value in values:
            if value in class_dist:
                class_dist[value] += 1
            else:
                class_dist[value] = 1
            
    max_label = max(class_dist, key=class_dist.get)
    max_value = max(class_dist.values())

    total_values = sum(class_dist.values())
    percentage = max_value / total_values
    
    return (max_label, max_value, total_values, percentage)


data = pd.read_json('src/restaurant-review.json')
restaurant_labeled = pd.read_csv('restaurant-review-labeled-data.csv', dtype=str)
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
#restaurant_labeled = restaurant_labeled[restaurant_labeled['sentences__Opinions__Opinion__polarity'] != 'neutral']

# X stores the text review column of the dataset
# Y stores the positive/negative label of the dataset
X = restaurant_labeled.iloc[1:, 3:4]
Y = restaurant_labeled.iloc[1:, 11:12]

class_dist = class_distribution(Y)

X = preprocessing(X)
X = featuring(X, 'TFIDF')
crossValidate = crossValidate('LinearSVC', X, Y)
#X_train, X_test, Y_train, Y_test = train_test_split(BOW_df, Y, test_size = 0.25, random_state = 0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
clf = classifier('LinearSVC', X_train, Y_train)
y_pred = list(clf.predict(X_test))
printAccuracy(y_pred,Y_test)
evaluateCM(y_pred, Y_test)

#test_data = pd.read_csv('src/restaurant-review.csv')
#test_X = test_data.iloc[:,-1]
#test_X = preprocessing(test_X)
#test_X = featuring(test_X, 'TFIDF')
#test_Y = list(clf.predict(test_X))
#produceSentiment('overall_senti', test_Y)

