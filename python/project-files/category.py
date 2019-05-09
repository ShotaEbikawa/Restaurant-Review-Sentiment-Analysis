# -*- coding: utf-8 -*-
# -*- coding: utf-8
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing
import nltk
import scipy.sparse
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
from Extractors import AverageWordLengthExtractor, ColumnSelector
from sklearn.cluster import KMeans as Kmeans
from sklearn.cluster import SpectralClustering as Spectral
from sklearn.cluster import AgglomerativeClustering
#from sklearn_extensions.fuzzy_kmeans import KMedians, FuzzyKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD

def pre_process(s):
    pass

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
    
    elif (classifier_name == 'LogisticRegression'):
        Logistic = LogisticRegression()
        Logistic.fit(X_train, Y_train)
        return Logistic
    
    elif (classifier_name == 'LinearSVC'):
        clf = svm.LinearSVC(multi_class='crammer_singer')
        clf.fit(X_train, Y_train) 
        return clf
        
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
    
    
def evaluate2CM(y_pred, Y_test):
    cm = confusion_matrix(y_pred, Y_test)
    print ('Confusion Matrix:\n', cm)
    
    tn, fp, fn, tp = cm.ravel()
    total = tn + tp + fn + fp
    
    accuracy = (tp + tn) / (total)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    
    print('Accuracy is ', round(accuracy * 100, 2), '%')
    print('Precision is ', round(precision, 2))
    print('Recall is ', round(recall, 2))
    print('f1 score is ', round(f1, 2))
    
    
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


violation_code = pd.read_csv('Violation Codes Key.csv')
vio_num = violation_code.iloc[:, 0:1]
violation_reviews = pd.read_csv('Restaurant Violations.csv')
violation_reviews = violation_reviews[violation_reviews['Violation_Number'] != 'None']
violation_reviews = violation_reviews[violation_reviews['Violation_Number'] != '23']

for i in range(1, 67):
    if i <= 9:
        #0 refers to service
        #1 refers to food quality
        violation_reviews.Violation_Number[violation_reviews.Violation_Number == str(i)] = 0
    elif (10 <= i <= 22):
        violation_reviews.Violation_Number[violation_reviews.Violation_Number == str(i)] = 1
    elif (26 <= i <= 27):
        violation_reviews.Violation_Number[violation_reviews.Violation_Number == str(i)] = 0
    elif (i == 28):
        violation_reviews.Violation_Number[violation_reviews.Violation_Number == str(i)] = 1
    elif (i == 29):
        violation_reviews.Violation_Number[violation_reviews.Violation_Number == str(i)] = 0
    elif (30 <= i <= 40):
        violation_reviews.Violation_Number[violation_reviews.Violation_Number == str(i)] = 1
    elif (41 <= i <= 66):
        violation_reviews.Violation_Number[violation_reviews.Violation_Number == str(i)] = 0
    else:
        pass

violation_Service = violation_reviews[(violation_reviews['Violation_Number'] == 0)]
violation_Food = violation_reviews[(violation_reviews['Violation_Number'] == 1)]

serviceX = violation_Service.iloc[1:, 2:3]
serviceX = serviceX.sample(n= 10000)
foodX = violation_Food.iloc[1:, 2:3]
foodX = foodX.sample(n= 10000)
serviceY = violation_Service.iloc[1:, 1:2]
serviceY = serviceY.sample(n=10000)
foodY = violation_Food.iloc[1:, 1:2]
foodY = foodY.sample(n= 10000)
X = pd.concat([serviceX, foodX])
Y = pd.concat([serviceY, foodY])
Y = Y['Violation_Number'].astype(int)
#X = preprocessing(X)
#wordX = Word2Vec(X)
tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words='english',
                      min_df=0., max_df=1.)
#ave = AverageWordLengthExtractor()
#w2v = Word2Vec(min_count = 1)
kmeans = Kmeans(n_clusters=2, )
spectral = Spectral(n_clusters=2, affinity= 'precomputed', n_init=100)
# Classifier for pipeline
svc = svm.LinearSVC(multi_class='ovr')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
#worddd = wordX.train(X_train, total_words = 750, epochs = 3)
pipeline_one = Pipeline([
        ('selector', ColumnSelector(key='text')),
        ('tfidf', tfidf)
])
            
pipeline = Pipeline([
    ('features', pipeline_one),
    ('classifier', svc)
])
       
pipeline.fit(X_train, Y_train)
y_pred = pipeline.predict(X_test)
evaluate(y_pred, Y_test)


# =============================================================================
# NOW I'M GOING TO PREDICT THE YELP_DATA WITH THE PIPELINE
# =============================================================================

yelp_data = pd.read_json('yelp_training_set_review.json', lines=True)
X2 = yelp_data.iloc[1:, 4:5]
category = pipeline.predict(X2)
category = pd.DataFrame(data=category, columns=['category'])
new_yelp_data = pd.concat([yelp_data, category], axis = 1, join = 'inner')


