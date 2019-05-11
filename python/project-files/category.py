# -*- coding: utf-8 -*-
# -*- coding: utf-8
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
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
from Extractors import AverageWordLengthExtractor, ColumnSelector, ClfSwitcher
from sklearn.cluster import KMeans as Kmeans
from sklearn.cluster import SpectralClustering as Spectral
from sklearn.cluster import AgglomerativeClustering
#from sklearn_extensions.fuzzy_kmeans import KMedians, FuzzyKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
#import json
from sklearn.model_selection import GridSearchCV
from sklearn import svm
    
    
def printAccuracy(pred, test):
    print(accuracy_score(y_pred, Y_test))

        
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

tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words='english',
                      min_df=0., max_df=1.)


#------------------------------------------------------------------------------------------------
# CLASSIFIERS (Multinomial Naive Bayes, Linear Support Vector Machines, Random Forest Classifier)
#------------------------------------------------------------------------------------------------
multinomial = MultinomialNB()
svc = svm.LinearSVC(multi_class='ovr')
rf = RandomForestClassifier()


#------------------------------------------------------------------------------------------------
# SPLIT THE DATASET INTO TEST AND TRAIN SETS
#------------------------------------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)



#------------------------------------------------------------------------------------------------
# IMPLEMENTING GRID SEARCH CV FOR HYPERTUNING THE CLASSIFIERS
#------------------------------------------------------------------------------------------------
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
param_range_fl = [1.0, 0.5, 0.1]

# pipeline that contains preprocessors, feature extractors (tfidf), and ClfSwitcher()
pipeline_one = Pipeline([
        ('selector', ColumnSelector(key='text')),
        ('tfidf', tfidf),
        ('clf', ClfSwitcher())
])



# list containing all the given parameters for seleted classifiers and feature extractors
# paremter name starting with clf__estimator__ is for the given classifier's parameter
# paramter name starting with tfidf__ is for the tfidf's parameter 


# list containing parameters for Linear Support Vector Machine and TFIDF    
grid_params_svm = [{'clf__estimator': [svc], # SVM if hinge loss / logreg if log loss
                   'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
                   'tfidf__stop_words': ['english', None],
                   'clf__estimator__max_iter': [1, 10, 100]
                   }]

# list containing parameters for Multinomial Naive Bayes and TFIDF
grid_params_mnb = [{'clf__estimator': [multinomial],
                    'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
                    'tfidf__stop_words': [None],
                    'clf__estimator__alpha': (1e-2, 1e-3, 1e-1),
                    }]
    
# list containing parameters for Random Forest Classifier and TFIDF
grid_params_rf = [{'clf__estimator': [rf],
                   'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
                   'tfidf__stop_words': [None],
                   'clf__estimator__min_samples_leaf': [1, 3, 10],
                   'clf__estimator__n_estimators': [10,11,12]
                   }]


# GridSearchCV for Linear Support Vector Machine
gs_svm = GridSearchCV(pipeline_one,
			grid_params_svm,
			scoring='accuracy',
			cv=10,
			)

#GridSearchCV for Multinomial Naive Bayes
gs_mnb = GridSearchCV(pipeline_one,
                      grid_params_mnb,
                      scoring='accuracy',
                      cv=10,
                      )

#GridSearchCV for Random Forest Classifier
gs_rf = GridSearchCV(pipeline_one,
                     grid_params_rf,
                     scoring='accuracy',
                     cv=10)

# list containing all of the GridSearchCV
grids = [gs_svm, gs_mnb, gs_rf]

# Dictionary containing all of the name of the classifiers
grid_dict = {0: 'Support Vector Machine (SVC)', 1: 'Multinomial Naive Bayes', 2:'Random Forest Classifier'}

print('Performing model optimizations...')
# Stores the best accuracy
best_acc = 0.0
# Stores index of grids (which is a list) containing best performing classifier
best_clf = 0
# Stores the best performing GridSearch 
best_gs = ''

# Iterate through grids
for idx, gs in enumerate(grids):
	print('\nEstimator: %s' % grid_dict[idx])
	gs.fit(X_train, Y_train)
	print('Best params: %s' % gs.best_params_)
	print('Best training accuracy: %.3f' % gs.best_score_)
	y_pred = gs.predict(X_test)
	print('Test set accuracy score for best params: %.3f ' % accuracy_score(Y_test, y_pred))
	if accuracy_score(Y_test, y_pred) > best_acc:
		best_acc = accuracy_score(Y_test, y_pred)
		best_gs = gs
		best_clf = idx
print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

# evaluate(y_pred, Y_test)
# =============================================================================
# NOW I'M GOING TO PREDICT THE YELP_DATA WITH THE PIPELINE
# =============================================================================

yelp_data = pd.read_json('yelp_training_set_review.json', lines=True)
X2 = yelp_data.iloc[1:, 4:5]
category = gs.predict(X2)
category = pd.DataFrame(data=category, columns=['category'])
new_yelp_data = pd.concat([yelp_data, category], axis = 1, join = 'inner')

