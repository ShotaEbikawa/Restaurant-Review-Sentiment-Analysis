# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:22:57 2019

@author: anastasiosgrigoriou
@author: shotaebikawa
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from extractors import ColumnSelector, ClfSwitcher
from matplotlib.pyplot import figure

    
restaurant_category = {0: 'Service', 1: 'Food Quality'}
        
def evaluate(y_pred, X_test, Y_test):
    print('Accuracy:\n', accuracy_score(Y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
        # Iterate through X_test total_iters times, printing the review text and predicted output label
#    y_index = 0
#    print(f'Showing {10} results of review text and predicted output label...\n')
#    for index, row in X_test.iterrows():
#        print(f'Review text: \n{row["text"]} \n\nPredicted: \n{y_pred[y_index]}\n')
#        y_index += 1
#        if y_index >= 10:
#            break
    
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

def convert_labels(violation_reviews):
    pd.options.mode.chained_assignment = None 
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
   

def gridsearch_clf(grid, grid_dict, X_train, Y_train, ):
    # These two lists will be used for plotting the graph later on
    classifier_list = ['SVC', 'MNB', 'RFC', 'LR']
    accuracy_list = []

    # Stores the best accuracy
    best_acc = 0.0
    # Stores index of grids (which is a list) containing best performing classifier
    best_clf = 0
    # Stores the best performing GridSearch 
    best_gs = ''
    
    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])
        gs.fit(X_train, Y_train)
        print('Best params: %s' % gs.best_params_)
        print('Best training accuracy: %.3f' % gs.best_score_)
        y_pred = gs.predict(X_test)
        evaluate(y_pred, X_test, Y_test)
        accuracy_list.append(accuracy_score(Y_test, y_pred))
        print('Test set accuracy score for best params: %.3f ' % accuracy_score(Y_test, y_pred))
        if accuracy_score(Y_test, y_pred) > best_acc:
            best_acc = accuracy_score(Y_test, y_pred)
            best_gs = gs
            best_clf = idx
    print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])
    figure(num=None, figsize=(12, 6))
    plt.bar(classifier_list, accuracy_list, width= 0.3)
    plt.ylim(0.9,0.96)
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')
    return best_gs
    
 
def predict_zomato_reviews(best_gs):
    data = pd.read_json('../datasets/restaurant-review-zomato.json')

    # Convert json formatted zomato data into dataframe with text and useful columns
    review_rows = []
    useful_rows = []
    for index, row in data.iterrows():
        for review in row['reviews']:
            if review and review['text']:
                review_rows.append(review['text'])
                useful_rows.append(0)

    review_rows = pd.DataFrame(review_rows)
    review_rows.columns = ['text']

    y_results = best_gs.predict(review_rows)

    # Print a few results for demonstration purposes
    y_index = 0
    print(f'Showing 10 results of previously unseen reviews and predicted output label...\n')
    for index, row in review_rows.iterrows():
        print(f'Review text: \n{row["text"]} \n\nPredicted: \n{restaurant_category[y_results[y_index]]}\n')
        y_index += 1
        if y_index >= 10:
            break  




violation_reviews = pd.read_csv('../datasets/restaurant-violations.csv')
violation_reviews = violation_reviews[violation_reviews['Violation_Number'] != 'None']
violation_reviews = violation_reviews[violation_reviews['Violation_Number'] != '23']
convert_labels(violation_reviews)
violation_Service = violation_reviews[(violation_reviews['Violation_Number'] == 0)]
violation_Food = violation_reviews[(violation_reviews['Violation_Number'] == 1)]

numSample = 10000
serviceX = violation_Service.iloc[1:, 2:3]
serviceX = serviceX.sample(n= numSample)

foodX = violation_Food.iloc[1:, 2:3]
foodX = foodX.sample(n= numSample)

serviceY = violation_Service.iloc[1:, 1:2]
serviceY = serviceY.sample(n=numSample)

foodY = violation_Food.iloc[1:, 1:2]
foodY = foodY.sample(n=numSample)

X = pd.concat([serviceX, foodX])
Y = pd.concat([serviceY, foodY])

get_majority_classifier(Y)
get_class_distribution(Y)

Y = Y['Violation_Number'].astype(int)

tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words='english',
                      min_df=0., max_df=1.)


#------------------------------------------------------------------------------------------------
# CLASSIFIERS (Multinomial Naive Bayes, Gaussian Naive Bayes, Linear Support Vector Machines,
# Random Forest Classifier), Logistic Regressoin
#------------------------------------------------------------------------------------------------
multinomial = MultinomialNB()
gaussian = GaussianNB()
svc = svm.LinearSVC()
rf = RandomForestClassifier()
lr = LogisticRegression()


#------------------------------------------------------------------------------------------------
# SPLIT THE DATASET INTO TEST AND TRAIN SETS
#------------------------------------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)



#------------------------------------------------------------------------------------------------
# IMPLEMENTING GRID SEARCH CV FOR HYPERTUNING THE CLASSIFIERS
#------------------------------------------------------------------------------------------------
# pipeline that contains preprocessors, feature extractors (tfidf), and ClfSwitcher()
pipeline_one = Pipeline([
        ('selector', ColumnSelector(key='text')),
        ('tfidf', tfidf),
        ('clf', ClfSwitcher()),
])


"""
list containing all the given parameters for seleted classifiers and feature extractors
paremter name starting with clf__estimator__ is for the given classifier's parameter
paramter name starting with tfidf__ is for the tfidf's parameter 
I commented out some of the parameters because it did not really affect the accuracy
and some of them took way too long to run.
The most useful parameter, in my opinion, was tfidf__max_df
"""

# list containing parameters for Linear Support Vector Machine and TFIDF    
grid_params_svm = [{'clf__estimator': [svc], # SVM if hinge loss / logreg if log loss
                   'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
                   'tfidf__ngram_range': [(1,1), (1,2), (2,3)]
                   }]

# list containing parameters for Multinomial Naive Bayes and TFIDF
grid_params_mnb = [{'clf__estimator': [multinomial],
                    'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
                    'tfidf__ngram_range': [(1,1), (1,2), (2,3)]
                    }]
  
    
# list containing parameters for Random Forest Classifier and TFIDF
grid_params_rf = [{'clf__estimator': [rf],
                   'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
                   'tfidf__ngram_range': [(1,1), (1,2), (2,3)]
                   }]

#list containing parameters for Logistic Regression and TFIDF
grid_params_lr = [{'clf__estimator': [lr],
                   'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),   
                   'tfidf__ngram_range': [(1,1), (1,2), (2,3)]
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

#GridSearchCV for Logistic Regression
gs_lr = GridSearchCV(pipeline_one,
                     grid_params_lr,
                     scoring='accuracy',
                     cv=10)

# list containing all of the GridSearchCV
grids = [gs_svm, gs_mnb, gs_rf, gs_lr]

# Dictionary containing all of the name of the classifiers
grid_dict = {0: 'Support Vector Machine (SVC)', 1: 'Multinomial Naive Bayes', 2:'Random Forest Classifier', 3: 'Logistic Regression'}


# Iterate through grids
best_gs = gridsearch_clf(grids, grid_dict, X_train, Y_train,)

#plot the graph 


# evaluate(y_pred, Y_test)
# =============================================================================
# NOW I'M GOING TO PREDICT THE YELP_DATA WITH THE PIPELINE
# =============================================================================

#yelp_data = pd.read_json('../datasets/yelp_training_set_review.json', lines=True)
#X2 = yelp_data.iloc[1:, 4:5]
#category = best_gs.predict(X2)
#category = pd.DataFrame(data=category, columns=['category'])
#new_yelp_data = pd.concat([yelp_data, category], axis = 1, join = 'inner')

# =============================================================================
# PREDICTING ZOMATO TEXT REVIEWS
# =============================================================================
predict_zomato_reviews(best_gs)
