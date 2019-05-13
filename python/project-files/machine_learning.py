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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from Extractors import ColumnSelector, PositiveWordCountExtractor, NegativeWordCountExtractor, UppercaseWordCountExtractor, AverageWordLengthExtractor, ExclamationPointCountExtractor, UsefulValueExtractor, ClfSwitcher
from matplotlib.pyplot import figure

# Global variables
lemmatizer = WordNetLemmatizer()

decoder = {1: 'negative', 3: 'neutral', 5: 'positive'}

num_samples = 10
total_iters = 10

classifier_list = ['SVC', 'MNB', 'RFC', 'LR']
accuracy_list = []

pipelines = []
best_accuracy_score = 0.0
best_pipeline = None


def add_vote_columns(yelp_df):
    funny_data = []
    useful_data = []
    cool_data = []

    # Add values for each key to each data array
    for vote in yelp_df['votes']:
        funny_data.append(vote['funny'])
        useful_data.append(vote['useful'])
        cool_data.append(vote['cool'])

    # Construct a new DataFrame with the array data, with corresponding column name
    funny_df = pd.DataFrame(data=funny_data, columns=['funny'])
    useful_df = pd.DataFrame(data=useful_data, columns=['useful'])
    cool_df = pd.DataFrame(data=cool_data, columns=['cool'])

    # Remove 'votes' column from original yelp DataFrame
    yelp_df = yelp_df.drop(columns=['votes'])

    # Concat the new df's with the original yelp_df
    yelp_df = pd.concat([yelp_df, funny_df, useful_df, cool_df], axis=1, join='inner')
    return yelp_df


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


def setup_feature_pipeline():
    # Open pos/neg words text file and store the words into a list
    positive_words = [pos_word.rstrip('\n') for pos_word in open('src/positive-words.txt')]
    negative_words = [neg_word.rstrip('\n') for neg_word in open('src/negative-words.txt')]

    # Features for pipeline
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english',
                            min_df=0., max_df=1.)

    pos_word_count = PositiveWordCountExtractor(positive_words)
    neg_word_count = NegativeWordCountExtractor(negative_words)

    uppercase_word_count = UppercaseWordCountExtractor()
    avg_word_length = AverageWordLengthExtractor()
    exclamation_count = ExclamationPointCountExtractor()

    useful_value = UsefulValueExtractor()

    # Pipeline that selects the text column and extracts tfidf score of terms with TfidfVectorizer
    tfidf_pipeline = Pipeline([
        ('selector', ColumnSelector(key='text')),
        ('tfidf', tfidf)
    ])

    # Pipeline that selects the text column and computes the number of 'positive' words in a review text
    pos_word_count_pipeline = Pipeline([
        ('selector', ColumnSelector(key='text')),
        ('pos-word', pos_word_count)
    ])

    # Pipeline that selects the text column and computes the number of 'negative' words in a review text
    neg_word_count_pipeline = Pipeline([
        ('selector', ColumnSelector(key='text')),
        ('neg-word', neg_word_count)
    ])

    # Pipeline that selects the text column and computes the number of uppercased words for each text review
    uppercase_word_count_pipeline = Pipeline([
        ('selector', ColumnSelector(key='text')),
        ('uppercase-word', uppercase_word_count)
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
        ('selector', ColumnSelector(key='useful')),
        ('useful-value', useful_value)
    ])

    # FeatureUnion all individual feature pipelines
    feature_pipeline = FeatureUnion([
        ('tfidf', tfidf_pipeline),
        ('pos-word-count', pos_word_count_pipeline),
        ('neg-word-count', neg_word_count_pipeline),
        ('uppercase-word-count', uppercase_word_count_pipeline),
        ('avg-word-length', avg_word_length_pipeline),
        ('exclamation', exclamation_pipeline),
        ('useful-value', useful_value_pipeline)
    ])

    return feature_pipeline


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


def get_grid_search_params():
    multinomial = MultinomialNB()
    gaussian = GaussianNB()
    svc = svm.LinearSVC(multi_class='ovr')
    rf = RandomForestClassifier()
    lr = LogisticRegression()

    # list containing parameters for Linear Support Vector Machine and TFIDF
    parameters = [
        {
            'clf__estimator': [svc], # SVM if hinge loss / logreg if log loss
            'tfidf__max_df': (0.25, 0.5, 0.75, 1.0)
        },
        {
            'clf__estimator': [multinomial],
            'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
            'clf__estimator__alpha': (1e-2, 1e-3, 1e-1)
        },
        {
            'clf__estimator': [rf],
            'tfidf__max_df': (0.25, 0.5, 0.75, 1.0)
        },
        {
            'clf__estimator': [lr],
            'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
            'clf__estimator__solver': ['liblinear', 'sag', 'saga'],
            'clf__estimator__C': [100, 1000]
        }
     ]
    return parameters


def make_prediction(X, Y, pipeline, classifier_name):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    # Instead of fitting the pipeline on the training data, 
    # do cross-validation too so that you know if it's overfitting.
    # This returns an array of values, each having the score for an individual run.
    # k=10 implies 10-fold cross validation
    #scores = k_fold_cross_validation(10, pipeline, X_train, Y_train)

    # pipeline.fit(X_train, Y_train.values.ravel())
    #
    # y_pred = pipeline.predict(X_test)
    #
    # evaluate(y_pred, Y_test, X_test, pipeline, classifier_name)

    parameters = get_grid_search_params()

    gscv = GridSearchCV(pipeline, parameters, cv=5, n_jobs=12, verbose=3)
    gscv.fit(X_train, Y_train.values.ravel())

    y_pred = gscv.predict(X_test)

    evaluate(y_pred, Y_test, X_test, pipeline, classifier_name)


def evaluate(y_pred, Y_test, X_test, pipeline, classifier_name):
    print(f'\nEvaluation results for classifier: {classifier_name}\n')
    acc_score = accuracy_score(Y_test, y_pred)
    accuracy_list.append(acc_score)
    pipelines.append((pipeline, acc_score))
    print('Accuracy:\n', acc_score)
    print('Confusion Matrix:\n', confusion_matrix(Y_test, y_pred))
    print('Classification report:\n', classification_report(Y_test, y_pred))

    # Iterate through X_test total_iters times, printing the review text and predicted output label
    y_index = 0
    print(f'Showing {total_iters} results of review text and predicted output label...\n')
    for index, row in X_test.iterrows():
        print(f'Review text: \n{row["text"]} \n\nPredicted: \n{decoder[y_pred[y_index]]}\n')
        y_index += 1
        if y_index >= total_iters:
            break
        
        
def plot_accuracy_graph():
    figure(num=None, figsize=(12, 6))
    plt.scatter(classifier_list, accuracy_list)
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')


def predict_zomato_reviews():
    data = pd.read_json('src/restaurant-review-zomato.json')

    review_rows = []
    useful_rows = []
    for index, row in data.iterrows():
        for review in row['reviews']:
            if review and review['text']:
                review_rows.append(review['text'])
                useful_rows.append(0)

    reviews_df = pd.DataFrame(data=review_rows, columns=['text'])
    useful_df = pd.DataFrame(data=useful_rows, columns=['useful'])

    reviews_df = pd.concat([reviews_df, useful_df], axis=1, join='inner')

    max_score = 0
    for pipeline, acc_score in pipelines:
        if acc_score > max_score:
            best_pipeline = pipeline

    y_results = best_pipeline.predict(reviews_df)

    y_index = 0
    print(f'Showing {total_iters} results of previously unseen reviews and predicted output label...\n')
    for index, row in reviews_df.iterrows():
        print(f'Review text: \n{row["text"]} \n\nPredicted: \n{decoder[y_results[y_index]]}\n')
        y_index += 1
        if y_index >= total_iters:
            break


# Read in json formatted yelp training set review data into a DataFrame
yelp_data = pd.read_json('yelp_training_set_review.json', lines=True)

# Add 'funny', 'useful', and 'cool' as separate columns in yelp_data DataFrame
yelp_data = add_vote_columns(yelp_data)

# Extract all 1, 3, 5 star rating reviews for 'negative', 'neutral', and 'positive' labels
yelp_data1 = yelp_data[(yelp_data['stars'] == 1)]
yelp_data3 = yelp_data[(yelp_data['stars'] == 3)]
yelp_data5 = yelp_data[(yelp_data['stars'] == 5)]

# Take all 1 star, 3 star, and 5 star rows for both text and useful columns to use as X text/train data
X = pd.concat([
    yelp_data1.loc[:, ['text', 'useful']].sample(n=num_samples),
    yelp_data3.loc[:, ['text', 'useful']].sample(n=num_samples),
    yelp_data5.loc[:, ['text', 'useful']].sample(n=num_samples)
])

# Take all 1 star, 3 star, and 5 star rows for star rating column for our Y data
Y = pd.concat([
    yelp_data1.loc[:, ['stars']].sample(n=num_samples),
    yelp_data3.loc[:, ['stars']].sample(n=num_samples),
    yelp_data5.loc[:, ['stars']].sample(n=num_samples)
])

# Get our majority classifier and class distribution and print their values
majority_classifier = get_majority_classifier(Y)
class_distribution = get_class_distribution(Y)

# Setup feature pipeline that includes all of our custom feature extractors
feature_pipeline = setup_feature_pipeline()

# A dictionary of classifiers with their name as key 
# We will use the dict to make predictions for our data and record the results for each classifier
classifier_dict = {
    'Support Vector Machine (SVC)': svm.LinearSVC(multi_class='crammer_singer'),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression()
}

# Iterate through our classifier dict, setup a pipeline with our original features, then make predictions
for name, classifier in classifier_dict.items():
    pipeline = Pipeline([
        ('features', feature_pipeline),
        ('clf', ClfSwitcher())
    ])

    make_prediction(X, Y, pipeline, name)
    
# Plot graph of classifiers and accuracy on x,y axis
plot_accuracy_graph()

# Predict zomato restaurant reviews using our best (most accurate) feature/classifier pipeline
predict_zomato_reviews()
