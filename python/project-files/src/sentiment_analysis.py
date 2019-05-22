# -*- coding: utf-8
"""
Created on Sat Apr 6 18:32:12 2019

@author: anastasiosgrigoriou
@author: shotaebikawa
"""
import copy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from extractors import ColumnSelector, PositiveWordCountExtractor, NegativeWordCountExtractor, UppercaseWordCountExtractor, AverageWordLengthExtractor, ExclamationPointCountExtractor, UsefulValueExtractor
import matplotlib
from matplotlib import pyplot as plt

# Global variables
decoder = {1: 'negative', 3: 'neutral', 5: 'positive'}

K = 10
num_samples = 100
total_iters = 10

classifier_list = ['SVC', 'MNB', 'RFC', 'LR']
accuracy_list = []

pipelines = []
best_pipeline = None

max_svm_iters = 1000000000
svc = svm.LinearSVC(multi_class='crammer_singer', max_iter=max_svm_iters)


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
    positive_words = [pos_word.rstrip('\n') for pos_word in open('../datasets/positive-words.txt')]
    negative_words = [neg_word.rstrip('\n') for neg_word in open('../datasets/negative-words.txt')]

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
        ('pos_word_count', pos_word_count_pipeline),
        ('neg_word_count', neg_word_count_pipeline),
        ('uppercase_word_count', uppercase_word_count_pipeline),
        ('avg_word_length', avg_word_length_pipeline),
        ('exclamation', exclamation_pipeline),
        ('useful_value', useful_value_pipeline)
    ])

    return feature_pipeline


def k_fold_cross_validation(k, pipeline, X_train, Y_train):

    # Instead of fitting the pipeline on the training data, 
    # do cross-validation too so that you know if it's overfitting.
    # This returns an array of values, each having the score for an individual run.
    # k=10 implies 10-fold cross validation
    scores = cross_val_score(
        pipeline,
        X_train,
        Y_train,
        cv=k,
        scoring='accuracy'
    )
    return scores


def make_prediction(X, Y, pipeline, classifier_name):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    pipeline.fit(X_train, Y_train.values.ravel())
    
    scores = k_fold_cross_validation(K, pipeline, X_train, Y_train.values.ravel())
    print(f'10-fold cross validation scores for {classifier_name}: {scores}')

    y_pred = pipeline.predict(X_test)

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
    plt.figure(num=None, figsize=(12, 6))
    plt.bar(classifier_list, accuracy_list, width=0.3)
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')


def predict_zomato_reviews():
    global best_pipeline
    data = pd.read_json('../datasets/restaurant-review-zomato.json')

    # Convert json formatted zomato data into dataframe with text and useful columns
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

    # Use the best performing classifier/pipeline to predict the zomato restaurant review sentiment label
    best_pipeline = max(pipelines, key=lambda item:item[1])[0]

    y_results = best_pipeline.predict(reviews_df)

    # Print a few results for demonstration purposes
    y_index = 0
    print(f'Showing {total_iters} results of previously unseen reviews and predicted output label...\n')
    for index, row in reviews_df.iterrows():
        print(f'Review text: \n{row["text"]} \n\nPredicted: \n{decoder[y_results[y_index]]}\n')
        y_index += 1
        if y_index >= total_iters:
            break


def perform_feature_ablation(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    # Perform 10-fold cross validation and compute the average accuracy score
    fp_total = copy.deepcopy(feature_pipeline)
    fp_total_pipeline = Pipeline([
        ('features', fp_total),
        ('classifier', svc)
    ])
    fp_total_scores = k_fold_cross_validation(K, fp_total_pipeline, X_train, Y_train.values.ravel())
    fp_total_score = sum(fp_total_scores) / K

    # Drop tfidf feature from original feature pipeline,
    # then perform 10-fold cross validation to get accuracy test run scores and compute average accuracy score,
    # then compare with total average accuracy score when tfidf feature was included (difference)
    fp_no_tfidf = copy.deepcopy(feature_pipeline)
    fp_no_tfidf.set_params(tfidf='drop')
    fp_no_tfidf_pipeline = Pipeline([
        ('features', fp_no_tfidf),
        ('classifier', svc)
    ])
    no_tfidf_scores = k_fold_cross_validation(K, fp_no_tfidf_pipeline, X_train, Y_train.values.ravel())
    no_tfidf_score = sum(no_tfidf_scores) / K
    tfidf_delta_accuracy = fp_total_score - no_tfidf_score
    print(f'Accuracy delta when adding tfidf feature: {tfidf_delta_accuracy}')

    # Repeat the same procedure, now taking out the positive word count feature
    fp_no_pos_word = copy.deepcopy(feature_pipeline)
    fp_no_pos_word.set_params(pos_word_count='drop')
    fp_no_pos_word_pipeline = Pipeline([
        ('features', fp_no_pos_word),
        ('classifier', svc)
    ])
    no_pos_word_scores = k_fold_cross_validation(K, fp_no_pos_word_pipeline, X_train, Y_train.values.ravel())
    no_pos_word_score = sum(no_pos_word_scores) / K
    pos_word_delta_accuracy = fp_total_score - no_pos_word_score
    print(f'Accuracy delta when adding positive word count feature: {pos_word_delta_accuracy}')

    # Repeat the same procedure, now taking out the negative word count feature
    fp_no_neg_word = copy.deepcopy(feature_pipeline)
    fp_no_neg_word.set_params(neg_word_count='drop')
    fp_no_neg_word_pipeline = Pipeline([
        ('features', fp_no_neg_word),
        ('classifier', svc)
    ])
    no_neg_word_scores = k_fold_cross_validation(K, fp_no_neg_word_pipeline, X_train, Y_train.values.ravel())
    no_neg_word_score = sum(no_neg_word_scores) / K
    neg_word_delta_accuracy = fp_total_score - no_neg_word_score
    print(f'Accuracy delta when adding negative word count feature: {neg_word_delta_accuracy}')

    # Repeat the same procedure, now taking out the uppercase word count feature
    fp_no_uppercase_word = copy.deepcopy(feature_pipeline)
    fp_no_uppercase_word.set_params(uppercase_word_count='drop')
    fp_no_uppercase_word_pipeline = Pipeline([
        ('features', fp_no_uppercase_word),
        ('classifier', svc)
    ])
    no_uppercase_word_scores = k_fold_cross_validation(K, fp_no_uppercase_word_pipeline, X_train, Y_train.values.ravel())
    no_uppercase_word_score = sum(no_uppercase_word_scores) / K
    uppercase_word_delta_accuracy = fp_total_score - no_uppercase_word_score
    print(f'Accuracy delta when adding uppercase word count feature: {uppercase_word_delta_accuracy}')

    # Repeat the same procedure, now taking out the average word length feature
    fp_no_avg_word = copy.deepcopy(feature_pipeline)
    fp_no_avg_word.set_params(avg_word_length='drop')
    fp_no_avg_word_pipeline = Pipeline([
        ('features', fp_no_avg_word),
        ('classifier', svc)
    ])
    no_avg_word_scores = k_fold_cross_validation(K, fp_no_avg_word_pipeline, X_train, Y_train.values.ravel())
    no_avg_word_score = sum(no_avg_word_scores) / K
    avg_word_delta_accuracy = fp_total_score - no_avg_word_score
    print(f'Accuracy delta when adding average word length feature: {avg_word_delta_accuracy}')

    # Repeat the same procedure, now taking out the exclamation point count feature
    fp_no_exclamation = copy.deepcopy(feature_pipeline)
    fp_no_exclamation.set_params(exclamation='drop')
    fp_no_exclamation_pipeline = Pipeline([
        ('features', fp_no_exclamation),
        ('classifier', svc)
    ])
    no_exclamation_scores = k_fold_cross_validation(K, fp_no_exclamation_pipeline, X_train, Y_train.values.ravel())
    no_exclamation_score = sum(no_exclamation_scores) / K
    exclamation_delta_accuracy = fp_total_score - no_exclamation_score
    print(f'Accuracy delta when adding exclamation point count feature: {exclamation_delta_accuracy}')

    # Repeat the same procedure, now taking out the useful column value feature
    fp_no_useful = copy.deepcopy(feature_pipeline)
    fp_no_useful.set_params(useful_value='drop')
    fp_no_useful_pipeline = Pipeline([
        ('features', fp_no_useful),
        ('classifier', svc)
    ])
    no_useful_scores = k_fold_cross_validation(K, fp_no_useful_pipeline, X_train, Y_train.values.ravel())
    no_useful_score = sum(no_useful_scores) / K
    useful_delta_accuracy = fp_total_score - no_useful_score
    print(f'Accuracy delta when adding useful column value feature: {useful_delta_accuracy}')

    # Plot the accuracy deltas
    # Use feature names on x axis, accuracy deltas on y axis
    features = ['tfidf', 'pos_word_count', 'neg_word_count', 'uppercase_word_count',
                'avg_word_length', 'exclamation_count', 'useful_value']
    accuracy_deltas = [tfidf_delta_accuracy, pos_word_delta_accuracy, neg_word_delta_accuracy,
                       uppercase_word_delta_accuracy, avg_word_delta_accuracy,
                       exclamation_delta_accuracy, useful_delta_accuracy]
    plt.figure(num=None, figsize=(12, 6))
    plt.bar(features, accuracy_deltas, width=0.3)
    plt.axhline(0, color='black')
    plt.xlabel('Features')
    plt.ylabel('Accuracy (delta)')


"""
BEGINNING EXECUTION OF PROGRAM
"""
# Read in json formatted yelp training set review data into a DataFrame
yelp_data = pd.read_json('../datasets/yelp_training_set_review_small.json', lines=True)

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
    'Support Vector Machine (SVC)': svm.LinearSVC(multi_class='crammer_singer', max_iter=max_svm_iters),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression()
}

# Iterate through our classifier dict, setup a pipeline with our original features, then make predictions
for name, classifier in classifier_dict.items():
    pipeline = Pipeline([
        ('features', feature_pipeline),
        ('classifier', classifier)
    ])

    make_prediction(X, Y, pipeline, name)

# Plot graph of classifiers and accuracy on x,y axis
plot_accuracy_graph()

# Predict zomato restaurant reviews using our best (most accurate) feature/classifier pipeline
predict_zomato_reviews()

# Perform feature ablation for all features:
# Compute accuracy after dropping each feature one by one, then compare the accuracy delta
#perform_feature_ablation(X, Y)
