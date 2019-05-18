#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:48:41 2019

@author: anastasiosgrigoriou
@author: shotaebikawa
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
import numpy as np
import warnings


class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Takes in series, outputs average word length for each review text in series,
    returns DataFrame
    """
    def __init__(self):
        pass

    def average_word_length(self, review_text):
        """Helper code to compute average word length in the text review"""
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                x = np.nanmean([len(word) for word in review_text.split()])
            except RuntimeWarning:
                x = np.NaN    
        return x 

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.average_word_length).to_frame()

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
    
class ExclamationPointCountExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def num_exclamation_points(self, review_text):
        """Helper code to compute number of exclamation points in the text review"""
        return len([char for char in review_text if char == "!"])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.num_exclamation_points).to_frame()

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class PositiveWordCountExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, positive_words):
        self.positive_words = positive_words
        
    def num_pos_words(self, review_text):
        """Helper code to compute number of 'positive' sentiment words present in text review"""
        return len([word for word in review_text.split() if word in self.positive_words])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.num_pos_words).to_frame()

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class NegativeWordCountExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, negative_words):
        self.negative_words = negative_words

    def num_neg_words(self, review_text):
        """Helper code to compute number of 'negative' sentiment words present in text review"""
        return len([word for word in review_text.split() if word in self.negative_words])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.num_neg_words).to_frame()

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
    
class UppercaseWordCountExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def num_uppercase_words(self, review_text):
        """
        First remove stop words, then only take words longer than 2 characters,
        then return total number of uppercased words
        """
        review_text = [word for word in review_text.split() if word not in stopwords.words('english') and len(word) > 2]
        return len([word for word in review_text if word.isupper()])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.num_uppercase_words).to_frame()

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
    
class UsefulValueExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def useful_value(self, useful):
        """Helper code to return the value in useful column"""
        return useful

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.useful_value).to_frame()

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
    
class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects the column from the DataFrame object identified by it's key
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class ClfSwitcher(BaseEstimator):
    """
    A Custom BaseEstimator that can switch between classifiers.
    :param estimator: sklearn object - The classifier
    """
    def __init__(
        self, 
        estimator=None
    ):
        self.estimator = estimator
    
    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)
