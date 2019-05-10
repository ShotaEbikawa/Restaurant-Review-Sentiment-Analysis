#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:48:41 2019

@author: anastasiosgrigoriou
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import linear_model
import numpy as np


class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Takes in series, outputs average word length for each review text in series,
    returns DataFrame
    """
    def __init__(self):
        pass

    def average_word_length(self, review_text):
        """Helper code to compute average word length in the text review"""
        return np.mean([len(word) for word in review_text.split()])

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

    def __init__(
        self, 
        estimator = linear_model.SGDClassifier(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
    
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