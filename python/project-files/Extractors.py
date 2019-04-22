#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:48:41 2019

@author: anastasiosgrigoriou
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Takes in series, outputs average word length for each review text in series,
    returns DataFrame
    """

    def __init__(self):
        pass

    def average_word_length(self, name):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in name.split()])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.average_word_length).to_frame()

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
    
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
