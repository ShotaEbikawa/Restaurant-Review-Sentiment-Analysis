#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:48:41 2019

@author: anastasiosgrigoriou
"""

from sklearn.base import BaseEstimator, TransformerMixin


class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, name):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in name.split()])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df['road_name'].apply(self.average_word_length)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self