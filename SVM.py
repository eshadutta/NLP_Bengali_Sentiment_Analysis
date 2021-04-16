#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 23:41:55 2021

@author: eshadutta
"""

import nltk
import numpy as np
import math
from sklearn.model_selection import train_test_split
from bnlp import NLTKTokenizer
from bnlp.corpus import stopwords, punctuations
from bnlp.corpus.util import remove_stopwords
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

if __name__ == "__main__":
    pos_file = open('all_positive_8500.txt')
    pos_lines = pos_file.readlines()
    pos_values = [1]*len(pos_lines)
    
    neg_file = open('all_negative_3307.txt')
    neg_lines = neg_file.readlines()
    neg_values = [0]*len(neg_lines)
    
    X = np.array(pos_lines + neg_lines)
    y = np.array(pos_values + neg_values)
    
    skf = StratifiedKFold(n_splits=10)
    bnltk = NLTKTokenizer()
    stopwords = stopwords()
    fold_count = 0
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #vectorizer = TfidfVectorizer(min_df = 5,
        #                     max_df = 0.8,
         #                    sublinear_tf = True,
          #                   use_idf = True)
        vectorizer = TfidfVectorizer(use_idf = True)
        train_vectors = vectorizer.fit_transform(X_train)
        test_vectors = vectorizer.transform(X_test)
        
        classifier_linear = svm.SVC(kernel='linear')

        classifier_linear.fit(train_vectors, y_train)

        y_pred = classifier_linear.predict(test_vectors)
        
        count = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                count += 1
            
        print("Fold: ", fold_count, " Test accuracy: ", count/len(y_test))
        
        fold_count += 1
        
        report = classification_report(y_test, y_pred, output_dict=True)
        print('positive: ', report['1'])
        print('negative: ', report['0'])

