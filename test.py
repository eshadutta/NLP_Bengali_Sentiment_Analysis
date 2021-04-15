#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:59:31 2021

@author: eshadutta
"""

import nltk
import numpy as np
import math
from sklearn.model_selection import train_test_split
from bnlp import NLTKTokenizer
from bnlp.corpus import stopwords, punctuations
from bnlp.corpus.util import remove_stopwords
from textblob import TextBlob
from googletrans import Translator
translator = Translator()

class NaiveBayes:
    def __init__(self):
        self.pos_vocab = {}
        self.neg_vocab = {}
        
    def classify(self, words):
        pp = pn = 0
        total_pos = sum([self.pos_vocab[word] for word in self.pos_vocab])
        total_neg = sum([self.neg_vocab[word] for word in self.neg_vocab])
        total_vocab = len(set(self.pos_vocab.keys()) | set(self.neg_vocab.keys()))
        
        for word in words:
            if word in self.pos_vocab:
                pp += (math.log(self.pos_vocab[word]+1)-math.log(total_pos + total_vocab + 1)) 
            else:
                pp -= math.log(total_pos + total_vocab + 1)
            if word in self.neg_vocab:
                pn += (math.log(self.neg_vocab[word]+1)-math.log(total_neg + total_vocab + 1)) 
            else:
                pn -= math.log(total_neg + total_vocab + 1)
                
        if pp >= pn:
            return 1
        return 0
    
    def addExample(self, klass, words):
        print(words)
        for word in words:
            if klass == 1:
                if word in self.pos_vocab:
                    self.pos_vocab[word] += 1
                else:
                    self.pos_vocab[word] = 1
            else:
                if word in self.neg_vocab:
                    self.neg_vocab[word] += 1
                else:
                    self.neg_vocab[word] = 1
    

#def main():
if __name__ == "__main__":
    pos_file = open('all_positive_8500.txt')
    pos_lines = pos_file.readlines()
    pos_values = [1]*len(pos_lines)
    
    neg_file = open('all_negative_3307.txt')
    neg_lines = neg_file.readlines()
    neg_values = [0]*len(neg_lines)
    
    x = np.array(pos_lines + neg_lines)
    y = np.array(pos_values + neg_values)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    
    nb = NaiveBayes()
    
    bnltk = NLTKTokenizer()
    
    stopwords = stopwords()
    
    for i in range(len(X_train)):
        words = bnltk.word_tokenize(X_train[i])
        #words = remove_stopwords(X_train[i], stopwords)
        klass = y_train[i]
        nb.addExample(klass,words)
        
    y_pred = []
        
    for i in range(len(X_test)):
        words = bnltk.word_tokenize(X_test[i])
        #words = remove_stopwords(X_test[i], stopwords)
        y_pred.append(nb.classify(words))
        
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            count += 1
            
    print("Test accuracy: ", count/len(y_test))
        
    
    
    
    

#if __name__ == "__main__":
#    main()





'''

out = open('output.txt', "w+")

for line in lines:
    t = line.replace('।','.').strip()
    en_text = translator.translate(t).text
    out.write(en_text)
    out.write("\n")
'''
    