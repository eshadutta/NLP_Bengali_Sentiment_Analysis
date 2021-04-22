import nltk
import numpy as np
import re
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
from bnlp import NLTKTokenizer
from bnlp.corpus import stopwords, punctuations
from bnlp.corpus.util import remove_stopwords
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


def clean_text(X):
    #removing whitespaces, punctuations, digits and english words from text
    converted_X =[]
    punctuation_list = ['[',',','-','_','=',':','+','$','@',
                        '~','!',';','/','^',']','{','}','(',')','<','>','.']
    whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
    bangla_digits = u"[\u09E6\u09E7\u09E8\u09E9\u09EA\u09EB\u09EC\u09ED\u09EE\u09EF]+"
    english_chars = u"[a-zA-Z0-9]"
    punc = u"[(),$%^&*+={}\[\]:\"|\'\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+"
    bangla_fullstop = u"\u0964"     #bangla fullstop(dari)
    punctSeq   = u"['\"“”‘’]+|[.?!,…]+|[:;]+"
    
    
    for x in X:
        x = re.sub(bangla_digits, " ", x)
        x = re.sub(punc, " ", x)
        x = re.sub(english_chars, " ", x)
        x = re.sub(bangla_fullstop, " ", x)
        x = re.sub(punctSeq, " ", x)
        x = whitespace.sub(" ", x).strip()
        
        x = re.sub(r'https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE)
        x = re.sub(r'\<a href', ' ', x)
        x = re.sub(r'&amp;‘:‘ ’', '', x) 
        x = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]। ,', ' ', x)
        x = re.sub(r'<br />', ' ', x)
        x = re.sub(r'\'', ' ', x)
        x = re.sub(r"[\@$#%~+-\.\'।\"]"," ",x)
        x = re.sub(r"(?m)^\s+", "", x)
        x = re.sub("[()]","",x)
        x = re.sub("[‘’]","",x)
        x = re.sub("[!]","",x)
        x = re.sub("[/]","",x)
        x = re.sub("[:]","",x)
        x = re.sub('\ |\?|\.|\!|\/|\;|\:', ' ',x)
        x = x.strip("/")
        converted_X.append(x)
    converted_X = np.array(converted_X)
    
    return converted_X

if __name__ == "__main__":
    pos_file = open('data/all_positive_8500.txt',encoding='utf8')
    pos_lines = pos_file.readlines()
    pos_values = [1]*len(pos_lines)
    
    neg_file = open('data/all_negative_3307.txt',encoding='utf8')
    neg_lines = neg_file.readlines()
    neg_values = [0]*len(neg_lines)
    
    X = np.array(pos_lines + neg_lines)
    y = np.array(pos_values + neg_values)
    
    
    skf = StratifiedKFold(n_splits=10, random_state = 42)
    fold_count = 0
    
    vocab = TfidfVectorizer().fit(X)
    X = vocab.transform(X)
    
    print("#### Multinomial Naive Bayes ####")
         
    for train_index, test_index in skf.split(X, y):
        fold_count += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        mnb = MultinomialNB()
        mnb.fit(X_train,y_train)
        predmnb = mnb.predict(X_test)
        #print("Confusion Matrix for Multinomial Naive Bayes:")
        #print(confusion_matrix(y_test,predmnb))
        print("MNB Fold: ", fold_count, " Score:",round(accuracy_score(y_test,predmnb)*100,2))
        report = classification_report(y_test,predmnb,output_dict=True)
        print(classification_report(y_test,predmnb))
        print("\n")
        
    
    print("#### Random Forest Classifier ####")
    fold_count = 0
    for train_index, test_index in skf.split(X, y):
        fold_count += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rmfr = RandomForestClassifier(n_estimators=50)
        rmfr.fit(X_train,y_train)
        predrmfr = rmfr.predict(X_test)
        #print("Confusion Matrix for Random Forest Classifier:")
        #print(confusion_matrix(y_test,predrmfr))
        print("RF Fold: ", fold_count, " Score:",round(accuracy_score(y_test,predrmfr)*100,2))
        print(classification_report(y_test,predrmfr))
        #fpr, tpr, _ = metrics.roc_curve(y_test,  predrmfr)
        #auc = metrics.roc_auc_score(y_test, predrmfr)
        #plt.plot(fpr,tpr,label="Fold : " + str(fold_count))
        print("\n")
    #plt.xlabel("False Positive Rate")
    #plt.ylabel("True Positive Rate")
    #plt.title("ROC for Random Forest Classifier")
    #plt.legend()
    #plt.savefig("ROC.jpg")
        
    
    print("#### Decision Tree Classifier ####")   
    fold_count = 0
    for train_index, test_index in skf.split(X, y):
        fold_count += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        dt = DecisionTreeClassifier()
        dt.fit(X_train,y_train)
        preddt = dt.predict(X_test)
        
        print("DT Fold: ", fold_count, " Score:",round(accuracy_score(y_test,preddt)*100,2))
        print(classification_report(y_test,preddt))
        print("\n")
        
    print("#### SVM ####")   
    fold_count = 0
    for train_index, test_index in skf.split(X, y):
        fold_count += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        svm = SVC(random_state=101,gamma='auto')
        svm.fit(X_train,y_train)
        predsvm = svm.predict(X_test)
        print("SVM Fold: ", fold_count, " Score:",round(accuracy_score(y_test,predsvm)*100,2))
        print(classification_report(y_test,predsvm))
        print("\n")
        
    print("#### KNN ####")   
    fold_count = 0
    for train_index, test_index in skf.split(X, y):
        fold_count += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index] 
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train,y_train)
        predknn = knn.predict(X_test)
        print("KNN Fold: ", fold_count, " Score:",round(accuracy_score(y_test,predknn)*100,2))
        print(classification_report(y_test,predknn))
        print("\n")
    
    print("#### XGB ####")   
    fold_count = 0
    for train_index, test_index in skf.split(X, y):
        fold_count += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index] 
        xg = xgb.XGBClassifier()
        xg.fit(X_train,y_train)
        predxg = xg.predict(X_test)
        print("xgb Fold: ", fold_count, " Score:",round(accuracy_score(y_test,predxg)*100,2))
        print(classification_report(y_test,predxg))
        print("\n")
                