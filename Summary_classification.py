# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:53:47 2017

@author: YeRong
"""

import pandas as pd
import os
#from Topic_Modelling import TOPIC_MODEL
import LDA_Gibbs_Sampling
import numpy as np
import re 
#import time 
import matplotlib.pyplot as plt
#from sklearn import svm
#from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
#from sklearn.ensemble import RandomForestClassifier as RFC
from bs4 import BeautifulSoup 
#import nltk
from nltk.corpus import stopwords 
stops = set(stopwords.words("english"))
from gensim.models import Word2Vec
#import sqlite3
#import string
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
from sklearn.metrics import  accuracy_score
import warnings
warnings.filterwarnings("ignore")
import util

def binarize_score(score):
    #set scores of 1-3 to 0 and 4-5 as 1
    if score <3:
        return 0
    else:
        return 1
        
def three_class_score(score):
    if score <3:
        return 0
    if score == 3:
        return 1
    if score > 3:
        return 2
    
def review_to_words(review):
    """
    Return a list of cleaned word tokens from the raw review
    
    """  
    #Remove any HTML tags and convert to lower case
    review_text = BeautifulSoup(review).get_text().lower() 
    
    #Replace smiliey and frown faces, ! and ? with coded word SM{int} in case these are valuable
    review_text=re.sub("(:\))",r' SM1',review_text)
    review_text=re.sub("(:\()",r' SM2',review_text)
    review_text=re.sub("(!)",r' SM3',review_text)
    review_text=re.sub("(\?)",r' SM4',review_text)    
    #keep 'not' and the next word as negation may be important
    review_text=re.sub(r"not\s\b(.*?)\b", r"not_\1", review_text)    
    #keep letters and the coded words above, replace the rest with whitespace
    nonnumbers_only=re.sub("[^a-zA-Z\_(SM\d)]"," ",review_text)      
    #Split into individual words on whitespace
    words = nonnumbers_only.split()                             
    #Remove stop words
    words = [w for w in words if not w in stops]
    return (words)

def avg_word_vectors(wordlist,size):
    """
    returns a vector of zero for reviews containing words where none of them
    met the min_count or were not seen in the training set    
    Otherwise return an average of the embeddings vectors    
    """    
    sumvec=np.zeros(shape=(1,size))
    wordcnt=0
    for w in wordlist:
        if w in model_w2v:
            sumvec += model_w2v[w]
            wordcnt +=1
    
    if wordcnt ==0:
        return sumvec
    else:
        return sumvec / wordcnt


path = os.getcwd()
filename = "experiment_food_data.csv"
full_path = os.path.join(path, filename)

food_data = pd.read_csv(filename,delimiter= ",")
food_review_data= food_data.Text
title = food_data.Summary
score = food_data.Score
food_data['Score_binary']=food_data['Score'].apply(binarize_score)
food_data['Score_3']=food_data['Score'].apply(three_class_score)
food_data['word_list']=food_data['Summary'].apply(review_to_words)

#perplexity = [67.733285504579698, 68.213576975115927, 67.805343212433399, 67.687125242706401,  67.246006936900812, 67.657215900624294, 67.143525002949261, 68.1965316124368,  69.213960767121861,  72.311919971434577]
loglikelihood = list()
perplexity = list()
accuracy = list()
#topics = [0 for n in range(20)]
top_n_word = 10
topic_number_list_old = [2,3,4,5,6,8,10,15,20,30]
#topic_number_list = [10]
topci_number = 10
#n_t_len = len(topic_number_list)
#assign number of topic here.
#prop_list = []
prop_list = [0.1,0.15,0.2,0.3,0.5,0.8,0.85,0.9,0.95]
accuracy_binary_word = []
accuracy_binary_lda = []
accuracy_3_word = []
accuracy_3_lda = []
accuracy_3_w2v = []
accuracy_binary_w2v = []

# first from num-perplexity plot find the optimal number of topics
print('=========== topic number = 10 ================')
#tp_model = TOPIC_MODEL()
#model= tp_model.fit(title,num_topic= 10,train_prop=1,random_seed = 1)

content = list(title)
corpus,vocab = util.corpus2dtm(util.content2corpus(content))
lda = LDA_Gibbs_Sampling.LDA(num_topic = 10,alpha = 1 )
model = lda.fit2(corpus)
    
for k,t_v in enumerate(model.results["topic-vocabulary"]):
    top_topic_words_index = t_v.argsort()[:-top_n_word:-1]
    top_topic_words = np.array(vocab)[top_topic_words_index]
    print('Topic {}: {}'.format(k, ' '.join(top_topic_words)))
    
# word2vec classification
for i in range(len(prop_list)):
    X_train, X_test, y_train, y_test = train_test_split(food_data['word_list'], food_data['Score_binary'], test_size=prop_list[i], random_state=13)
    X_3train, X_3test, y_3train, y_3test = train_test_split(food_data['word_list'], food_data['Score_3'], test_size=prop_list[i], random_state=13)
    #size of hidden layer (length of continuous word representation)
    dimsize=400
    #train word2vec on 80% of training data
    model_w2v = Word2Vec(X_train.values, size=dimsize, window=5, min_count=5, workers=4)
    #create average vector for train and test from model
    #returned list of numpy arrays are then stacked 
    X_train=np.concatenate([avg_word_vectors(w,dimsize) for w in X_train])
    X_test=np.concatenate([avg_word_vectors(w,dimsize) for w in X_test])
    
    X_3train = np.concatenate([avg_word_vectors(w,dimsize) for w in X_3train])
    X_3test = np.concatenate([avg_word_vectors(w,dimsize) for w in X_3test])
    #SGD
    clf = linear_model.SGDClassifier(loss='log')
    clf.fit(X_train, y_train)
    p=clf.predict_proba(X_test)
    acc = roc_auc_score(y_test,p[:,1])
    accuracy_binary_w2v.append(acc)
    
    clf3 = linear_model.SGDClassifier(loss='log')
    clf3.fit(X_3train, y_3train)
    pr=clf3.predict(X_3test)
    acc3 = accuracy_score(y_3test,pr)
    accuracy_3_w2v.append(acc3)
    
# lda classification
for i in range(len(prop_list)):
    X_lda_train, X_lda_test, y_train, y_test = train_test_split(model.results["document-topic-theta"], food_data['Score_binary'], test_size=prop_list[i], random_state=13)
    X_lda_3train, X_lda_3test, y_3train, y_3test = train_test_split(model.results["document-topic-theta"], food_data['Score_3'], test_size=prop_list[i], random_state=13)
    
    clf_lda = linear_model.SGDClassifier(loss='log')
    clf_lda.fit(X_lda_train, y_train)
    p=clf_lda.predict_proba(X_lda_test)
    acc_lda = roc_auc_score(y_test,p[:,1])
    accuracy_binary_lda.append(acc_lda)
    
    clf3 = linear_model.SGDClassifier(loss='log')
    clf3.fit(X_lda_3train, y_3train)
    p=clf3.predict(X_lda_3test)
    acc3 = accuracy_score(y_3test,p)
    accuracy_3_lda.append(acc3)
    
    X_word_train, X_word_test, y_train, y_test = train_test_split(corpus, food_data['Score_binary'], test_size=prop_list[i], random_state=13)
    X_word_3train, X_word_3test, y_3train, y_3test = train_test_split(corpus, food_data['Score_3'], test_size=prop_list[i], random_state=13)

    clf_word = linear_model.SGDClassifier(loss='log')
    clf_word.fit(X_word_train, y_train)
    p=clf_word.predict_proba(X_word_test)
    acc_word = roc_auc_score(y_test,p[:,1])
    accuracy_binary_word.append(acc_word)
    
    clf3 = linear_model.SGDClassifier(loss='log')
    clf3.fit(X_word_3train, y_3train)
    p=clf3.predict(X_word_3test)
    acc3 = accuracy_score(y_3test,p)
    accuracy_3_word.append(acc3)
     
#plot the accuracy

plt.figure(figsize=(8, 6))
plt.plot(prop_list,accuracy_binary_w2v,'b--',label = 'word2vec feature')
plt.plot(prop_list,accuracy_binary_lda,label = 'lda feature')
plt.plot(prop_list,accuracy_binary_word, 'r-.',label = 'word feature')
plt.xlabel('proportion of training')
plt.ylabel('accuracy')
plt.title('Binary Classification - negative/positive')
plt.xlim((0,1))
plt.ylim((0.5,0.8))
plt.legend(loc = 'upper right')
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(prop_list,accuracy_3_w2v,'b--',label = 'word2vec feature')
plt.plot(prop_list,accuracy_3_lda,label = 'lda feature')
plt.plot(prop_list,accuracy_3_word, 'r-.',label = 'word feature')
plt.xlabel('proportion of training')
plt.ylabel('accuracy')
plt.title('Binary Classification - negative/neutral/positive')
plt.legend(loc = 'upper right')
plt.show()
