# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:09:31 2017

@author: YeRong
"""

import pandas as pd
import os
from Topic_Modelling import TOPIC_MODEL
import numpy as np
import re 
import time 
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression

def weight_count(row):
#well, seems no use lah...
    if row["Score"] in [1,2]:
        return 1/82012
    elif row["Score"] ==3:
        return 1/42639
    return 1/443777


path = os.getcwd()
filename = "experiment_food_data.csv"
full_path = os.path.join(path, filename)

food_data = pd.read_csv(filename,delimiter= ",")
food_review_data= food_data.Text
title = food_data.Summary
score = food_data.Score

#perplexity = [337.41041277780926, 318.06611909153571, 304.09620247668897, 298.8329599035672, 295.79340702181281, 293.86828429884423, 293.52162339346364,301.98194790733504,313.13045030760082]
loglikelihood = list()
perplexity = list()
accuracy = list()
#topics = [0 for n in range(20)]
top_n_word = 10
topic_number_list_old = [5,10,15,20,25,30,50,75,100,150]
topic_number_list = [30]
n_t_len = len(topic_number_list)
#assign number of topic here.
prop_list = [0.1,0.15,0.2,0.3,0.5]
accuracy_binary_word = []
accuracy_binary_lda = []
accuracy_multi_word = []
accuracy_multi_lda = []

for i in range(len(prop_list)):
    start_time= time.time()
    # first from num-perplexity plot find the optimal number of topics
    print('=========== topic number =',topic_number_list[0],'================')
    tp_model = TOPIC_MODEL()
    model= tp_model.fit(food_review_data,num_topic= topic_number_list[0],train_prop=prop_list[i],random_seed = 2)
    
    vocab_train = tp_model.vocab_train
    for k,t_v in enumerate(model.results["topic-vocabulary"]):
        top_topic_words_index = t_v.argsort()[:-top_n_word:-1]
        top_topic_words = np.array(vocab_train)[top_topic_words_index]
        print('Topic {}: {}'.format(k, ' '.join(top_topic_words)))
    
    print('perplexity = ',model.results["perplexity"])
    perplexity.append(model.results["perplexity"])
    loglikelihood.append(model.results['log_likelihood'])
    #plot the log likelihood
    #plt.scatter(range(len(model.results['log_likelihood'])),model.results['log_likelihood'])
    #plt.xlabel('number of iteration')
    #plt.ylabel('log likelihood')
    
    # classification - SVM
    topics = [0 for n in range(topic_number_list[0])]
    attitude = {'negative': 0, 'positive': 0}
    food_review_train = []
    title_train = []
    food_review_test = []
    title_test = []
    
    for i in range(len(model.results['train_index'])):
        if model.results['train_index'][i] == 1:
            food_review_train.append(food_review_data[i])
            title_train.append(title[i])
        else:
            food_review_test.append(food_review_data[i])
            title_test.append(title[i])
    
    x_train = list()
    for k in range(len(food_review_train)):    
        x_train.append(list(model.results["document-topic-theta"][k]))    
        theta = model.results["document-topic-theta"][k]
        t = theta.argmax()
        if k<10:
            print("{}: (top topic: {})".format(food_review_data[k][:25], t))
        topics[t] +=1 
    
    x_test = list()
    for k in range(len(food_review_test)):
        x_test.append(list(model.results['theta-test'][k]))

    # binary
    sentiment = []
    sentiment_train = []
    sentiment_test = []
    # 3-class
    attitude = []
    attitude_train = []
    attitude_test = []

    for i in range(len(score)):
        if score[i] == 1 or score[i] ==2:
            sentiment.append(0)
            attitude.append(0)
        else:
            sentiment.append(1)
            if score[i] == 3:
                attitude.append(1)
            else:
                attitude.append(2)         
                
    for i in range(len(model.results['train_index'])):
        if model.results['train_index'][i] == 1:
            sentiment_train.append(sentiment[i])
            attitude_train.append(attitude[i])
        else:
            sentiment_test.append(sentiment[i])
            attitude_test.append(attitude[i])
    
    
    # basic binary classification - logit
    print('============= binary classification ==================')
    corpus_train = model.results['corpus_train']
    corpus_test = model.results['corpus_test']
    #word feature - bin
    clf_basic_binary = LogisticRegression(C=1000)
    clf_basic_binary.fit(corpus_train,sentiment_train)
    sentiment_prediction_basic = clf_basic_binary.predict(corpus_test)
    acc = clf_basic_binary.score(corpus_test,sentiment_test)
    print('binary word acc =',acc)
    accuracy_binary_word.append(acc)
    
    # logistic for lda binary classification
    clf_binary_lda = LogisticRegression(C=1000)
    clf_binary_lda.fit(x_train, sentiment_train)
    sentiment_prediction_lda = clf_binary_lda.predict(x_test)
    acc = clf_binary_lda.score(x_test,sentiment_test)
    print('binary lda acc =',acc)
    accuracy_binary_lda.append(acc)
    
    # multi-class svm
    print('============= 3-class classification ==================')
    clf_basic_multi = LogisticRegression(C=1000, max_iter=1000, multi_class='ovr')
    clf_basic_multi.fit(corpus_train,attitude_train)
    attitude_prediction_basic = clf_basic_multi.predict(corpus_test)
    acc = clf_basic_multi.score(corpus_test,attitude_test)
    print('3-class word acc =',acc)
    accuracy_multi_word.append(acc)
    
    # lda
    clf_multi_lda = LogisticRegression(C=1000, max_iter=1000, multi_class='ovr')
    clf_multi_lda.fit(x_train,attitude_train)
    attitude_prediction_lda = clf_multi_lda.predict(x_test)
    acc = clf_multi_lda.score(x_test,attitude_test)
    print('3-class lda acc =',acc)
    accuracy_multi_lda.append(acc)
    # classification - end
    
    present_time = time.time()
    print("This iteration ---",round(present_time - start_time,2),"seconds ---")

# Also, plot the number of topics vs. perplexity