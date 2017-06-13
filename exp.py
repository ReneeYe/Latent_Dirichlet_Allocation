# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:09:31 2017

@author: Think
"""

import pandas as pd
import os
from Topic_Modelling import TOPIC_MODEL
import numpy as np
import re 
import time 
import matplotlib.pyplot as plt
from sklearn import svm

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

#perplexity = [420.389644669,386.012709945,390.968633901,359.01287044,363.477783552,374.527104483,369.757459764,369.33909436,430.10437824,400.793577018]
loglikelihood = list()
perplexity = list()
topics = [0 for n in range(20)]
top_n_word = 10
topic_number_list_old = [5,10,15,20,25,30,50,75,100,150]
topic_number_list = [20]
n_t_len = len(topic_number_list)
#assign number of topic here.
for i in range(n_t_len):
    start_time= time.time()
    print('=========== topic number =',topic_number_list[i],'================')
    tp_model = TOPIC_MODEL()
    model= tp_model.fit(food_review_data,num_topic= topic_number_list[i],train_prop=0.9,random_seed = 1)
    
    vocab_train = tp_model.vocab_train
    for i,t_v in enumerate(model.results["topic-vocabulary"]):
        top_topic_words_index = t_v.argsort()[:-top_n_word:-1]
        top_topic_words = np.array(vocab_train)[top_topic_words_index]
        print('Topic {}: {}'.format(i, ' '.join(top_topic_words)))
    
    print('perplexity = ',model.results["perplexity"])
    perplexity.append(model.results["perplexity"])
    loglikelihood.append(model.results['log_likelihood'])
    #plot the log likelihood
    #plt.scatter(range(len(model.results['log_likelihood'])),model.results['log_likelihood'])
    #plt.xlabel('number of iteration')
    #plt.ylabel('log likelihood')
    topics = [0 for n in range(20)]
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
    
    x = list()
    for k in range(len(food_review_train)):    
        x.append(list(model.results["document-topic-theta"][k]))        
        theta = model.results["document-topic-theta"][k]
        t = theta.argmax()
        if theta[9] > theta[12]:
            emotion = 'negative'
        else:
            emotion = 'positive'
        attitude[emotion] += 1
        # print("{}: (top topic: {},{})".format(food_review_data[k][:25], t,emotion))
        topics[t] +=1 
    
    sentiment = []
    for i in range(len(score)):
        if score[i] == 1 or score[i] ==2:
            sentiment.append(0)
        else:
            sentiment.append(1)
    sentiment_train = []
    sentiment_test = []
    for i in range(len(model.results['train_index'])):
        if model.results['train_index'][i] == 1:
            sentiment_train.append(sentiment[i])
        else:
            sentiment_test.append(sentiment[i])
        
    present_time = time.time()
    print("This iteration ---",round(present_time - start_time,2),"seconds ---")
