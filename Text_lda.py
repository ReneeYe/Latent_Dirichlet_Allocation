# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:09:31 2017

@author: YeRong
"""

import pandas as pd
import os
from Topic_Modelling import TOPIC_MODEL
import numpy as np
#import re 
import time 
import matplotlib.pyplot as plt
#from sklearn import svm
#from sklearn.linear_model import LogisticRegression

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
#topics = [0 for n in range(20)]
top_n_word = 10
#topic_number_list_old = [5,10,15,20,25,30,50,75,100,150]
topic_number_list = [5,10,15,20,25,30,50,75,100]
n_t_len = len(topic_number_list)
#assign number of topic here.
#prop_list = []
for i in range(n_t_len):
    start_time= time.time()
    # first from num-perplexity plot find the optimal number of topics
    print('=========== topic number =',topic_number_list[i],'================')
    tp_model = TOPIC_MODEL()
    model= tp_model.fit(food_review_data,num_topic= topic_number_list[i],train_prop=0.2,random_seed = 2)
    
    vocab_train = tp_model.vocab_train
    for k,t_v in enumerate(model.results["topic-vocabulary"]):
        top_topic_words_index = t_v.argsort()[:-top_n_word:-1]
        top_topic_words = np.array(vocab_train)[top_topic_words_index]
        print('Topic {}: {}'.format(k, ' '.join(top_topic_words)))
    
    print('perplexity = ',model.results["perplexity"])
    perplexity.append(model.results["perplexity"])
    loglikelihood.append(model.results['log_likelihood'])
    #plot the log likelihood
    plt.scatter(range(len(model.results['log_likelihood'])),model.results['log_likelihood'])
    plt.xlabel('number of iteration')
    plt.ylabel('log likelihood')
    
    present_time = time.time()
    print("This iteration ---",round(present_time - start_time,2),"seconds ---")

plt.plot(topic_number_list,perplexity)
plt.xlabel('number of topics')
plt.ylabel('perplexity')