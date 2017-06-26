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


path = os.getcwd()
filename = "experiment_food_data.csv"
full_path = os.path.join(path, filename)

food_data = pd.read_csv(filename,delimiter= ",")
food_review_data= food_data.Text
title = food_data.Summary
score = food_data.Score

#perplexity = [67.733285504579698, 68.213576975115927, 67.805343212433399, 67.687125242706401,  67.246006936900812, 67.657215900624294, 67.143525002949261, 68.1965316124368,  69.213960767121861,  72.311919971434577]
loglikelihood = list()
perplexity = list()
accuracy = list()
#topics = [0 for n in range(20)]
top_n_word = 10
#topic_number_list_old = [2,3,4,5,6,8,10,15,20,30]
topic_number_list = [2,3,4,5,6,8,10,15,20,30]
n_t_len = len(topic_number_list)
#assign number of topic here.
#prop_list = []
prop_list = [0.1,0.15,0.2,0.3,0.5,0.8]
accuracy_binary_word = []
accuracy_binary_lda = []
accuracy_multi_word = []
accuracy_multi_lda = []

for i in range(len(prop_list)):
#for i in range(n_t_len):
    start_time= time.time()
    # first from num-perplexity plot find the optimal number of topics
    print('=========== proportion of training =',prop_list[i],' ================')
    tp_model = TOPIC_MODEL()
    model= tp_model.fit(title,num_topic= 10,train_prop=prop_list[i],random_seed = 1)
    
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
    
    present_time = time.time()
    print("This iteration ---",round(present_time - start_time,2),"seconds ---")

plt.plot(topic_number_list,perplexity)
plt.xlabel('number of topics')
plt.ylabel('perplexity')