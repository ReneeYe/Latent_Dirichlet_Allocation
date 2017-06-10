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

perplexity = list()
loglikelihood = list()
top_n_word = 10

topic_number_list = range(15,18)
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
    plt.scatter(range(len(model.results['log_likelihood'])),model.results['log_likelihood'])
    plt.xlabel('number of iteration')
    plt.ylabel('log likelihood')
    present_time = time.time()
    print("This iteration ---",round(present_time - start_time,2),"seconds ---")
