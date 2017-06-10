import os
import util
import LDA_Gibbs_Sampling
import numpy as np


TOPIC = 10 

datapath= os.path.join(os.path.dirname(__file__),'datasets')

print('running LDA.gibbs')
lda = LDA_Gibbs_Sampling.LDA(num_topic = TOPIC,alpha = 1 )
print('fitting...')
model =  lda.fit(datapath)
print('LDA model complete')

title = model.title
vocab = model.vocab

top_n_word = 6

for i,t_v in enumerate(model.results["topic-vocabulary"]):
    top_topic_words_index = t_v.argsort()[:-top_n_word:-1]
    top_topic_words = np.array(vocab)[top_topic_words_index]
    print('Topic {}: {}'.format(i, ' '.join(top_topic_words))) 

for i in range(100):
    print("{} (top topic: {})".format(title[i], model.results["document-topic-theta"][i].argmax()))

count_list = np.zeros(25,dtype=np.intc)
for i,t in enumerate(title):
    index = model.results["document-topic-theta"][i].argmax()
    count_list[index] += 1
print(list(range(25)))
print(count_list)    