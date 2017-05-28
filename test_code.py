import os
import util
import LDA_Gibbs_Sampling

TOPIC = 10 

datapath= os.path.join(os.path.dirname(__file__),'datasets')

lda = LDA_Gibbs_Sampling.LDA(num_topic = TOPIC,alpha = 1 )
model =  lda.fit(datapath)

title = model.title
vocab = model.vocab

top_n_word = 6

for i in range(TOPIC):
	topic_words = model.results["topic-vocabulary"][i].argsort()[-top_n_word:][::-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words))) 
 