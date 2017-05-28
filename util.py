import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os
import re

def array2list(corpus):
	D,W= corpus.shape
	d_list = []
	w_list = []

	for d in range(D):
		for w in range(W):
			if corpus[d][w] > 0:
				d_list.extend([d for i in range(corpus[d][w])])
				w_list.extend([w for i in range(corpus[d][w])])
	return d_list,w_list


def path2corpus(path):
	txt_list = os.listdir(path)[1:]
	filenames = [os.path.join(path,txt) for txt in txt_list]
	vectorizer = CountVectorizer(input = 'filename',lowercase = True,stop_words = 'english')
	dtm = vectorizer.fit_transform(filenames)
	vocab = vectorizer.get_feature_names()
	title  = [ re.sub(".txt$","", txt) for txt in txt_list]
	return dtm.toarray(), title, vocab