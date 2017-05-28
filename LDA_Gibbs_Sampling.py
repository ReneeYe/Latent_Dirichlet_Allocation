import numpy as np
import util



class LDA(object):
	"""
	The implement of LDA via collapsed gibbs sampling.

	"""
	def __init__(self, num_topic,alpha=0.1,beta=0.01,num_iter=2000,random_seed=1):
		#assure no meaningless values
		assert alpha>0 
		assert beta >0

		self.num_topic = num_topic
		self.num_iter = num_iter
		self.alpha = alpha
		self.beta = beta 
		self.random_seed = random_seed


	def fit(self, textfiles_path):

		self.corpus, self.title, self.vocab= corpus, title, vocab  = util.path2corpus(textfiles_path) 
		np.random.seed(self.random_seed)
		self._initialise(corpus)

		#gibbs sampling
		for it in range(self.num_iter):
			self._gibbs_sampling()

		self._output_prep()

		del self.d_list
		del self.w_list
		del self.z_list
		del self._n_zw
		del self._n_z
		del self._n_dz
		del self._n_d

		return self

	def _initialise(self, corpus):
		#Four arrays need to be initialed
		#Topic-Vocabulary, Topic, Document-Topic, Document
		self.D, self.V = D, V = corpus.shape
		n_t = self.num_topic

		self._n_zw = np.zeros((n_t, V),dtype=np.intc)
		self._n_z = np.zeros((n_t),dtype = np.intc)
		self._n_dz = np.zeros((D,n_t),dtype=np.intc)
		self._n_d = np.zeros((D),dtype= np.intc)


		self._n = n = int(corpus.sum())
		self.d_list, self.w_list = util.array2list(corpus)
		self.z_list = []

		for i in range(n):
			d= self.d_list[i] 
			w= self.w_list[i] 
			#get topic assignment randomly
			z = np.random.randint(0,n_t,dtype=np.intc)
			self.z_list.append(z)

			self._n_zw[z][w] += 1
			self._n_z[z] += 1
			self._n_dz[d][z] += 1 
			self._n_d[d] += 1
		
		#assert self._n_zw.sum() == self._n

		 

	def _gibbs_sampling(self):
		"""
		Only iterate once
		"""
		beta = self.beta
		alpha = self.alpha
		n_t = self.num_topic
		V = self.V

		for i in range(self._n):
			d= self.d_list[i] 
			w= self.w_list[i] 
			z= self.z_list[i]

			self._n_zw[z][w] -= 1
			self._n_z[z] -= 1
			self._n_dz[d][z] -= 1 
			self._n_d[d] -= 1

			#K dimension
			multi_prop = (self._n_zw[:,w]+beta).astype(float)/(self._n_z + V*beta) * \
						(self._n_dz[d]+ alpha).astype(float)/(self._n_d[d] + n_t*alpha)


			assert multi_prop.shape[0] == n_t

			new_z =  np.random.multinomial(1, multi_prop/sum(multi_prop)).argmax()

			self.z_list[i] = new_z
			self._n_zw[new_z][w] += 1
			self._n_z[new_z] += 1
			self._n_dz[d][new_z] += 1 
			self._n_d[d] += 1

	def _output_prep(self):
		"""
		prepare for two arrays, 1. Topic-Vocabulary matrix to understand the topic
								2. Document-Topic to see the most relevent topic 
		"""
		beta = self.beta
		alpha = self.alpha
		n_t = self.num_topic
		V = self.V
		self.results = {}
		self.results["topic-vocabulary"] = (self._n_zw + beta).astype(float)/(self._n_z + V*beta)[:,np.newaxis]
		self.results["document-topic"] = (self._n_dz+ alpha).astype(float)/(self._n_d +  n_t*alpha)[:,np.newaxis]


