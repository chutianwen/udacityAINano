import warnings

from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold

from asl_utils import combine_sequences
import numpy as np

class ModelSelector(object):
	'''
	base class for model selection (strategy design pattern)
	'''

	def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
	             n_constant=3,
	             min_n_components=2, max_n_components=10,
	             random_state=14, verbose=False):
		self.words = all_word_sequences
		self.hwords = all_word_Xlengths
		# Sequences of words. [Seq1, Seq2, Seq3 ... SeqN]
		self.sequences = all_word_sequences[this_word]
		self.X, self.lengths = all_word_Xlengths[this_word]
		self.this_word = this_word
		self.n_constant = n_constant
		self.min_n_components = min_n_components
		self.max_n_components = max_n_components
		self.random_state = random_state
		self.verbose = verbose

	def select(self):
		raise NotImplementedError

	def base_model(self, num_states):
		# with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		# warnings.filterwarnings("ignore", category=RuntimeWarning)
		try:
			hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
			                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

			if self.verbose:
				print("model created for {} with {} states".format(self.this_word, num_states))
			return hmm_model
		except:
			if self.verbose:
				print("failure on {} with {} states".format(self.this_word, num_states))
			return None


class SelectorCV(ModelSelector):
	''' select best model based on average log Likelihood of cross-validation folds

	'''

	def scoreModel(self, num_states, split_method):

		log_l = cnt = 0
		if split_method is None:
			try:
				model = self.base_model(num_states)
				log_l += model.score(self.X, self.sequences)
			except Exception as e:
				pass
		else:
			for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
				x_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
				x_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

				try:
					model = GaussianHMM(n_components=num_states, n_iter=1000, random_state=self.random_state).fit(x_train, lengths_train)
					log_l += model.score(x_test, lengths_test)
					cnt += 1

				except Exception as e:
					pass

		# print("num_states:{}\tScore:{}\tCnt:{}".format(num_states, log_l, cnt))

		if cnt == 0:
			return -float('inf')
		else:
			return log_l / cnt

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		n_split = min(len(self.sequences), 3)
		split_method = None
		if n_split > 2:
			split_method = KFold(n_split)
		num_states = range(self.min_n_components, self.max_n_components + 1)
		best_num = max(num_states, key=lambda num: self.scoreModel(num, split_method))
		return self.base_model(best_num)


class SelectorConstant(ModelSelector):
	""" select the model with value self.n_constant

	"""

	def select(self):
		""" select based on n_constant value

		:return: GaussianHMM object
		"""
		best_num_components = self.n_constant
		return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
	""" select the model with the lowest Bayesian Information Criterion(BIC) score

	http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
	Bayesian information criteria: BIC = -2 * logL + p * logN
	"""

	def select(self):
		""" select the best model for self.this_word based on
		BIC score for n between self.min_n_components and self.max_n_components

		:return: GaussianHMM object
		"""
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		score_best = float("inf")
		best_model = None
		for numStates in range(self.min_n_components, self.max_n_components + 1):
			try:
				model = self.base_model(numStates)
				log_l = model.score(self.X, self.lengths)
				p = numStates
				log_n = np.log(len(self.lengths))
				score = -2 * log_l + p * log_n
				if score < score_best:
					score_best = score
					best_model = model
			except Exception as e:
				pass

		while best_model is None:
			best_model = self.base_model(np.random.randint(self.min_n_components, self.max_n_components + 1))

		return best_model

class SelectorDIC(ModelSelector):
	''' select best model based on Discriminative Information Criterion

	Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
	Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
	https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
	DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
	'''
	def antiTerm(self, model):
		score_sum = 0
		cnt = 0
		for word in self.hwords:
			if word != self.this_word:
				X, lengths = self.hwords[word]
				try:
					log_l = model.score(X, lengths)
					score_sum += log_l
					cnt += 1
				except Exception as e:
					pass
		if cnt == 0:
			return score_sum
		else:
			return score_sum/ cnt

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		score_best = float("-inf")
		best_model = None
		for numStates in range(self.min_n_components, self.max_n_components + 1):
			try:
				model = self.base_model(numStates)
				log_l = model.score(self.X, self.lengths)
				antiTerm = self.antiTerm(model)
				score = log_l - antiTerm
				if score > score_best:
					score_best = score
					best_model = model
			except Exception as e:
				pass

		while best_model is None:
			best_model = self.base_model(np.random.randint(self.min_n_components, self.max_n_components + 1))

		return best_model