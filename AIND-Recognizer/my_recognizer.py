import warnings

from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
	""" Recognize test word sequences from word models set

   :param models: dict of trained models
	   {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
	   both lists are ordered by the test set word_id
	   probabilities is a list of dictionaries where each key a word and value is Log Liklihood
		   [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
			{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
			]
	   guesses is a list of the best guess words ordered by the test set word_id
		   ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	probabilities = []
	guesses = []

	# return probabilities, guesses
	all_words_x_lengths = test_set.get_all_Xlengths()

	for id in range(len(test_set.wordlist)):
		x, lengths = all_words_x_lengths[id]
		word_probs = {}
		best_score = float("-inf")
		word_guess = None
		for word_candidate, model in models.items():
			try:
				score = model.score(x, lengths)
				word_probs[word_candidate] = score
				if score > best_score:
					best_score = score
					word_guess = word_candidate
			except Exception as e:
				pass

		probabilities.append(word_probs)
		guesses.append(word_guess)

	return probabilities, guesses
