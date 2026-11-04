import csv
import random
import time
import numpy as np



class DiscoBand():
	""" Basic stochastic generator """

	def __init__(self, n_dim, seed=None):

		super(DiscoBand, self).__init__()

		self.n_dim   = n_dim
		self.seed    = seed
		self.rng     = np.random.default_rng(seed)  # use self.rng for all sampling
		self.harmony = self.sample_harmonic_rules()
		self.motifs  = self.sample_disco_motifs()

	def sample_harmonic_rules(self):
		# Stochastically chooses the super-structure of the generative model.
		# Outputs a dictionary (harmony) containing all the parameters of the superstructure.
		# Note: use self.rng.XXX instead of np.random.XXX for sampling 	
		pass

	def sample_disco_motifs(self):
		# Stochastically chooses the mini-structure of the generative model.
		# Outputs a list, called motifs, containing the stochastic parameters of the motifs.
		# Note: use self.rng.XXX instead of np.random.XXX for sampling 	
		pass

	def write_song(self, n_tokens):
		# Generates a stochastic sequence of observations of length n_tokens
		# Outputs a numpy array of dimensions (1, n_tokens, n_dim)
		# Note: use self.rng.XXX instead of np.random.XXX for sampling 	
		pass

	def write_album(self, n_tokens, n_songs):
		# Generates a stochastic batch (n_songs batches) of sequences (n_tokens length)
		# Outputs a numpy array of dimensions (n_songs, n_tokens, n_dim)
		pass



class DiscoFirm():
	""" Batched stochastic generator """

	def __init__(self, n_dim, n_songs, n_bands, seed=None):

		super(DiscoFirm, self).__init__()

		self.n_dim   = n_dim
		self.n_songs = n_songs
		self.seed    = np.random.SeedSequence(seed)  # seed=None draws fresh OS entropy
		self.bands   = [DiscoBand(n_dim, s) for s in self.seed.spawn(n_bands)]

	def generate_discography(self, n_tokens, n_songs):
		# Generates a stochastic batch including sequences from different stochastic GMs
		# Outputs a numpy array of dimensions (n_songs * n_bands, n_tokens, n_dim)
		pass



