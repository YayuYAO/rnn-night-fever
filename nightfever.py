import csv
import random
import time
import numpy as np
import torch
import torch.nn as nn

class NightFever():
	""" Core model with the three RNN circuit """
	pass


class RNN(nn.Module):
	""" Basic RNN block"""

	def __init__(self, n_dim, n_hidden, dev, batch_size=None):
		
		super(RNN, self).__init__()

		self.n_dim     = n_dim
		self.n_hidden  = n_hidden
		self.dev       = dev

		self.rnn = nn.GRU(n_dim, n_hidden, n_layers=1, batch_first=True).to(self.dev)
		self.out  = nn.Linear(n_hidden, n_dim, bias=True).to(self.dev)
		self.sig  = nn.Sigmoid().to(self.dev)

		if batch_size is not None:
			self.init_hidden(batch_size)
		else:
			self.hidden = None

	def init_hidden(self, batch_size):
		h_size = [self.n_layers, batch_size, self.n_hidden]
		self.hidden = torch.zeros(*h_size).to(self.dev)

	def forward(self, x):

		self.init_hidden(x.shape[0])
		x, self.hidden = self.rnn(x, self.hidden)
		# Keeping the outputs off 0 and 1 bounds the BCELoss gradient (see P_EPS)
		x_out  = self.sig(self.out(x)).clamp(P_EPS, 1 - P_EPS)

		return x_out

