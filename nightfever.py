import csv
import random
import time
import numpy as np
import torch
import torch.nn as nn
import songwriter


## Global variables

P_EPS = 1e-6



## Classes

class NightFever(nn.Module):
	""" Core model with the three RNN circuit """

	def __init__(self, n_dim, dev, sizes=None, batch_size=None):

		super(NightFever, self).__init__()

		if sizes is None:
			n_hidden_perception = 2 * n_dim
			n_hidden_slow = 6 * n_dim 
			n_hidden_fast = 512
		else:
			n_hidden_perception, n_hidden_slow, n_hidden_fast = sizes

		self.n_hidden_perception = n_hidden_perception
		self.n_hidden_slow = n_hidden_slow
		self.n_hidden_fast = n_hidden_fast
		self.n_dim = n_dim
		self.dev   = dev
		self.batch_size = batch_size

		self.perception = RNN(n_dim, 1, n_hidden_perception, dev, batch_size)
		self.fast = RNN(n_dim, 1, n_hidden_fast, dev, batch_size)
		self.slow = RNN(n_dim, 2, n_hidden_slow, dev, batch_size)

		self.perc_fast = nn.Linear(n_hidden_perception, n_hidden_fast)
		self.fast_perc = nn.Linear(n_hidden_fast, n_hidden_perception)
		## linear maps from fast to slow, fast to perc, ...  


	def forward(self, x):
		(batch_size, n_tokens, n_dim) = x.shape    

		if batch_size is None:
			self.perception.init_hidden(batch_size)
			self.fast.init_hidden(batch_size)
			self.slow.init_hidden(batch_size)

		total_out = torch.zeros(batch_size, n_tokens, n_dim)

		for t in range(n_tokens):  # take the song step by step
			out = self.perception(x[:, t])
			perc2fast = self.perc_fast(out)
			perc2slow = self.perc_slow(out)
			out = self.fast(perc2fast)
			fast2slow = self.fast_slow(out)
			fast2perc = self.fast_perc(out)
			out = self.slow(perc2slow + fast2slow)
			slow2perc = self.slow_perc(out)
			slow2fast = self.slow_fast(out)
			_ = self.fast(slow2fast)
			total_out[:, t] = self.perception(fast2perc + slow2perc)

		return total_out 


class RNN(nn.Module):
	""" Basic RNN block"""

	def __init__(self, n_dim, n_layers, n_hidden, dev, batch_size=None):

		super(RNN, self).__init__()

		self.n_dim     = n_dim
		self.n_hidden  = n_hidden
		self.n_layers  = n_layers
		self.dev       = dev

		self.rnn = nn.GRU(n_dim, n_hidden, n_layers, batch_first=True).to(self.dev)
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




def loss_function():
	return nn.BCELoss()


def train_network(n_dim=10, n_hidden=32, lr=0.01, n_batches=301, sigma=0.5):
	# n_dim how many "notes" each step of a song has 

	band = songwriter.DiscoBand(n_dim = n_dim) # one world, fixed rules

	# NETWORK
	# the GRU, 32 memory; Bigger batch = smoother but slower learning; 
	# bigger memory = more capacity but slower.
	net = RNN(n_dim=n_dim, n_layers=1, n_hidden=n_hidden, dev='cpu')
	loss_fn = loss_function()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)

	# n_hidden = 32 hidden memory, bigger = more capacity
	# BCELoss (binary cross-entropy) -> binary note, probability between 0 and 1.
	# Optimizer does the learning. Too big -> unstable, Too small -> slow learning


	# Training loop

	for step in range(n_batches): # = N +1 to make sure step N is included

		optimizer.zero_grad()  # forget last step, clear the old corrections

		# clean = noise free songs 
		clean = band.write_album(20, 32) # 32 songs, 20 steps
		clean = torch.tensor(clean, dtype = torch.float32) # numpy

		noisy  = clean + sigma * torch.randn_like(clean) # make noisy copy of the songs (Gaussian); 0.5 is how much noise, we can increase the make the job harder
		output = net(noisy) # network tries to clean it
		loss   = loss_fn(output, clean) # prints out a number saying how wrong that attempt was

		loss.backward() # work out which way to manipulate every weight/figure out the new nudge
		optimizer.step() # take the manipulation/apply the nudge


		if step % 50 == 0:
			print(f"step {step:3d} loss {loss.item():.4f}")
