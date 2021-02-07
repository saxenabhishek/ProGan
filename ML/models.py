import torch
from torch import nn

class Generator(torch.nn.Module):
	'''
	Input: takes in one positional argument 'noiseSize' which is size of random noise vector
	Returns: output of generator
	'''
	def __init__(self, **kwargs):
		super(Generator, self).__init__()
		self.n_features = 512 * 7 * 7 * kwargs['noiseSize'] # CNN O/P * Noise Vec.
		self.n_out = 176 * 3 * 3
		self.noiseSize = kwargs['noise_size']
		self.kSize = kwargs['kernel_size']

		self.linear = nn.Linear(n_features, 1024 * 4 * 4) # not sure about this, cos original DCGAN arch. mein original random noise vector is (100, 1)

		# according to me, before we concatenate w/ the noise and feed to this class, we should pass the image encoding through a Dense layer into a (250, 1) ka output shape. Phir usko we can concatenate with a random noise vector of (250, 1) to get n_features above

		self.conv1 = nn.Sequential(
			nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, stride = 2, kernel_size = kSize, padding = 1, bias = False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True)
		)

		self.conv2 = nn.Sequential(
			nn.ConvTranspose2d(in_channels = 512, out_channels = 256, stride = 2, kernel_size = kSize, padding = 1, bias = False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True)
		)

		self.conv3 = nn.Sequential(
			nn.ConvTranspose2d(in_channels = 256, out_channels = 128, stride = 2, kernel_size = kSize, padding = 1, bias = False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True)
		)

		self.conv4 = nn.Sequential(
			nn.ConvTranspose2d(in_channels = 128, out_channels = 3, stride = 2, kernel_size = kSize, padding = 1, bias = False),
			nn.Tanh()
		)
	
	def forward(self, inp):
		x = self.linear(inp)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		output = self.conv4(x)

		return output


class Discriminator(torch.nn.Module):
	'''
	Input: takes in flattened image of size 176 * 3 * 3
	Returns: probability of image being real
	'''
	def __init__(self):
		super(Discriminator, self).__init__()
		n_features = 176 * 3 * 3 # size of unflattened image passed to discriminator
		n_out = 1

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
			nn.LeakyReLU(0.2, inplace=True)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True)
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 4, stride = 2, padding = 1, bias = False),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.2, inplace=True)
		)

		self.outLayer = nn.Sequential(
			nn.Linear(1024 * 4 * 4, n_out),
			nn.Sigmoid()
		)
	
	def forward(self, inp):
		x = self.conv1(inp)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		output = self.outLayer(x)

		return output