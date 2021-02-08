import torch
from torch import nn
from torchvision import models


class ResNetEncoder(nn.Module):
	def __init__(self, vec_shape):
		super(ResNetEncoder, self).__init__()
		self.model =  models.resnet18(pretrained=True)
		self.model.fc = nn.Linear(self.model.fc.in_features, vec_shape)
	
	def forward(self, image):
		return self.model(image)


class Generator(nn.Module):
	def __init__(self, **kwargs):
		super(Generator, self).__init__()
	
		self.device = kwargs['device']
		self.noisedim = kwargs["noisedim"]
		self.vector_shape = kwargs['vec_shape']
		self.input_shape = self.vector_shape + self.noisedim
		self.im_channels = kwargs["im_channels"]
		self.batch_size = kwargs['batch_size'] 
	
		self.gen = nn.Sequential(
				self.genblock(input_channels=self.input_shape, hidden_size=512, kernel_size=4, stride=1, padding=1),
				self.genblock(input_channels=512, hidden_size=350, kernel_size=4, stride=2, padding=1),
				self.genblock(input_channels=350, hidden_size=250, kernel_size=4, stride=2, padding=1),
				self.genblock(input_channels=250, hidden_size=150, kernel_size=4, stride=2, padding=1),
				self.genblock(input_channels=150, hidden_size=self.im_channels, kernel_size=4, stride=2, padding=1, last_layer=True) ## final layer returning tanh
  )

	def genblock(self, input_channels, hidden_size, kernel_size, stride, padding, last_layer=False):
		if not last_layer:
			genblock = nn.Sequential(
					  nn.ConvTranspose2d(input_channels, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
					  nn.BatchNorm2d(hidden_size),
					  nn.ReLU(True),
					)

		else:
			genblock = nn.Sequential(
						nn.ConvTranspose2d(input_channels, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
						nn.Tanh(),

			)
		
		return genblock


	def geninput(self):
		return self.encodedvec.view(len(self.encodedvec), self.encodedvec.shape[1], 1, 1)


	def concat(self):
		self.inputnoise = self.make_noise()
		encoded = torch.cat([self.feat, self.inputnoise], dim=1)
		return encoded

	def make_noise(self):
		return torch.randn(self.batch_size, self.noisedim, device=self.device)
	
	def forward(self, feat):
		self.feat = feat
		self.encodedvec = self.concat()    

		self.genin = self.geninput()    
		# print('Going in ', self.genin.shape) 
		return self.gen(self.genin)


class Discriminator(torch.nn.Module):
	'''
	Input: takes in flattened image of size 176 * 3 * 3
	Returns: probability of image being real
	'''
	def __init__(self):
		super(Discriminator, self).__init__()
		n_features = 176 * 3 * 3 # size of unflattened image passed to discriminator
		n_out = 1

		self.disc = nn.Sequential(
			self.discGen(inputChannels = 3, outputChannels = 128, first_layer = True),
			self.discGen(inputChannels = 128, outputChannels = 256, first_layer = False),
			self.discGen(inputChannels = 256, outputChannels = 512, first_layer = False)
		)

	def discGen(self, inputChannels, outputChannels, first_layer = True):
		if first_layer:
			return nn.Sequential(
				nn.Conv2d(in_channels = inputChannels, out_channels = outputChannels, kernel_size = 4, stride = 2, padding = 1, bias = False),
				nn.LeakyReLU(0.2, inplace=True)
			)
		else:
			return nn.Sequential(
			nn.Conv2d(in_channels = inputChannels, out_channels = outputChannels, kernel_size = 4, stride = 2, padding = 1, bias = False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True)
		)

	def forward(self, inp):
		return self.disc(inp)

if __name__ == "__main__":
	vec_shape = 500
	batch_size = 1
	device = "cpu"
	resnet = ResNetEncoder(vec_shape)
	resnet = resnet.to(device)
	gen = Generator(device=device, noisedim=250, im_channels=3, batch_size=batch_size, vec_shape=vec_shape)
	gen = gen.to(device)
	for i in range(2):
	 	print(gen(resnet(torch.randn(batch_size, 3, 224, 224, device=device))).shape)
	
	disc = Discriminator()