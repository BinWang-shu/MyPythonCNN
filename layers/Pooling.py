import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt


class MaxPooling(object):
	def __init__(self, shape, ksize=2, stride=2):
		self.input_shape = shape
		self.ksize = ksize
		self.stride = stride
		self.batchsize = shape[0]
		self.output_channels = shape[-1]
		
		self.output_H = int((shape[1] - self.ksize) / self.stride + 1)
		self.output_W = int((shape[2] - self.ksize) / self.stride + 1)
		self.output_shape = [self.batchsize, self.output_H, self.output_W,
						self.output_channels]
		self.index = np.zeros(self.output_shape)

	def forward(self, x):
		_out = np.zeros(self.output_shape)
		for b in range(self.batchsize):
			for c in range(self.output_channels):
				for i in range(self.output_H):
					for j in range(self.output_W):
						_out[b, i, j, c] = np.max(x[b, (i * self.stride):(i * self.stride + self.ksize),
											 (j * self.stride):(j * self.stride + self.ksize), c])
						index = np.argmax(x[b, (i * self.stride):(i * self.stride + self.ksize),
											 (j * self.stride):(j * self.stride + self.ksize), c])
						self.index[b, i, j, c] = index
		return _out

	def gradient(self, eta):
		_in = np.zeros(self.input_shape)
		for b in range(self.batchsize):
			for c in range(self.output_channels):
				for i in range(self.output_H):
					for j in range(self.output_W):
						k = int(self.index[b, i, j, c] / self.ksize)
						l = int(self.index[b, i, j, c] % self.ksize)
						_in[b, i * self.stride + k, j * self.stride + l, c] += eta[b, i, j, c]

		next_delta = _in
		return next_delta


class AvgPooling(object):
	def __init__(self, shape, ksize=2, stride=2):
		self.input_shape = shape
		self.ksize = ksize
		self.stride = stride
		self.batchsize = shape[0]
		self.output_channels = shape[-1]
		
		self.output_H = int((shape[1] - self.ksize) / self.stride + 1)
		self.output_W = int((shape[2] - self.ksize) / self.stride + 1)
		self.output_shape = [self.batchsize, self.output_H, self.output_W,
						self.output_channels]

	def forward(self, x):
		_out = np.zeros(self.output_shape)
		for b in range(self.batchsize):
			for c in range(self.output_channels):
				for i in range(self.output_H):
					for j in range(self.output_W):
						_out[b, i, j, c] = np.mean(x[b, (i * self.stride):(i * self.stride + self.ksize),
											 (j * self.stride):(j * self.stride + self.ksize), c])

		return _out

	def gradient(self, eta):
		_in = np.zeros(self.input_shape)
		for b in range(self.batchsize):
			for c in range(self.output_channels):
				for i in range(self.output_H):
					for j in range(self.output_W):
						_in[b, (i * self.stride) : (i * self.stride + self.ksize), 
							   (j * self.stride) : (j * self.stride + self.ksize), c] += \
								   	eta[b, i, j, c] / (self.ksize * self.ksize)
		next_delta = _in
		return next_delta



if __name__ == "__main__":
	img = Image.open('test.jpg')
	img = np.array(img)[np.newaxis,:]
	# img = img.transpose((0, 2, 3, 1))
	print (img.shape)
	# img2 = Image.open('test.jpg')
	# img = np.array([img,img2]).reshape([2, img.shape[0], img.shape[1], img.shape[2]])

	pool = MaxPooling(img.shape, 2, 2)
	img1 = pool.forward(img)
	img2 = pool.gradient(img1)
	print (img[0,:,:,1])
	print (img1[0,:,:,1])
	print (img2[0,:,:,1])

	plt.imshow(img1[0])
	plt.show()