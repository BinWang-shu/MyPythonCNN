import numpy as np 
import math



class Conv2D(object):
	"""docstring for ClassName"""
	# def __init__(self, shape, output_channels, ksize=3, stride=1, method='VALID'):
	def __init__(self, shape, output_channels, ksize=3, stride=1, padding=1):
		super(Conv2D, self).__init__()
		self.input_shape = shape
		self.output_channels = output_channels
		self.input_channels = shape[-1]
		self.batchsize = shape[0]
		self.stride = stride
		self.ksize = ksize
		self.padding = padding
		weight_scale = math.sqrt(ksize*ksize*self.input_channels*output_channels/2)
		self.weights = np.random.standard_normal(
			           (ksize, ksize, self.input_channels, self.output_channels)) / weight_scale
		self.bias = np.random.standard_normal(self.output_channels) / weight_scale
		
		# w = (W + 2*pad - kernel) / stride + 1 
		H = int((shape[1] + 2 * padding - ksize) / self.stride + 1)
		W = int((shape[2] + 2 * padding - ksize) / self.stride + 1)
		self.eta = np.zeros((shape[0], H, W, self.output_channels))

		self.w_gradient = np.zeros(self.weights.shape)
		self.b_gradient = np.zeros(self.bias.shape)
		self.output_shape = self.eta.shape

	def forward(self, x):
		col_weights = self.weights.reshape(-1, self.output_channels)
		x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), 
			       (0, 0)), 'constant', constant_values=0)

		self.col_image = []
		conv_out = np.zeros(self.eta.shape)
		for i in range(self.batchsize):
			img_i = x[i][np.newaxis, :]
			self.col_image_i = im2col(img_i, self.ksize, self.stride)
			conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
			self.col_image.append(self.col_image_i)
		self.col_image = np.array(self.col_image)
		return conv_out

	def gradient(self, eta):
		## get the grad of weights and bias
		self.eta = eta
		col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])
		for i in range(self.batchsize):
			self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)

		self.b_gradient += np.sum(col_eta, axis=(0, 1))

		## get the next_dalta
		# padding_eta = padding + (W - eta_W) * (stride + 1) / 2
		padding_eta_h = int(self.padding + (self.input_shape[1] - eta.shape[1]) * (self.stride + 1) / 2)
		padding_eta_w = int(self.padding + (self.input_shape[2] - eta.shape[2]) * (self.stride + 1) / 2)

		pad_delta = np.pad(self.eta, ((0, 0), (padding_eta_h, padding_eta_h),
				               (padding_eta_w, padding_eta_w), (0, 0)), 'constant', constant_values=0)

		flip_weights = np.flipud(np.fliplr(self.weights))
		flip_weights = flip_weights.swapaxes(2, 3) # have the question 
		# print (flip_weights.shape)
		col_flip_weights = flip_weights.reshape([-1, self.input_channels])
		# print (col_flip_weights.shape)
		col_pad_delta = np.array([im2col(pad_delta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
		# print (col_pad_delta.shape)
		next_dalta = np.dot(col_pad_delta, col_flip_weights)
		next_dalta = np.reshape(next_dalta, self.input_shape)

		return next_dalta    

	def backward(self, alpha=0.00001, weight_decay=0.0004):
		self.weights *= (1 - weight_decay)
		self.bias *= (1 - weight_decay)
		self.weights -= alpha * self.w_gradient
		self.bias -= alpha * self.b_gradient

		self.w_gradient = np.zeros(self.weights.shape)
		self.b_gradient = np.zeros(self.bias.shape)


def im2col(image, ksize, stride):
	image_col = []

	for i in range(0, image.shape[1]- ksize + 1, stride):
		for j in range(0, image.shape[2] - ksize + 1, stride):
			col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
			image_col.append(col)

	image_col = np.array(image_col)
	return image_col 

if __name__ == "__main__":
	img = np.ones((1, 32, 32, 3))
	img *= 2
	conv = Conv2D(img.shape, 12, 4, 2, 1)
	next = conv.forward(img)
	print (next.shape)
	next1 = next.copy() + 1

	conv.gradient(next1-next)
	# print (conv.w_gradient)
	# print (conv.b_gradient)
	conv.backward()
