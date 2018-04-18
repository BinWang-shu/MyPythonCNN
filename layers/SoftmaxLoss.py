import numpy as np 


class SoftmaxLoss(object):
	def __init__(self, shape):
		self.softmax = np.zeros(shape)
		self.eta = np.zeros(shape)
		self.batchsize = shape[0]

	def cal_loss(self, prediction, label):
		self.label = label
		self.prediction = prediction
		self.predict(prediction)
		self.loss = 0
		for i in range(self.batchsize):
			self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, label[i]]

		return self.loss

	def predict(self, prediction):
		exp_prediction = np.zeros(prediction.shape)
		self.softmax = np.zeros(prediction.shape)
		for i in range(self.batchsize):
			prediction[i, :] -= np.max(prediction[i, :])
			exp_prediction[i] = np.exp(prediction[i])
			self.softmax[i] = exp_prediction[i] / np.sum(exp_prediction[i])

		return self.softmax

	def gradient(self):
		self.next_delta = self.softmax.copy()
		for i in range(self.batchsize):
			self.next_delta[i, self.label[i]] -= 1
		return self.next_delta


if __name__ == "__main__":
    img = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    label = np.array([[1], [2]])
    sf = SoftmaxLoss(img.shape)
    loss = sf.cal_loss(img, label)

    sf.gradient()

    print (loss)
    print (sf.gradient())