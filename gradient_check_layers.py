import numpy as np
from layers.Conv2d import Conv2D
from layers.FC import FullyConnect
from layers.Pooling import MaxPooling, AvgPooling
from layers.SoftmaxLoss import SoftmaxLoss
from layers.ReLU import ReLU

### grad_check

e = 1e-3
x = np.random.rand(1, 28, 28, 3)

a = x + e
b = x - e
c = x

## conv2d
conv1 = Conv2D(a.shape, 12, 3, 1, 1)
conv2 = Conv2D(b.shape, 12, 3, 1, 1)
conv3 = Conv2D(c.shape, 12, 3, 1, 1)

conv2.weights = conv1.weights
conv2.bias = conv1.bias
conv3.weights = conv1.weights
conv3.bias = conv1.bias

conv_out_a = conv1.forward(a)
conv_out_b = conv2.forward(b)
conv_g1 = np.sum(conv_out_a - conv_out_b) / (2 * e)

conv_out_c = conv3.forward(c)
eta = np.ones(conv_out_c.shape)
conv_g2 = np.sum(conv3.gradient(eta))

print ('conv2d gradient check:', conv_g1, conv_g2)


## max_pool2d
pool1 = MaxPooling(a.shape, 3, 2)
pool2 = MaxPooling(b.shape, 3, 2)
pool3 = MaxPooling(c.shape, 3, 2)

pool_out_a = pool1.forward(a)
pool_out_b = pool2.forward(b)
pool_out_c = pool3.forward(c)

pool_g1 = np.sum(pool_out_a - pool_out_b) / (2 * e)

eta = np.ones(pool_out_c.shape)
pool_g2 = np.sum(pool3.gradient(eta))

print ('max_pool2d gradient check:', pool_g1, pool_g2)

## avg_pool2d
pool1 = AvgPooling(a.shape, 2, 2)
pool2 = AvgPooling(b.shape, 2, 2)
pool3 = AvgPooling(c.shape, 2, 2)

pool_out_a = pool1.forward(a)
pool_out_b = pool2.forward(b)
pool_out_c = pool3.forward(c)

pool_g1 = np.sum(pool_out_a - pool_out_b) / (2 * e)

eta = np.ones(pool_out_c.shape)
pool_g2 = np.sum(pool3.gradient(eta))

print ('avg_pool2d gradient check:', pool_g1, pool_g2)

## FC
fc1 = FullyConnect(a.shape, 2)
fc2 = FullyConnect(b.shape, 2)
fc3 = FullyConnect(c.shape, 2)

fc2.weights = fc1.weights
fc2.bias = fc1.bias
fc3.weights = fc1.weights
fc3.bias = fc1.bias

fc_out_a = fc1.forward(a)
fc_out_b = fc2.forward(b)
fc_g1 = np.sum(fc_out_a - fc_out_b) / (2 * e)

fc_out_c = fc3.forward(c)
eta = np.ones(fc_out_c.shape)
fc_g2 = np.sum(fc3.gradient(eta))

print ('fc gradient check:', fc_g1, fc_g2)

## relu
relu1 = ReLU(a.shape)
relu2 = ReLU(b.shape)
relu3 = ReLU(c.shape)

relu_out_a = relu1.forward(a)
relu_out_b = relu2.forward(b)
relu_out_c = relu3.forward(c)

relu_g1 = np.sum(relu_out_a - relu_out_b) / (2 * e)

eta = np.ones(relu_out_c.shape)
relu_g2 = np.sum(relu3.gradient(eta))

print ('relu gradient check:', relu_g1, relu_g2)

## softmaxloss
e = 1e-3
x = np.random.rand(1, 10)

a = x + e
b = x - e
c = x
label = np.array([1, 2])
print (label)

softmax1 = SoftmaxLoss(a.shape)
softmax2 = SoftmaxLoss(b.shape)
softmax3 = SoftmaxLoss(c.shape)

softmax_out_a = softmax1.cal_loss(a, label)
softmax_out_b = softmax2.cal_loss(b, label)
softmax_out_c = softmax3.cal_loss(c, label)

softmax_g1 = np.sum(softmax_out_a - softmax_out_b) / (2 * e)

eta = np.ones(softmax_out_c.shape)
softmax_g2 = np.sum(softmax3.gradient())

print ('softmax gradient check:', softmax_g1, softmax_g2)