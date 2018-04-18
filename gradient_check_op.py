import tensor.Operator as op
import tensor.Variable as var
import tensor.Activation as activation
import numpy as np 


### grad_check

e = 1e-3
a = var.Variable((1, 28, 28, 3), 'a')
b = var.Variable((1, 28, 28, 3), 'b')
c = var.Variable((1, 28, 28, 3), 'c')

b.data = a.data.copy()
c.data = a.data.copy()
a.data += e
b.data -= e

## conv2d
conv1_out = op.Conv2D(a, (3, 3, 3, 3), name='conv1', stride=1, padding=1).output_variables
conv2_out = op.Conv2D(b, (3, 3, 3, 3), name='conv2', stride=1, padding=1).output_variables
conv3_out = op.Conv2D(c, (3, 3, 3, 3), name='conv3', stride=1, padding=1).output_variables

conv1 = var.GLOBAL_VARIABLE_SCOPE['conv1']
conv2 = var.GLOBAL_VARIABLE_SCOPE['conv2']
conv3 = var.GLOBAL_VARIABLE_SCOPE['conv3']
var.GLOBAL_VARIABLE_SCOPE['conv1'].weights.data = var.GLOBAL_VARIABLE_SCOPE['conv2'].weights.data
var.GLOBAL_VARIABLE_SCOPE['conv1'].bias.data = var.GLOBAL_VARIABLE_SCOPE['conv2'].bias.data
var.GLOBAL_VARIABLE_SCOPE['conv3'].weights.data = var.GLOBAL_VARIABLE_SCOPE['conv2'].weights.data
var.GLOBAL_VARIABLE_SCOPE['conv3'].bias.data = var.GLOBAL_VARIABLE_SCOPE['conv2'].bias.data

conv1_out.eval()
conv1_out.diff = (np.ones(conv1_out.diff.shape))
conv2_out.eval()
conv2_out.diff = (np.ones(conv1_out.diff.shape))
conv3_out.eval()
conv3_out.diff = (np.ones(conv3_out.diff.shape))

c.diff_eval()
g1 = np.sum(conv1_out.data-conv2_out.data)/(2*e)
g2 = np.sum(c.diff) 
print (g1, g2)

## max_pool
# pool1_out = op.MaxPooling(a, 3, 2, 'pool1').output_variables
# pool2_out = op.MaxPooling(b, 3, 2, 'pool2').output_variables
# pool3_out = op.MaxPooling(c, 3, 2, 'pool3').output_variables

# pool1_out.eval()
# pool1_out.diff = (np.ones(pool1_out.diff.shape))
# pool2_out.eval()
# pool2_out.diff = (np.ones(pool1_out.diff.shape))
# pool3_out.eval()
# pool3_out.diff = (np.ones(pool3_out.diff.shape))

# c.diff_eval()
# g1 = np.sum(pool1_out.data-pool2_out.data)/(2*e)
# g2 = np.sum(c.diff) 
# print (g1, g2)


# ## avg_pool
# pool1_out = op.AvgPooling(a, 3, 2, 'pool1').output_variables
# pool2_out = op.AvgPooling(b, 3, 2, 'pool2').output_variables
# pool3_out = op.AvgPooling(c, 3, 2, 'pool3').output_variables

# pool1_out.eval()
# pool1_out.diff = (np.ones(pool1_out.diff.shape))
# pool2_out.eval()
# pool2_out.diff = (np.ones(pool1_out.diff.shape))
# pool3_out.eval()
# pool3_out.diff = (np.ones(pool3_out.diff.shape))

# c.diff_eval()
# g1 = np.sum(pool1_out.data-pool2_out.data)/(2*e)
# g2 = np.sum(c.diff) 
# print (g1, g2)

# ## FC
# fc1_out = op.FullyConnect(a, name='fc1', output_num=10).output_variables
# fc2_out = op.FullyConnect(b, name='fc2', output_num=10).output_variables
# fc3_out = op.FullyConnect(c, name='fc3', output_num=10).output_variables

# # fc1 = var.GLOBAL_VARIABLE_SCOPE['fc1']
# # fc2 = var.GLOBAL_VARIABLE_SCOPE['fc2']
# # fc3 = var.GLOBAL_VARIABLE_SCOPE['fc3']
# var.GLOBAL_VARIABLE_SCOPE['fc1'].weights.data = var.GLOBAL_VARIABLE_SCOPE['fc2'].weights.data
# var.GLOBAL_VARIABLE_SCOPE['fc1'].bias.data = var.GLOBAL_VARIABLE_SCOPE['fc2'].bias.data
# var.GLOBAL_VARIABLE_SCOPE['fc3'].weights.data = var.GLOBAL_VARIABLE_SCOPE['fc2'].weights.data
# var.GLOBAL_VARIABLE_SCOPE['fc3'].bias.data = var.GLOBAL_VARIABLE_SCOPE['fc2'].bias.data

# fc1_out.eval()
# fc1_out.diff = (np.ones(fc1_out.diff.shape))
# fc2_out.eval()
# fc2_out.diff = (np.ones(fc1_out.diff.shape))
# fc3_out.eval()
# fc3_out.diff = (np.ones(fc3_out.diff.shape))

# c.diff_eval()
# g1 = np.sum(fc1_out.data-fc2_out.data)/(2*e)
# g2 = np.sum(c.diff) 
# print (g1, g2)