 # MyPythonCNN
Writing some cnn layers ans the computation graph in python

This repository is mainly based on https://github.com/wuziheng/CNN-Numpy

There are some problems in https://github.com/wuziheng/CNN-Numpy. When the stride != 0, the forward and backward of the convolution will error. When the kernel size != the stride, the error will happen on pooling. So  have improved the algorithm of the convolution and pool. 

convolution:<br>
	forward(x):<br>
	x * conv = out<br>
	im2col(x) dot col(conv) ==> out.<br>
	backward(eta):<br>
	imcol(eta) dot col(conv.T) ==> input.eta<br>
	if stride != 1, I expand the eta based on the stride and backwards it like the stride == 1.<br>

MaxPooling:<br>
	forward(x):<br>
	x * pool = out<br>
	save the index of the max ==> self.index<br>
	backward(eta):<br>
	assgin each pixel in eta to the related place of input based on the index.<br>
	if the kernelsize > stride, some pixels in input will be used over once.So in the backward, I use add to assign.<br>

AvgPooling:<br>
	like MaxPooling.<br>



