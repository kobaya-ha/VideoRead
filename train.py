#!/usr/bin/env python
#coding: UTF-8
import numpy as np
import six.moves.cPickle as pickle
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Variable
with open('samvideo.sixteen', mode ='r') as f:
		data = pickle.load(f)

print data["samvideo"].shape
data = data["samvideo"]

class MyChain(chainer.Chain):
	def __init__(self):
		super(MyChain, self).__init__(
			conv1=L.ConvolutionND(3,3,10,ksize = (3,3,3), stride= (1,3,3), pad = 0),
			conv2=L.ConvolutionND(3,10,15,ksize=5, stride=2, pad = 0),
			#conv3=L.ConvolutionND(3,15,15,ksize=3, stride=2, pad = 0),
			#conv4=L.ConvolutionND(3,15, 15,ksize=3, stride=2, pad = 0),
			#fc5=L.Linear(544768, 4096),
			#fc6=L.Linear(4096, 4096),
			fc7=L.Linear(145350, 3),
			#conv5=L.Convolution2D(10, 3, ksize=1, stride=1)
		)
		self.train = True

	def __call__(self, x, t):
		x = Variable(x)
		t = Variable(t)

		print x.data.shape
		h = self.conv1(x)
		print h.data.shape
		h = self.conv2(h)
		print h.data.shape
		#h = self.conv3(h)
		#h = self.conv4(h)
		#h = self.fc5(h)
		#h = self.fc6(h)
	
		h = self.fc7(h)
		#h = self.conv5(h)
		print h.data
		
		self.loss = F.softmax_cross_entropy(h,t)
		self.accuracy = F.accuracy(h,t)
		return self.loss

model = MyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

for i in range(20):
	model.zerograds()
	t = np.asarray([0,1,2]).astype(np.int32)
	loss = model(data, t)
	loss.backward()
	optimizer.update()
	print loss.data
