#!/usr/bin/env python
#coding: UTF-8
import numpy as np
#from chainer import cuda
xp = np #cuda.cupy
import sys
#import six.moves.cPickle as pickle
import chainer
import chainer.links as L
import chainer.functions as F
import videoread
from chainer import optimizers, Variable, serializers, training
from chainer.training import extensions


#from sklearn.cross_validation import train_test_split

v = videoread.VideoRead()

data, label = v.makelist_all_class(sys.argv[1])
#data = chainer.cuda.to_gpu(data) #GPU用の￥記述
#label = chainer.cuda.to_gpu(label)

"""
バイナリから呼び出す時の記述
data = np.empty((0,18,240,320,3), dtype=np.float32)
for file in filelist:
	print  file
	with open(file, mode ='r') as f:
		single_data = pickle.load(f)
		data = np.append(data, single_data["data"])
"""



class MyChain(chainer.Chain):
	def __init__(self):
		super(MyChain, self).__init__(
			conv1=L.ConvolutionND(3,3,10,ksize = (3,3,3), stride= (1,3,3), pad = 0),
			conv2=L.ConvolutionND(3,10,15,ksize=5, stride=(1,2,2), pad = 0),
			conv3=L.ConvolutionND(3,15,15,ksize=3, stride=2, pad = 0),
			#conv4=L.ConvolutionND(3,15, 15,ksize=3, stride=2, pad = 0),
			fc5=L.Linear(33750, 4096),
			fc6=L.Linear(4096, 4096),
			fc7=L.Linear(4096, 102),
		)
		self.train = True

	def __call__(self, x, t):
		#x = Variable(x)
		#t = Variable(t)

		print x.data.shape
		h = self.conv1(x)
		print h.data.shape
		h = self.conv2(h)
		print h.data.shape
		h = self.conv3(h)
		print h.data.shape
		#h = self.conv4(h)
		#print h.data.shape
		h = self.fc5(h)
		print h.data.shape
		h = self.fc6(h)
		print h.data.shape
		h = self.fc7(h)
		print h.data.shape
		#print h.data
		
		self.loss = F.softmax_cross_entropy(h,t)
		self.accuracy = F.accuracy(h,t)
		return self.loss


model = MyChain()

#cuda.get_device(0).use()
#model.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(model)
#x_train, x_test, y_train, y_test = train_test_split(data, label, test_size =0.1)



n = 102 #データサイズ
bs = 10 #ミニバッチサイズ

for i in range(20):
	sffindx = np.random.permutation(n)

	#cupyindx = xp.array(sffindx, xp.int32)
	#cupyindx = cupyindx.astype(xp.int32)

	#print type(cupyindx[0])
	for j in range(0, n, bs):
		#print cupyindx[j:(j+bs) if (j+bs) < n else n]
		data_b = Variable(data[sffindx[j:(j+bs) if (j+bs) < n else n]])
		label_b = Variable(label[sffindx[j:(j+bs) if (j+bs) < n else n]])
		model.zerograds()
		loss = model(data_b, label_b)
		loss.backward()
		optimizer.update()
		print loss.data
"""

train_iter = chainer.iterators.MultiprocessIterator(data, bs)

#set up a trainer
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (10, 'epoch'), "result")


trainer.extend(extensions.ProgressBar(update_interval=10))
serializers.save_npz('my.model', model)
"""
