#!/usr/bin/env python
#coding: UTF-8
import numpy as np
from chainer import cuda
xp = np #cuda.cupy
import sys
#import six.moves.cPickle as pickle
import chainer
import chainer.links as L
import chainer.functions as F
import videoread
from chainer import optimizers, Variable, serializers, training, datasets
from chainer.training import extensions
import c3dnet
import argparse

#from sklearn.cross_validation import train_test_split


#data = chainer.cuda.to_gpu(data) #GPU用の￥記述
#label = chainer.cuda.to_gpu(label)

"""
バイナリから呼び出す時の記述 //たぶん書き換えないと動きません
data = np.empty((0,18,240,320,3), dtype=np.float32)
for file in filelist:
	print  file
	with open(file, mode ='r') as f:
		single_data = pickle.load(f)
		data = np.append(data, single_data["data"])
"""


class TestModeEvaluator(extensions.Evaluator):

	def evaluate(self):
		model = self.get_target('main')
		model.train = False
		ret = super(TestModeEvaluator, self).evaluate()
		model.train = True
		return ret



def main(self):
	model = c3dnet.C3D()
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	
	n = 102 #データサイズ
	bs = 10 #ミニバッチサイズ

	for i in range(20):
		sffindx = np.random.permutation(n)
		for j in range(0, n, bs):
			#print cupyindx[j:(j+bs) if (j+bs) < n else n]
			data_b = Variable(self.data[sffindx[j:(j+bs) if (j+bs) < n else n]])
			label_b = Variable(self.label[sffindx[j:(j+bs) if (j+bs) < n else n]])
			model.zerograds()
			loss = model(data_b, label_b)
			loss.backward()
			optimizer.update()
			print loss.data


def main_v111():
	#オプションの追加
	parser = argparse.ArgumentParser(description='Chainer : C3D')
	parser.add_argument('--arch', '-a', default='ADAM',
                        help='Convnet architecture')
	parser.add_argument('--batchsize', '-b', type=int, default=10,
	                    help='Number of images in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=20,
	                    help='Number of sweeps over the dataset to train')
	parser.add_argument('--gpu', '-g', type=int, default=0,
	                    help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--out', '-o', default='result',
	                    help='Directory to output the result')
	parser.add_argument('--resume', '-r', default='resume',
	                    help='Resume the training from snapshot')
	parser.add_argument('--unit', '-u', type=int, default=1000,
	                    help='Number of units')
	parser.add_argument('--input', '-i', default='data/testUCF-10',
	                    help='Directory to input data')

	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('# unit: {}'.format(args.unit))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# epoch: {}'.format(args.epoch))
	print('# input: {}'.format(args.input))
	print('')

	#データセットの読み込み
	v = videoread.VideoRead()

	train, test = v.combine_data_label(args.input)
	print type(train)
	

	#モデルの設定
	model = c3dnet.C3D()
	if args.gpu >= 0:
		chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
		model.to_gpu()  # Copy the model to the GPU

	#Setup an optimizer"
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	#make iterators"
	train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
	test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
	#set up a trainer"
	updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


	val_interval = (10 ), 'iteration'
	log_interval = (10 ), 'iteration'
	#trainer.extend(TestModeEvaluator(val_iter, model, device=args.gpu),
     #              trigger=val_interval)
	trainer.extend(extensions.Evaluator(
		test_iter, model, device=args.gpu))

	trainer.extend(extensions.dump_graph('main/loss'))
	trainer.extend(extensions.snapshot(), trigger=val_interval)
	trainer.extend(extensions.snapshot_object(
	    model, 'model_iter_{.updater.iteration}'), trigger=val_interval)


	serializers.save_npz('my.model', model)

	trainer.extend(extensions.LogReport(trigger=log_interval))
	trainer.extend(extensions.observe_lr(), trigger=log_interval)
	trainer.extend(extensions.PrintReport([
	    'epoch', 'iteration', 'main/loss', 'validation/main/loss',
	    'main/accuracy', 'validation/main/accuracy', 'lr'
	]), trigger=log_interval)
	#Progress barを表示
	trainer.extend(extensions.ProgressBar())#update_interval=10))

	trainer.run()

if __name__ == '__main__':
	main_v111()
