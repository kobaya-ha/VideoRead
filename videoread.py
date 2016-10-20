#!/usr/bin/env python
#coding: UTF-8
import cv2
import os
import sys
import numpy as np
#import six.moves.cPickle as pickle

"""
chainerでは配列の形を[id番号, チャネル数，d1, d2, d3]
にする必要がある
im00くらいでしか動かないのでそこで配列のファイルを生成
してください．→gpマシンで動くようになりました．
バイナリに書き込まなくても問題ありません．
読み込み場所：data
保存先：binary_list

"""
class VideoRead:

	def makelist(self, filename): #1つの動画に対する処理
		assert(os.path.exists(filename)), "not exist file"
		video_path = filename #"/export/data/dataset/UCF-101/Archery/v_Archery_g01_c01.avi"
		framenum = 0
		name, ext = os.path.splitext(filename)
		cap = cv2.VideoCapture(video_path)
		videolist = np.empty((0,240,320,3), dtype=np.float32)

		while(framenum < 90):
			if(framenum % 5 == 0):
				ret, frame = cap.read()
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				try:
					frame = np.array(frame, dtype=np.float32)
					frame = frame[np.newaxis]
					videolist = np.append(videolist, frame, axis=0)
				except:
					break
			
			framenum += 1
		cap.release()

		videolist = videolist.reshape((1,) + videolist.shape)

		return videolist


	def makelist_dir(self, dirname, num_label): #1つのカテゴリに対する処理
		assert(os.path.exists(dirname)), "not exist directory"
		filelist = os.listdir(dirname)
		all_video = np.empty((0,18,240,320,3), dtype=np.float32)
		print filelist
		for file in filelist:
			all_video = np.append(all_video, self.makelist(dirname+"/"+file), axis=0)

		all_video = all_video.transpose(0,4,1,2,3)
		
		#------バイナリに書き込む用の記述--------------------------------
		#with open('binary_list/'+dirname+'.pkl', mode ='wb') as f:
		#	assert(os.path.exists('binary_list')), "not exist directory"
		#	pickle.dump(data,f)
		#----------------------------------------------------------------
		label_list = np.ones(len(filelist), dtype=np.int32) * num_label
		
		return all_video, label_list


	
	def makelist_all_class(self, path): #各カテゴリごとにデータを生成
		num_label = 1 #ラベル番号
		assert(os.path.exists(path)), "not exist directory"
		dirlist = os.listdir(path)
		data = np.empty((0,3,18,240,320), dtype=np.float32)
		labels = np.empty((0), dtype=np.int32)
		for dir in dirlist:
			print dir
			datum, label = self.makelist_dir(os.path.join(path, dir), num_label)
			data = np.append(data, datum, axis=0)
			labels = np.append(labels, label, axis=0)
			num_label += 1

		return data, labels

#dataの中身は辞書型でdataとlabelが付いている？？
v = VideoRead()
d, l = v.makelist_all_class(sys.argv[1])


print d.shape
print l