#!/usr/bin/env python
#coding: UTF-8
import cv2
import os
import sys
import numpy as np
import six.moves.cPickle as pickle

"""
chainerでは配列の形を[id番号, チャネル数，d1, d2, d3]
にする必要がある
im00くらいでしか動かないのでそこで配列のファイルを生成
してください．
読み込み場所：data
保存先：binary_list

"""
class VideoRead:

	def makelist(self, filename):
		assert(os.path.exists(filename)), "not exist file"
		video_path = filename #"/export/data/dataset/UCF-101/Archery/v_Archery_g01_c01.avi"
		framenum = 0
		name, ext = os.path.splitext(filename)
		cap = cv2.VideoCapture(video_path)
		videolist = np.empty((0,240,320,3), dtype=np.float32)

		while(framenum < 90):
			if(framenum % 5 == 0):
				print framenum
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
		print videolist.shape
		return videolist
		


	def makelist_dir(self, dirname):
		assert(os.path.exists(dirname)), "not exist directory"
		filelist = os.listdir(dirname)
		all_video = np.empty((0,18,240,320,3), dtype=np.float32)
		print all_video.shape
		print filelist
		for file in filelist:
			print  file
			all_video = np.append(all_video, self.makelist(dirname+"/"+file), axis=0)

		all_video = all_video.transpose(0,4,1,2,3)
		print all_video.shape
		data = {"data":all_video}# ,"label": t}

		with open('binary_list/'+dirname+'.pkl', mode ='wb') as f:
			assert(os.path.exists('binary_list')), "not exist directory"
			pickle.dump(data,f)


v = VideoRead()

v.makelist_dir(sys.argv[1])