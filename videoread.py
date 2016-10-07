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
したほうがよい．

"""

def makelist(filename):
	assert(os.path.exists(filename)), "not exist file"
	video_path = filename #"/export/data/dataset/UCF-101/Archery/v_Archery_g01_c01.avi"
	framenum = 0
	name, ext = os.path.splitext(filename)
	cap = cv2.VideoCapture(video_path)
	videolist = np.empty((0,240,320,3), dtype=np.float32)

	while(framenum < 16):

		framenum += 1
		ret, frame = cap.read()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		try:
			frame = np.array(frame, dtype=np.float32)
			frame = frame[np.newaxis]
			videolist = np.append(videolist, frame, axis=0)
		except:
			break
	cap.release()

	videolist = videolist.reshape((1,) + videolist.shape)
	print videolist.shape
	return videolist
	
	#data = {name:videolist}
	#with open('videolist.pkl', mode ='wb') as f:
	#	pickle.dump(data,f)

#makelist("samvideo/s2.avi")


def makelist_dir(dirname):
	assert(os.path.exists(dirname)), "not exist directory"
	filelist = os.listdir(dirname)
	all_video = np.empty((0,16,240,320,3), dtype=np.float32)
	print all_video.shape
	print filelist
	for file in filelist:
		print file
		all_video = np.append(all_video, makelist(dirname+"/"+file), axis=0)

	all_video = all_video.transpose(0,4,1,2,3)
	print all_video.shape
	data = {dirname:all_video}

	with open(dirname+'.sixteen', mode ='wb') as f:
		pickle.dump(data,f)


makelist_dir(sys.argv[1])


"""

ret, frame = cap.read()
print frame.shape
videolist = np.empty((0,240,320,3), dtype=np.float32)

#frame:[y][x][rgb]
sam = np.array(frame, dtype=np.float32)
sam = sam[np.newaxis]
#print sam[1][1][1].dtype
dam = np.array(frame, dtype=np.float32)
dam = dam[np.newaxis]
videolist = np.append(videolist, sam, axis=0)
videolist = np.append(videolist, dam, axis=0)
print videolist.shape
print videolist.ndim
print videolist[1][1][1][2]
print videolist[0][1][1][1]

revideo = videolist.reshape((1,) + videolist.shape)
print revideo.shape
"""