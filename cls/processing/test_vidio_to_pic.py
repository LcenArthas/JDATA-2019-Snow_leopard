#使test文件视频分成图片帧，并另存为

import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

video_file = './data/test/'                                      #存放原始视频的位置,改
video_list = os.listdir(video_file)

os.makedirs('./data/test_pic/')                                   #新建文件夹

for video in tqdm(video_list):
    # print(video)
    filename = './data/test_pic/' + video
    os.makedirs(filename)                                              #新建文件夹
    cap = cv2.VideoCapture(os.path.join(video_file, video))
    # print(cap.get(7))                                                  #total frame number
    index = 1
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        if ret == True:
            frame = frame[:950, :]                                    #删除图片下方的噪声带
            save_path = './data/test_pic/{0}/{1}.jpg'.format(str(video), str(index))
            index += 1
            cv2.imwrite(save_path, frame)
    cap.release()