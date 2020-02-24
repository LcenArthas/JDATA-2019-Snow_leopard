#这是第一步
#把最原始的视频数据变为图片
#保存的格式是 tigerID_vidioID_picID.jpg 保存在 pic_train (所有id都是从1开始)

import os
import cv2
import matplotlib.pyplot as plt

video_file = './data/train/'                                       #存放原始视频的位置
tiger_list = os.listdir(video_file)

#含有噪声过多的视频，直接删除
#id:28-> e74c0bba5242.AVI, d3214881ed19.AVI
#id:27-> 6614fb92e625.AVI, 85b899cf545c.AVI
#id:25-> cca5279df089.AVI, 172852d14363.AVI, 7b04558a05d0.AVI
#id:24-> ffd25d87412a.AVI, d2d7a1257bfd.AVI, 776ed0a90022.AVI, 8e0b8006b62f.AVI
del_list = ['e74c0bba5242.AVI', 'd3214881ed19.AVI',
            '6614fb92e625.AVI', '85b899cf545c.AVI',
            'cca5279df089.AVI', '172852d14363.AVI', '7b04558a05d0.AVI',
            'ffd25d87412a.AVI', 'd2d7a1257bfd.AVI', '776ed0a90022.AVI', '8e0b8006b62f.AVI']


for tiger_id in tiger_list:
    T_id = tiger_id.split('_')[-1]                                       #老虎的ID
    tiger_video_file = os.path.join(video_file, tiger_id)
    video_list = os.listdir(tiger_video_file)
    print('The tiger ID is {}.'.format(T_id))
    for V_id, video in enumerate(video_list):
        if video in del_list:
            continue
        V_id += 1
        # print(V_id)
        cap = cv2.VideoCapture(os.path.join(tiger_video_file, video))
        print(cap.get(7))                                                #total frame number
        index = 1
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            if ret == True:
                frame = frame[:950, :]                                  #删除图片下方的噪声带
                save_path = './data/pic_train/{0}_{1}_{2}.jpg'.format(str(T_id), str(V_id), str(index))
                # save_path = "test_dataset/capture_pic/" + video.split('.')[0] + "/img1/{:>06d}.jpg".format(index)
                # print(save_path)
                index += 1
                cv2.imwrite(save_path, frame)
        cap.release()