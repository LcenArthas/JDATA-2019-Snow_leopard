#这是第二部
#这个脚本的作用在于随机挑出1000张图片用于训练检测网络

import os
import shutil
import random

train_pic = '../train_data/pic_train/'
pic_list = os.listdir(train_pic)
print(len(pic_list))

det_list = random.sample(pic_list, 5000)
print(len(det_list))

for p in det_list:
    shutil.copyfile(os.path.join(train_pic, p), '../train_data/det_train/pic/' + p)

print('Finish!')