#这是第四步
#用于把清洗之后的数据分出验证集

#同时对清洗出来的数据做EDA

#============================EDA===========================
# import os
# import random
# import shutil
# from tqdm import tqdm
# from collections import defaultdict
#
# clearn_path = '../train_data/clearn_train/'
#
# all_pic_list = os.listdir(clearn_path)
# data_len = len(all_pic_list)
# print('一共有{}张图片'.format(data_len))
#
# tiger_dic = defaultdict(list)
# video_dic = defaultdict(list)
# for i in tqdm(all_pic_list):
#     tiger_id, video_id, frame_id = i.split('_')[0], i.split('_')[1], i.split('_')[2].split('.')[0]
#     video_dic[video_id].append(frame_id)
#     tiger_dic[tiger_id].append(video_id)
#
# print('分析老虎的id和video的关系：')
# for tiger_id, video_list in tiger_dic.items():
#     print('老虎的id：{}, 有{}段视频.'.format(tiger_id, len(set(video_list))))



import os
import random
import shutil
from tqdm import tqdm

clearn_path = '../train_data/clearn_train/'

all_pic_list = os.listdir(clearn_path)
data_len = len(all_pic_list)

val_num = int(data_len * 0.05)           #划分0.1来验证

val_list = random.sample(all_pic_list, val_num)
train_list = list(set(all_pic_list).difference(set(val_list)))

for i in tqdm(train_list):
    shutil.copyfile(clearn_path + i, '../train_data/new_train/' + i)

for i in tqdm(val_list):
    shutil.copyfile(clearn_path + i, '../train_data/new_val/' + i)
