#检测行人的结果保存txt(格式如同MOT challenge)
#另外保存有检测框的图片结果

from mmdet.apis import init_detector, inference_detector, show_result
from PIL import Image
from tqdm import tqdm
import numpy as np
import mmcv
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config_file = './det/configs/cascade_rcnn_x101_64x4d_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = './work_dirs/mot17_pre_cascade_rcnn_x101_64x4d_fpn_1x/epoch_12.pth'
checkpoint_file = './det/work_dirs/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

print("MODEL FINISHED!")

#检测,返回结果，并且保存图片
def det(img, score_thr=0.5):
    result = inference_detector(model, img)
    bboxes = result[0]                                                      #result[0]表示非0的检测结果
    inds = np.where(bboxes[:, -1] > score_thr)[0]

    model.CLASSES = ['1']                                                   #这里的更改是为了不报错，对于只有一种类型的情况
    #保存图片
    # img1 = show_result(img, result, model.CLASSES, show=False, score_thr=score_thr)
    # im = Image.fromarray(img1)
    # pic_name = img.split('/')[-1]
    # save_path = '/mnt/liucen/JD_Tiger/test_data/show_det/' + pic_name
    # im.save(save_path)

    return bboxes[inds, :]


#存储图片的文件夹地址列表
video_files = './data/test_pic/'
os.makedirs('./data/new_test_pic/')

img_file = os.listdir(video_files)

for i, video_id in enumerate(img_file):
    print('No.{} Name:{}.'.format(i, video_id))
    #新建文件夹
    os.makedirs('./data/new_test_pic/' + video_id)
    for pic in tqdm(os.listdir(video_files + video_id + '/')):
        img = video_files + video_id + '/' + pic
        result = det(img, score_thr=0.95)

        if result.shape[0] == 0:                         #检测无目标
            continue
        else:
            img = Image.open(img)

            # 对得到的检测位置外围扩充
            width, height = img.size
            x1 = result[0, 0] - 40
            if x1 < 0:
                x1 = 0
            y1 = result[0, 1] - 40
            if y1 < 0:
                y1 = 0
            x2 = result[0, 2] + 40
            if x2 > width:
                x2 = width
            y2 = result[0, 3] + 40
            if y2 > height:
                y2 = height
            # img = img.crop((x1, y1, x2, y2))

            ########################

            img = img.crop((result[0, 0], result[0, 1], result[0, 2], result[0, 3]))
            img.save('./data/new_test_pic/' + video_id + '/' + pic)