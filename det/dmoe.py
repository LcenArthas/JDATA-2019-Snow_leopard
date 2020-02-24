#检测行人的结果保存txt(格式如同MOT challenge)
#另外保存有检测框的图片结果

from mmdet.apis import init_detector, inference_detector, show_result
from PIL import Image
from tqdm import tqdm
import numpy as np
import mmcv
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config_file = './configs/cascade_rcnn_x101_64x4d_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = './work_dirs/mot17_pre_cascade_rcnn_x101_64x4d_fpn_1x/epoch_12.pth'
checkpoint_file = './work_dirs/epoch_12.pth'

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
    # save_path = '/mnt/liucen/JD_Tiger/train_data/show_det/' + pic_name
    # im.save(save_path)

    return bboxes[inds, :]


#存储图片的文件夹地址列表
img_files = '/mnt/liucen/JD_Tiger/train_data/pic_train/'


for pic in tqdm(os.listdir(img_files)):
    img = img_files + pic
    result = det(img, score_thr=0.95)

    if result.shape[0] == 0:                         #检测无目标
        continue
    else:
        img = Image.open(img)

        #对得到的检测位置外围扩充
        width, height = img.size
        x1 = result[0, 0] - 40
        if x1 < 0 :
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
        img = img.crop((x1, y1, x2, y2))
        img.save('/mnt/liucen/JD_Tiger/train_data/clearn_train_big/' + pic)
        ########################

        # img = img.crop((result[0, 0], result[0, 1], result[0, 2], result[0, 3]))
        # img.save('/mnt/liucen/JD_Tiger/train_data/clearn_train/' + pic)

# for f in img_files:
#     os.makedirs(f + 'det/')
#     with open(f + 'det/det.txt', 'w') as txt:
#         for index in tqdm(range(3000)):
#             img = f + 'img1/{:>06d}.jpg'.format(index+1)
#             #检测行人
#             result = det(img, f,score_thr=0.3)
#             #如果图片里无目标
#             if result.shape[0] == 0:
#                 continue
#             else:
#                 for i in range(result.shape[0]):
#                     ll = [str(index+1),',-1', ',',str(result[i, 0]), ',',str(result[i, 1]),',',
#                           str(result[i, 2]-result[i, 0]), ',', str(result[i, 3]-result[i, 1]), ',', str(result[i, 4]), '\n']
#                     txt.writelines(ll)
