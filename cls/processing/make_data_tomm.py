#这是第三步
#这个脚本的作用是把标注的检测数据集转换为可以用mmdetection跑得模型


# -*- coding=utf-8 -*-
#!/usr/bin/python

import sys
import os
import shutil
from tqdm import tqdm
import numpy as np
import json
from PIL import Image

root_path = './data/det_train/'

# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {'1': 1}

#把每个txt文件进行处理，返回一个字典，字典的key就是帧，值是个list，其中包括id，坐标，类别
def json2dir(label_json):

    f = open(os.path.join(root_path + 'outputs/', label_json), 'r')
    f_json = json.load(f)
    dir = {}

    if len(f_json['outputs']) == 0 or len(f_json['outputs']['object']) == 0:
        dir['flag'] = 0                     #一个标志位，0表示没有标注
    else:
        dir['flag'] = 1
        dir['bbox'] = f_json["outputs"]["object"][0]["bndbox"]       #也是个字典
        dir['shape'] = f_json["size"]                                #也是个字典
    return dir

#返回图片的长宽，输入的是label_txt名字和frame
def get_size(pic):
    im = Image.open(pic)
    return im.size[0], im.size[1]

#传进来是标签列表名字，标签根目录，要保存的文件名字
def convert(label_list, json_file):
    '''
    :param label_list: 需要转换的txt列表
    :param json_file: 导出json文件的路径
    :return: None
    '''

    # 标注基本结构
    json_dict = {"images":[],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}
    imageID = 1
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for label_json in label_list:                                        #循环标签列表
        pic_name = label_json.split('.')[0] + '.jpg'

        json_dir = json2dir(label_json)
        if json_dir['flag'] == 0:                                       #无目标
            continue

        else:
            ## The filename must be a number
            image_id = imageID                # 图片ID
            imageID += 1

            # 复制图片
            picfile = root_path + 'pic/' + pic_name
            shutil.copyfile(picfile, './det/data/coco/train2017/' + pic_name)

            # 图片的基本信息
            width, height = get_size(picfile)
            image = {'file_name': pic_name,
                     'height': height,
                     'width': width,
                     'id':image_id}
            json_dict['images'].append(image)

            # 处理每个标注的检测框

            #设置类别
            category_id = 1

            #设置坐标
            xmin = json_dir['bbox']['xmin']
            ymin = json_dir['bbox']['ymin']
            w = json_dir['bbox']['xmax'] - json_dir['bbox']['xmin']
            h = json_dir['bbox']['ymax'] - json_dir['bbox']['ymin']

            annotation = dict()
            annotation['area'] = w * h
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, w, h]
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [[xmin, ymin, xmin, ymin]]

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    # 写入类别ID字典
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    # 导出到json
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

def JDTIGER_TO_COCO():
    #设置转换的label文件
    label_list = os.listdir(root_path + 'outputs/')
    json_file = './det/data/coco/annotations/instances_train2017.json'
    convert(label_list, json_file)                                  #转换

if __name__ == '__main__':
    os.mkdir('./det/data/')
    os.mkdir('./det/data/coco')
    os.mkdir('./det/data/coco/annotations/')
    os.mkdir('./det/data/coco/train2017/')
    JDTIGER_TO_COCO()