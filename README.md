# JDATA-2019-Snow_leopard

:trophy: JDATA2019 雪豹识别挑战赛冠军方案

:sparkles:赛题链接：[JDATA2019 京东雪豹识别挑战赛](https://jdata.jd.com/html/detail.html?id=9)

这份代码主要是基于PyTorch框架实现，检测部分Cascade_rcnn(resnext101+fpn); 识别部分用的是 Vgg19 分类网络。

:running: 准备工作
-----

## :one: 下载代码

```
https://github.com/LcenArthas/JDATA-2019-Snow_leopard.git
```

## :two: 配置环境

 - Ubantu16.04

 - Python 3.6
 
 其余环境请参考 [mmdetection](https://github.com/open-mmlab/mmdetection)
 
 ------------------------------------------------------------
 
 :sparkles: 训练部分
--------

## :one: 准备数据
:small_orange_diamond: 训练集放入  `./data/train/`, 测试集放入  `./data/test/`, 自助标注的图片和坐标放入  `./data/det_train/`

- [自助标注检测文件下载](https://pan.baidu.com/s/1XHUkFgRvyhmnyf8p101v2Q) 

:small_orange_diamond: 将视频数据分割成图片：

```
cd ./data/
mkdir pic_train/
python cls/processing/vidio_to_pic.py
```

:small_orange_diamond: 将标注的数据集转换成Coco格式：

```
python cls/processing/make_data_tomm.py
```

:small_orange_diamond: 设置预训练模型：
 将COCO预训练模型放入  `./det/pre_model/`中
 - [coco预训练模型](https://pan.baidu.com/s/1XHUkFgRvyhmnyf8p101v2Q) 

```
python det/change_premodel.py
```

## :two: 开始训练

```
python train.py
```

----------------------------------------------------
