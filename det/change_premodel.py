#这个脚本目的在于
# mmdetection中configs文件一般默认是使用imagenet预训练的backbone权重参数，
# 但是使用coco预训练一般来说会使模型收敛更快，效果更好，是比赛提分的一个小trick！
# 1.修改coco预训练模型的类别不一致问题

import torch
num_classes = 2
model_coco = torch.load("./det/pre_model/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth")    #这是改变cacasde rcnn
# model_coco = torch.load("pre_model/htc_x101_64x4d_fpn_20e_20190408-497f2561.pth")    #这是改变cacasde mask

# weight
model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][
                                                        :num_classes, :]
model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][
                                                        :num_classes, :]
model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][
                                                        :num_classes, :]
# bias
model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][:num_classes]
model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][:num_classes]
model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][:num_classes]
# save new model
torch.save(model_coco, "./det/pre_model/coco_pretrained_weights_classes_%d.pth" % num_classes)



# 3.mmdet中只使用mask_rcnn来检测
# 首先在model的dict中删去mask_roi_extractor和mask_head字段及其附属内容，
# 接着在train_cfg的dict中删除所有的mask_size=28，
# 最后在data中把所有的with_mask=True改为with_mask=False即可
