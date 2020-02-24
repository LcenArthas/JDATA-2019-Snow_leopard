#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
import os
#from dataset import *

#from skimage import io
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd

from tqdm import tqdm
from conf import settings
import numpy as np
from utils import get_network, get_test_dataloader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='vgg19', help='net type')                     #改
    parser.add_argument('-weights', type=str, default='./cls/checkpoints/vgg19/vgg19-55-best.pth', help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=16, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=2, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-num_classes', type=int, default=35, help='class_num')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained model')
    parser.add_argument('-bool_pretrained', type=str, default=False, help='vgg_pretrained model')  # 除了resnset是否用于训练模型预训练
    args = parser.parse_args()

    net = get_network(args)

    state_dictBA = torch.load(args.weights)
    net.load_state_dict(state_dictBA)

    net.cuda()
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    video = []
    class_id = []
    for video_file in tqdm(os.listdir('./data/new_test_pic/')):   #改
        test_path = './data/new_test_pic/' + video_file + '/'     #改
        video.append(video_file)
        if len(os.listdir(test_path)) == 0:
            id = 1
            class_id.append('snow_leopard_' + str(id + 1))
        else:
            cifar100_test_loader = get_test_dataloader(
                test_path,
                num_workers=args.w,
                batch_size=args.b,
            )

            list_id = []
            for n_iter, (image1, image2, image3, image4 , image5, image6) in enumerate(cifar100_test_loader):
                image1 = image1.cuda()
                image2 = image2.cuda()
                image3 = image3.cuda()
                image4 = image4.cuda()
                image5 = image5.cuda()
                image6 = image6.cuda()

                output1 = net(image1)
                output2 = net(image2)
                output3 = net(image3)
                output4 = net(image4)
                output5 = net(image5)
                output6 = net(image6)
                # output = torch.add(torch.add(output1, output2), torch.add(output3, output4),torch.add(output5, output6)) / 6
                output = (output1+output2+output3+output4+output5+output6)/6
                # output = torch.add(output1, output2) / 2

                _, pred = output.topk(1, 1, largest=True, sorted=True)

                # list_id.append(list(np.array(pred.cpu())[0]))
                if np.array(pred.cpu()).shape[0] == 1:
                    list_id.extend(np.array(pred.cpu()).tolist())
                else:
                    list_id.extend(np.squeeze(np.array(pred.cpu())).tolist())
            id = max(list_id, key=list_id.count)

            class_id.append('snow_leopard_' + str(id + 1))

    df = pd.DataFrame({'video_name': video, 'class': class_id})
    df.to_csv('submission.csv', index=False)
