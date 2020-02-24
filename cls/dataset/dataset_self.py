#自定义的数据读取

import os
import imageio
from conf import settings
from PIL import Image
from imgaug import  augmenters as iaa
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
# import jpeg4py as jpeg
import cv2
import sys

#TTA的过程中的数据增强
def tta_amaugimg(image):
    seq = iaa.Sequential(
        iaa.Grayscale(alpha=1)
    )
    image = seq.augment_image(image)
    return image

def amaugimg(image):
    #数据增强
    # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    seq = iaa.Sequential(
        [
            iaa.SomeOf((0,4),
                       [
                           iaa.AdditiveGaussianNoise(scale=(10, 90)),
                           iaa.Add((-10, 10), per_channel=0.5),
                           iaa.Grayscale(alpha=(0.5, 1.0)),
                           iaa.Affine(rotate=(-10,10)),
                           iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                           iaa.AverageBlur(k=(4, 10))
                       ],
                       random_state=True
            )
        ]
    )
    image = seq.augment_image(image)

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return image

class Dataset(data.Dataset):
    def __init__(self, root, phase='train'):
        self.phase = phase
        self.input_shape = (3, settings.IMAGE_SIZE, settings.IMAGE_SIZE)
        self.phase = phase

        imgs = []
        if phase != 'test':                                               #train & val
            for pic in os.listdir(root):
                imgs.append(pic)

            imgs = [os.path.join(root, img) for img in imgs]
            self.imgs = np.random.permutation(imgs)

        else:
            self.imgs = [os.path.join(root, img) for img in os.listdir(root)]

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomRotation(25),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        if self.phase == 'train':
            splits = sample.split('/')[-1]
            label = int(splits.split('_')[0])-1
            img_path = sample
            data = cv2.imread(img_path)
            data = cv2.resize(data, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))
            data = amaugimg(data)
            data = self.transforms(data)
            return data, label
        elif self.phase == 'val':
            splits = sample.split('/')[-1]
            label = int(splits.split('_')[0])-1
            img_path = sample
            data = cv2.imread(img_path)
            data = cv2.resize(data, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))
            # data = Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
            data = self.transforms(data)
            return data, label
        else:                                                #对于测试
            data = cv2.imread(sample)
            data = cv2.resize(data, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))

            #####TTA
            img1 = data                                      #原图
            img2 = cv2.flip(data, 1)                         #水平反转图
            img3 = Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB)) #RGB图
            img4 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)) #RGB水平反转
            img5 = tta_amaugimg(img1)                        #灰度图
            img6 = tta_amaugimg(img2)                        #反转灰度图

            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
            img3 = self.transforms(img3)
            img4 = self.transforms(img4)
            img5 = self.transforms(img5)
            img6 = self.transforms(img6)

            return img1, img2, img3, img4, img5, img6

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(root='/mnt/sdb2/liucen/CCF_face/new_training/train/',
                      phase='train',
                      input_shape=(1, 112, 112))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        plt.imshow(img)
        plt.show()

        # cv2.imshow('img', img)
        # cv2.waitKey()
        break
        # dst.decode_segmap(labels.numpy()[0], plot=True)