import os
from PIL import Image
from imgaug import  augmenters as iaa
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys


def amaugimg(image):
    #数据增强
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    sometimes = lambda  aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            iaa.SomeOf((0,4),
                       [
                           iaa.CoarseDropout(
                               (0.03, 0.15), size_percent=(0.01, 0.05),
                               per_channel=0.2
                           ),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01* 255), per_channel=0.5),
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 1.5)),
                               iaa.AverageBlur(k=(2,7)),
                           ]),
                           iaa.Grayscale(alpha=(0.0, 1.0)),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25))
                       ],
                       random_state=True
            )
        ]
    )
    image = seq.augment_image(image)

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return image

class Dataset(data.Dataset):
    def __init__(self, root, phase='train', input_shape=(3, 112, 112)):
        self.phase = phase
        self.input_shape = input_shape
        self.phase = phase

        imgs = []
        if phase == 'train':

            for pic in os.listdir(root):
                imgs.append(pic)

            imgs = [os.path.join(root, img) for img in imgs]
            self.imgs = np.random.permutation(imgs)

        else:
            self.imgs = [root]

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize(self.input_shape[1:]),
                # T.RandomCrop(self.input_shape[1:]),
                T.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop((64, 64)),
                T.Resize(self.input_shape[1:]),
                # T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        if self.phase == 'train':
            splits = sample.split('/', 7)[-1]                #分割6次，可变
            label = int(splits.split('_')[0])
            img_path = sample
            data = Image.open(img_path).convert('RGB')
            # data = amaugimg(data)
            data = self.transforms(data)
            return data, label
        else:                                                #对于测试
            data = Image.open(sample)
            data = data.convert('RGB')
            img1 = data                                      #原图
            img2 = data.transpose(Image.FLIP_LEFT_RIGHT)     #水平反转图

            data = data.convert('L')                         #灰度图
            img3 = cv2.cvtColor(np.asarray(data), cv2.COLOR_RGB2BGR)
            img3 = Image.fromarray(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

            img4 = img3.transpose(Image.FLIP_LEFT_RIGHT)     #灰度反转

            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
            img3 = self.transforms(img3)
            img4 = self.transforms(img4)

            return img1, img2

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