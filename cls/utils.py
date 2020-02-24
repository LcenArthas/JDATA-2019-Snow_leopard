import sys

import numpy

import torch
from bisect import bisect_right
from dataset.dataset_self import Dataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader

def get_network(args, use_gpu=True):
    """ return given network
    """

    if args.net == 'vgg16':
        net = torchvision.models.vgg16_bn(pretrained=args.bool_pretrained)
        net.classifier[6] = nn.Linear(4096, args.num_classes, bias=True)
    elif args.net == 'vgg13':
        net = torchvision.models.vgg13_bn(pretrained=args.bool_pretrained)
        net.classifier[6] = nn.Linear(4096, args.num_classes, bias=True)
    elif args.net == 'vgg11':
        net = torchvision.models.vgg11_bn(pretrained=args.bool_pretrained)
        net.classifier[6] = nn.Linear(4096, args.num_classes, bias=True)
    elif args.net == 'vgg19':
        net = torchvision.models.vgg19_bn(pretrained=args.bool_pretrained)
        net.classifier[6] = nn.Linear(4096, args.num_classes, bias=True)

    ####effcientnet
    elif args.net == 'efficientnet-b5':
        from efficientnet_pytorch import EfficientNet
        if args.bool_pretrained == True:
            net = EfficientNet.from_pretrained('efficientnet-b5')
        else:
            net = EfficientNet.from_name('efficientnet-b5')
        net._fc = nn.Linear(2048, args.num_classes, bias=True)

    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        net = torchvision.models.densenet161(pretrained=args.bool_pretrained)
        in_features = net.classifier.in_features
        net.classifier = nn.Linear(in_features, args.num_classes, bias=True)
    elif args.net == 'densenet169':
        net = torchvision.models.densenet169(pretrained=args.bool_pretrained)
        in_features = net.classifier.in_features
        net.classifier = nn.Linear(in_features, args.num_classes, bias=True)
    elif args.net == 'densenet201':
        net = torchvision.models.densenet201(pretrained=args.bool_pretrained)
        in_features = net.classifier.in_features
        net.classifier = nn.Linear(in_features, args.num_classes, bias=True)

    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()

    ################## ResNet ########################################################
    elif args.net == 'resnet18':
        from models.resnet import resnet
        net = resnet(args.num_classes, 2, args.pretrained, args.net)
    elif args.net == 'resnet34':
        from models.resnet import resnet
        net = resnet(args.num_classes, 2, args.pretrained, args.net)
    elif args.net == 'resnet50':
        from models.resnet import resnet
        net = resnet(args.num_classes, 2, args.pretrained, args.net)
    elif args.net == 'resnet101':
        from models.resnet import resnet
        net = resnet(args.num_classes, 2, args.pretrained, args.net)
    elif args.net == 'resnet152':
        from models.resnet import resnet
        net = resnet(args.num_classes, 2, args.pretrained, args.net)

    ##################################################################################
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()

    ##################################################################
    elif args.net == 'se_resnext50':
        from models.resnext import se_resnext
        net = se_resnext(args.num_classes, 2, args.pretrained, args.net)
    elif args.net == 'se_resnext101':
        from models.resnext import se_resnext
        net = se_resnext(args.num_classes, 2, args.pretrained, args.net)

    #################################################################
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()

    #########################################################
    elif args.net == 'se_resnet50':
        from models.senet import seresnet
        net = seresnet(args.num_classes, 2, args.pretrained, args.net)
    elif args.net == 'se_resnet101':
        from models.senet import seresnet
        net = seresnet(args.num_classes, 2, args.pretrained, args.net)
    elif args.net == 'se_resnet152':
        from models.senet import seresnet
        net = seresnet(args.num_classes, 2, args.pretrained, args.net)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if use_gpu:
        net = net.cuda()

    return net

def get_training_dataloader(root_path, batch_size=16, num_workers=2, shuffle=True):

    tiger_training = Dataset(root=root_path, phase='train')
    tiger_training_loader = DataLoader(
        tiger_training,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size
    )

    return tiger_training_loader

def get_val_dataloader(root_path, batch_size=16, num_workers=2, shuffle=True):

    tiger_val = Dataset(root=root_path, phase='val')
    tiger_val_loader = DataLoader(
        tiger_val,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size
    )

    return tiger_val_loader

def get_test_dataloader(root_path, batch_size=16, num_workers=2, shuffle=True):

    tiger_test = Dataset(root=root_path, phase='test')
    tiger_test_loader = DataLoader(
        tiger_test,
        num_workers=num_workers,
        batch_size=batch_size
    )

    return tiger_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class WarmupMultiStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.lr
        weight_decay = 0.0005
        if "bias" in key:
            lr = cfg.lr * 1
            weight_decay = 0.0005
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = getattr(torch.optim, 'SGD')(params, momentum=0.9)
    return optimizer