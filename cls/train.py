import os
import sys
import argparse
from datetime import datetime
from criterion.LabelSmoothing import LSR, CrossEntropyLabelSmooth

import numpy as np
import torch
import time
from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_val_dataloader, WarmupMultiStepLR, make_optimizer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(tiger_training_loader):

        # images = Variable(images)
        # labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(tiger_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        if batch_index % 100 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(tiger_training_loader) * args.b
            ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in tiger_val_loader:

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(tiger_val_loader.dataset),
        correct.float() / len(tiger_val_loader.dataset)
    ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(tiger_val_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(tiger_val_loader.dataset), epoch)

    return correct.float() / len(tiger_val_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='vgg11', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=16, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')            #原来是64,se-res101,152是32 $448->16
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')    #res224时候都是0.001,seres50是0.0005,seres101是0.00018/ 448res时候是0.0025
    parser.add_argument('-pretrained', type=str, default='./pretrained/se_resnet152-d17c99b7.pth', help='pretrained model')      #预训练
    parser.add_argument('-bool_pretrained', type=str, default=True,help='vgg_pretrained model')  # 除了resnset是否用于训练模型预训练
    parser.add_argument('-checkpoint', type=str, default=None, help='pretrained model')  #断点继续
    parser.add_argument('-num_classes', type=int, default=35, help='class_num')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)

    if args.checkpoint != None:                                                       #断点继续训练
        state_dictBA = torch.load(args.checkpoint)
        net.load_state_dict(state_dictBA)
        
    #data preprocessing:
    tiger_training_loader = get_training_dataloader(
        root_path=settings.TRAIN_PATH,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    tiger_val_loader = get_val_dataloader(
        root_path=settings.TRAIN_PATH,
        num_workers=args.w,
        batch_size=8,
        shuffle=args.s
    )

    loss_function = CrossEntropyLabelSmooth(num_classes=35)

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)   #learning rate decay
    # iter_per_epoch = len(tiger_training_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #原来的lr策略
    optimizer = make_optimizer(args, net)
    warmup_scheduler = WarmupMultiStepLR(
        optimizer,
        settings.MILESTONES,
        gamma=0.5,                                     #0.1, 0.5
        warmup_factor=1.0 / 3,
        warmup_iters=0,
        warmup_method="linear",
        last_epoch=-1,
    )

    #cycle lr
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max = 10,
    #     eta_min = 0.000001
    # )

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):

        start_clock = time.time()
        warmup_scheduler.step()
        train(epoch)
        print('Time used:', time.time()-start_clock)

        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()