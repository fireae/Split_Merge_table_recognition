# -*- coding: utf-8 -*-
# Author: craig.li(solitaire10@163.com)
# Script for training merge model


import argparse
import json
import os
import torch
import torch.backends.cudnn as cudnn

from torch import optim
from torch.utils.data import DataLoader
from dataset.dataset import ImageDataset
from loss.loss import bce_loss
from modules.split_modules import SplitModel
from split.test import test


def train(opt, net):
  """
  Train the split model
  Args:
    opt(dic): Options
    net(torch.model): Split model instance
  """
  with open(opt.json_dir, 'r') as f:
    labels = json.load(f)
  dir_img = opt.img_dir

  with open(opt.val_json, 'r') as f:
    val_labels = json.load(f)
  val_img_dir = opt.val_img_dir

  train_set = ImageDataset(dir_img, labels, opt.featureW, scale=opt.scale)
  train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

  val_set = ImageDataset(val_img_dir, val_labels, opt.featureW, scale=opt.scale)
  val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False)

  print('Data loaded!')

  loss_func = bce_loss
  optimizer = optim.Adam(net.parameters(),
                         lr=opt.lr,
                         weight_decay=0.001)
  best_accuracy = 0
  for epoch in range(opt.epochs):
    print('epoch:{}'.format(epoch + 1))
    net.train()
    epoch_loss = 0
    correct_count = 0
    count = 0
    for i, b in enumerate(train_loader):
      img, label = b
      if opt.gpu:
        img = img.cuda()
        label = [x.cuda() for x in label]
      pred_label = net(img)
      loss = loss_func(pred_label, label, [0.1, 0.25, 1])
      epoch_loss += loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      times = 1
      correct_count += (torch.sum(
        (pred_label[0] > 0.5).type(torch.IntTensor) == label[0][0].repeat(times,
                                                                          1).type(
          torch.IntTensor)).item() + torch.sum(
        (pred_label[1] > 0.5).type(torch.IntTensor) == label[1][0].repeat(times,
                                                                          1).type(
          torch.IntTensor)).item())
      count += label[0].view(-1).size()[0] * times + label[1].view(-1).size()[
        0] * times
    accuracy = correct_count / (count)
    print(
      'Epoch finished ! Loss: {0} , Accuracy: {1}'.format(epoch_loss / (i + 1),
                                                          accuracy))
    val_loss, val_acc = test(opt, net, val_loader)
    if val_acc > best_accuracy:
      best_accuracy = val_acc
      torch.save(net.state_dict(),
                 opt.saved_dir + 'CP{}.pth'.format(epoch + 1))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=1,
                      help='batch size of the training set')
  parser.add_argument('--epochs', type=int, default=50, help='epochs')
  parser.add_argument('--gpu', type=bool, default=True, help='if use gpu')
  parser.add_argument('--gpu_list', type=str, default='0',
                      help='which gpu could use')
  parser.add_argument('--lr', type=float, default=0.00075,
                      help='learning rate, default=0.00075 for Adam')
  parser.add_argument('--saved_dir', type=str, required=True,
                      help='saved directory for output models')
  parser.add_argument('--json_dir', type=str, required=True,
                      help='labels of the data')
  parser.add_argument('--img_dir', type=str, required=True,
                      help='image directory for input data')
  parser.add_argument('--val_json', type=str, required=True,
                      help='labels of the validation data')
  parser.add_argument('--val_img_dir', type=str, required=True,
                      help='image directory for validation data')
  parser.add_argument('--featureW', type=int, default=8, help='width of output')
  parser.add_argument('--scale', type=float, default=0.5,
                      help='scale of the image')

  opt = parser.parse_args()

  net = SplitModel(3)
  if opt.gpu:
    cudnn.benchmark = True
    cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_list
    net = torch.nn.DataParallel(net).cuda()

  if not os.path.exists(opt.saved_dir):
    os.mkdir(opt.saved_dir)

  train(opt, net)
