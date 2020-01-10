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
from modules.merge_modules import MergeModel
from merge.test import test
from loss.loss import merge_loss


def train(opt, net):
  """
  Train the merge model
  Args:
    opt(dic): Options
    net(torch.model): Merge model instance
  """
  # load labels
  with open(opt.json_dir, 'r') as f:
    labels = json.load(f)
  dir_img = opt.img_dir

  with open(opt.val_json, 'r') as f:
    val_labels = json.load(f)
  val_img_dir = opt.val_img_dir

  train_set = ImageDataset(dir_img, labels, opt.featureW, scale=opt.scale,
                           mode='merge')
  train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

  val_set = ImageDataset(val_img_dir, val_labels, opt.featureW, scale=opt.scale,
                         mode='merge')
  val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False)

  print('Data loaded!')

  # defines loss function
  loss_func = merge_loss
  optimizer = optim.Adam(net.parameters(),
                         lr=opt.lr,
                         weight_decay=0.001)
  best_f1 = 0
  for epoch in range(opt.epochs):
    print('epoch:{}'.format(epoch + 1))
    net.train()
    epoch_loss = 0
    number_batchs = 0
    for i, b in enumerate(train_loader):
      img, label, arc = b
      if opt.gpu:
        img = img.cuda()
        label = [x.cuda() for x in label]
      pred_label = net(img, arc)
      loss, _, _ = loss_func(pred_label, label, 10.)
      if loss.requires_grad:
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        number_batchs += 1

    print('Epoch finished ! Loss: {0} '.format(epoch_loss / number_batchs))
    val_loss, precision, recall, f1 = test(opt, net, val_loader)
    # save model if best f1 score less than current f1 score
    if f1 > best_f1:
      best_f1 = f1
      torch.save(net.state_dict(),
                 opt.saved_dir + 'CP{}.pth'.format(epoch + 1))
    # write training information of current epoch to the log file
    with open(os.path.join(opt.saved_dir, 'log.txt'), 'a') as f:
      f.write(
        'Epoch {0}, val loss : {1}, precision : {2}, recall : {3}, f1 score : {4}  \n   tra loss : {5} \n\n'.format(
          epoch + 1, val_loss, precision, recall, f1,
          epoch_loss / number_batchs))


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

  net = MergeModel(3)
  if opt.gpu:
    cudnn.benchmark = True
    cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_list
    net = torch.nn.DataParallel(net).cuda()

  if not os.path.exists(opt.saved_dir):
    os.mkdir(opt.saved_dir)

  with open(os.path.join(opt.saved_dir, 'log.txt'), 'w') as f:
    configuration = '--batch_size: {0} \n' \
                    '--epochs: {1} \n' \
                    '--gpu: {2} \n' \
                    '--gpu_list: {3} \n' \
                    '--lr: {4} \n' \
                    '--saved_dir: {5} \n' \
                    '--json_dir: {6} \n' \
                    '--img_dir: {7} \n' \
                    '--val_json: {8} \n' \
                    '--val_img_dir: {9} \n' \
                    '--featureW: {10} \n' \
                    '--scale: {11} \n'.format(
      opt.batch_size, opt.epochs, opt.gpu, opt.gpu_list, opt.lr, opt.saved_dir,
      opt.json_dir, opt.img_dir,
      opt.val_json, opt.val_img_dir, opt.featureW, opt.scale)
    f.write(configuration + '\n\n Logs: \n')

  train(opt, net)
