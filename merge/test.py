# -*- coding: utf-8 -*-
# Author: craig.li(solitaire10@163.com)
# Script for testing Merge model.


import argparse
import json
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn

from dataset.dataset import ImageDataset
from modules.merge_modules import MergeModel
from loss.loss import merge_loss
from torch.utils.data import DataLoader


def test(opt, net, data=None):
  """
  Test script for Merge model
  Args:
    opt(dic): Options
    net(torch.model): Merge model instance
    data(dataloader): Dataloader or None, if load data with configuration in opt.
  Return:
    total_loss(torch.tensor): The total loss of the dataset
    precision(torch.tensor): Precision (TP / TP + FP)
    recall(torch.tensor): Recall (TP / TP + FN)
    f1(torch.tensor): f1 score (2 * precision * recall / (precision + recall))
  """
  if not data:
    with open(opt.json_dir, 'r') as f:
      labels = json.load(f)
    dir_img = opt.img_dir

    test_set = ImageDataset(dir_img, labels, opt.featureW, scale=opt.scale,
                            mode='merge')
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)
  else:
    test_loader = data

  loss_func = merge_loss

  for epoch in range(1):
    net.eval()
    epoch_loss = 0
    number_batchs = 0
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    for i, b in enumerate(test_loader):
      with torch.no_grad():
        img, label, arc = b
        if opt.gpu:
          img = img.cuda()
          label = [x.cuda() for x in label]
        pred_label = net(img, arc)
        loss, D, R = loss_func(pred_label, label, 10.)
        epoch_loss += loss

        tp = torch.sum(
          ((D.view(-1)[
              (label[0].view(-1) > 0.5).type(torch.ByteTensor)] > 0.5).type(
            torch.IntTensor) ==
           label[0].view(-1)[
             (label[0].view(-1) > 0.5).type(torch.ByteTensor)].type(
             torch.IntTensor))).item() + torch.sum(
          ((R.view(-1)[
              (label[1].view(-1) > 0.5).type(torch.ByteTensor)] > 0.5).type(
            torch.IntTensor) ==
           label[1].view(-1)[
             (label[1].view(-1) > 0.5).type(torch.ByteTensor)].type(
             torch.IntTensor))).item()
        tn = torch.sum(
          ((D.view(-1)[
              (label[0].view(-1) <= 0.5).type(torch.ByteTensor)] > 0.5).type(
            torch.IntTensor) ==
           label[0].view(-1)[
             (label[0].view(-1) <= 0.5).type(torch.ByteTensor)].type(
             torch.IntTensor))).item() + torch.sum(
          ((R.view(-1)[
              (label[1].view(-1) <= 0.5).type(torch.ByteTensor)] > 0.5).type(
            torch.IntTensor) ==
           label[1].view(-1)[
             (label[1].view(-1) <= 0.5).type(torch.ByteTensor)].type(
             torch.IntTensor))).item()
        fn = torch.sum(
          (label[0].view(-1) > 0.5).type(torch.ByteTensor)).item() + torch.sum(
          (label[1].view(-1) > 0.5).type(torch.ByteTensor)).item() - tp
        fp = torch.sum(
          (label[0].view(-1) < 0.5).type(torch.ByteTensor)).item() + torch.sum(
          (label[1].view(-1) < 0.5).type(torch.ByteTensor)).item() - tn

        total_fn += fn
        total_fp += fp
        total_tn += tn
        total_tp += tp
        number_batchs += 1
    total_loss = epoch_loss / number_batchs
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(
      'Validation finished ! Loss: {0} ; Precision: {1} ; Recall: {2} ; F1 Score: {3}'.format(
        total_loss,
        precision,
        recall,
        f1))
    return total_loss, precision, recall, f1


def model_select(opt, net):
  """
  Select best model with highest f1 score
  Args:
    opt(dict): Options
    net(torch.model): Merge model instance
  """
  model_dir = opt.model_dir
  models = os.listdir(model_dir)
  losses = []
  f1s = []
  for model in models:
    print(model)
    net.load_state_dict(torch.load(os.path.join(model_dir, model)))
    loss, precision, recall, f1 = test(opt, net)
    losses.append(loss)
    f1s.append(f1)
  min_loss_index = np.argmin(np.array(losses))
  max_f1_index = np.argmax(np.array(f1s))
  print('losses', min_loss_index, losses[min_loss_index],
        models[min_loss_index])
  print('f1 score', max_f1_index, f1s[max_f1_index], models[max_f1_index])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=32,
                      help='batch size of the training set')
  parser.add_argument('--gpu', type=bool, default=True, help='if use gpu')
  parser.add_argument('--gpu_list', type=str, default='0',
                      help='which gpu could use')
  parser.add_argument('--model_dir', type=str, required=True,
                      help='saved directory for output models')
  parser.add_argument('--json_dir', type=str, required=True,
                      help='labels of the data')
  parser.add_argument('--img_dir', type=str, required=True,
                      help='image directory for input data')
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

  net.load_state_dict(torch.load('merge_models_1225/CP90.pth'))
  print(test(opt, net))
  # model_select(opt, net)
