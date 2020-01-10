# -*- coding: utf-8 -*-
# Author: craig.li(solitaire10@163.com)
# Defines two loss functions which are used in split model and merge model.

import torch

from functools import reduce


def bce_loss(pred, label):
  """
  Loss function for Split model
  Args:
    pred(torch.tensor): Prediction
    label(torch.tensor): Ground truth
  Return:
    loss(torch.tensor): Loss of the input image
  """
  row_pred, column_pred = pred
  row_label, column_label = label

  criterion = torch.nn.BCELoss(torch.tensor([10.])).cuda()

  lr3 = criterion(row_pred[0].view(-1), row_label.view(-1))

  lc3 = criterion(column_pred[0].view(-1), column_label.view(-1))

  loss = lr3 + lc3

  return loss


def merge_loss(pred, label, weight):
  """
  Loss function for training Merge model which is Binary Cross-Entropy loss.

  Args:
    pred(torch.tensor): Prediction of the input image
    label(torch.tensor): Ground truth of corresponding image
    weight(float): Weight to balance positive and negative samples
  Return:
    loss(torch.tensor): Loss of the input image
    D(torch.tensor): A matrix with size (M - 1) x N, indicates the probability
        of two neighbor cells should be merged in vertical direction.
    R(torch.tensor): A matrix with size M * (N - 1), indicates the probability
        of two neighbor cells should be merged in horizontal direction.
        Where M is the height of the label, and N is the width of the label
  """
  pu, pd, pl, pr = pred
  D = 0.5 * pu[:, :-1, :] * pd[:, 1:, :] + 0.25 * (pu[:, :-1, :] + pd[:, 1:, :])
  R = 0.5 * pr[:, :, :-1] * pl[:, :, 1:] + 0.25 * (pr[:, :, :-1] + pl[:, :, 1:])

  DT, RT = label

  criterion = torch.nn.BCELoss(torch.tensor([weight])).cuda()

  ld = criterion(D.view(-1), DT.view(-1))
  lr = criterion(R.view(-1), RT.view(-1))
  losses = []
  if D.view(-1).shape[0] != 0:
    losses.append(ld)
  if R.view(-1).shape[0] != 0:
    losses.append(lr)
  if len(losses) == 0:
    loss = torch.tensor(0).cuda()
  else:
    loss = reduce(lambda x, y: x + y, losses)
  return loss, D, R
