# -*- coding: utf-8 -*-
# Author: craig.li(solitaire10@163.com)
# Defines several modules to build up Split model

import numpy as np
import torch
import torch.nn as nn


class SplitModel(nn.Module):
  """
  Split model refer to ICDAR 2019 Deep Splitting and Merging for Table Structure Decomposition
  """

  def __init__(self, input_channels):
    """
    Initialization of split model
    Args:
      input_channels(int): The number of input data channel
    """
    super(SplitModel, self).__init__()
    self.sfcn = SFCN(input_channels)
    self.rpn1 = ProjectionNet(18, 0, True, False, 0)
    self.rpn2 = ProjectionNet(36, 0, True, False, 0)
    self.rpn3 = ProjectionNet(36, 0, False, True, 0.3)
    self.rpn4 = ProjectionNet(37, 0, False, True, 0)
    # self.rpn5 = ProjectionNet(37, 0, False, True, 0)

    self.cpn1 = ProjectionNet(18, 1, True, False, 0)
    self.cpn2 = ProjectionNet(36, 1, True, False, 0)
    self.cpn3 = ProjectionNet(36, 1, False, True, 0.3)
    self.cpn4 = ProjectionNet(37, 1, False, True, 0)
    # self.cpn5 = ProjectionNet(37, 1, False, True, 0)

    self._init_weights()

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
          m.bias.data.fill_(0.01)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

  def forward(self, x):
    """
    Forward pass of the split model
    Args:
      x(torch.tensor): Input tensor with shape (b,c,h,w)
    Return:
      output(list(torch.tensor)): output of the split model, two vectors indicates if there
          is a line in horizontal and vertical direction
    """
    feature = self.sfcn(x)

    row_feature = self.rpn3(self.rpn2(self.rpn1(feature)))
    r3 = row_feature[:, -1, :, :]
    # row_feature = self.rpn4(row_feature)
    # r4 = row_feature[:, -1, :, :]
    # row_feature = self.rpn5(row_feature)
    # r5 = row_feature[:, -1, :, :]

    cow_feature = self.cpn3(self.cpn2(self.cpn1(feature)))
    c3 = cow_feature[:, -1, :, :]
    # cow_feature = self.cpn4(cow_feature)
    # c4 = cow_feature[:, -1, :, :]
    # cow_feature = self.cpn5(cow_feature)
    # c5 = cow_feature[:, -1, :, :]

    # r = self.rpn1(feature)[:,-1,:,:]
    # c = self.cpn1(feature)[:,-1,:,:]
    # return (
    #    torch.cat([r3[:, :, 0], r4[:, :, 0]], 0),#, r5[:, :, 0]
    #    torch.cat([c3[:, 0, :], c4[:, 0, :]], 0))#, c5[:, 0, :]
    return (r3[:, :, 0], c3[:, 0, :])


class ProjectPooling(nn.Module):
  """
  Project pooling, replace each value in the input with its row(column) average,
    Row project pooling:
        $$ \hat F_{ij} = \frac {1}{W} \sum_{j'=1}^{W} F_i',j' $$
    Column project pooling:
        $$ \hat F_{ij} = \frac {1}{H} \sum_{i'=1}^{H} F_i',j' $$
  """

  def __init__(self, direction):
    """
    Initialization of Project pooling layer
    Args:
      direction(int): Specifies the direction of this layer, 0 for row and 1 for column
    """
    super(ProjectPooling, self).__init__()
    self.direction = direction

  def forward(self, x):
    """
    Forward pass of project pooling layer
    Args:
      x(torch.tensor): Input tensor with shape (b,c,h,w)
    Return:
      output: Output of Project pooling layer with the same shape with input tensor
    """
    b, c, h, w = x.size()
    output_slice = torch.from_numpy(np.ones([b, c, h, w])).type(
      torch.FloatTensor).cuda()
    if self.direction == 0:
      return torch.mean(x, 3).unsqueeze(3) * output_slice
    elif self.direction == 1:
      return torch.mean(x, 2).unsqueeze(2) * output_slice
    else:
      raise Exception(
        'Wrong direction, the direction should be 0 for horizontal and 1 for vertical')


class ProjectionNet(nn.Module):
  """
  Projection Module contains three parallel conv layers with dilation factor 2,3,4, followed by
      a project pooling module
  """

  def __init__(self, input_channels, direction, max_pooling=False,
               sigmoid=False, dropout=0.5):
    super(ProjectionNet, self).__init__()
    self.conv_branch1 = nn.Sequential(
      nn.Conv2d(input_channels, 6, 3, stride=1, padding=2, dilation=2),
      nn.GroupNorm(3, 6), nn.ReLU(True))
    self.conv_branch2 = nn.Sequential(
      nn.Conv2d(input_channels, 6, 3, stride=1, padding=3, dilation=3),
      nn.GroupNorm(3, 6), nn.ReLU(True))
    self.conv_branch3 = nn.Sequential(
      nn.Conv2d(input_channels, 6, 3, stride=1, padding=4, dilation=4),
      nn.GroupNorm(3, 6), nn.ReLU(True))

    self.project_module = ProjectionModule(18, direction, max_pooling, sigmoid,
                                           dropout=dropout)

  def forward(self, x):
    """
    Forward pass of Project module
    Args:
      x(torch.tensor): Input tensor with shape (b,c,h,w)
    Return:
      output(torch.tensor): Output tensor of this module, the shape is same with
          input tensor
    """
    conv_out = torch.cat(
      [m(x) for m in [self.conv_branch1, self.conv_branch2, self.conv_branch3]],
      1)
    output = self.project_module(conv_out)
    return output


class ProjectionModule(nn.Module):
  """
  Projection block
  """

  def __init__(self, input_channels, direction, max_pooling=False,
               sigmoid=False, dropout=0.5):
    """
    Initialization of Project module
    Args:
      input_channels(int): The number of input channels of the module
      direction(int): Direction of project pooling module, 0 for row, 1 for column
      max_pooling(bool): If there is a max pooling layer in the module, if it's a
          row project pooling layer, a (1,2) max pooling layer would be applied,
          (2,1) max pooling would be applied if it's a column project pooling layer
      sigmoid(bool): If need to ge the output matrix
      dropout(float): Drop out ratio
    """
    super(ProjectionModule, self).__init__()
    self.direction = direction
    self.max_pooling = max_pooling
    self.sigmoid = sigmoid
    self.max_pool = nn.MaxPool2d((1, 2)) if direction == 0 else nn.MaxPool2d(
      (2, 1))
    self.feature_conv = nn.Sequential(
      nn.Conv2d(input_channels, input_channels, 1, bias=False)
      , nn.GroupNorm(6, input_channels), nn.ReLU(True))
    self.prediction_conv = nn.Sequential(nn.Dropout2d(p=dropout),
                                         nn.Conv2d(input_channels, 1, 1,
                                                   bias=False))

    self.feature_project = ProjectPooling(direction)
    self.prediction_project = nn.Sequential(ProjectPooling(direction),
                                            nn.Sigmoid())

  def forward(self, x):
    """
    Forward pass of Project module
    Args:
      x(torch.tensor): Input tensor with shape (b,c,h,w)
    Return:
      output(torch.tensor): Output tensor of this module, if a max pooling layer is
          applied, the output shape would be decreased to half of the original shape
          in opposite direction
    """
    base_input = x
    if self.max_pooling:
      base_input = self.max_pool(x)
    feature = self.feature_conv(base_input)
    feature = self.feature_project(feature)
    tensors = [base_input, feature]
    if self.sigmoid:
      prediction = self.prediction_conv(base_input)
      prediction = self.prediction_project(prediction)
      tensors.append(prediction)
    output = torch.cat(tensors, 1)
    return output


class SFCN(nn.Module):
  """
  Shared fully convolution module composed of three conv layers, and the last conv layer is
    a dilation conv layer with the factor 2
  """

  def __init__(self, input_channels):
    """
    Initialization of SFCN instance
    Args:
      input_channels(int): The number of input channels of the module
    """
    super(SFCN, self).__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(input_channels, 18, 7, stride=1, padding=3, bias=False),
      nn.ReLU(True))
    self.conv2 = nn.Sequential(
      nn.Conv2d(18, 18, 7, stride=1, padding=3, bias=False),
      nn.ReLU(True))
    self.conv3 = nn.Sequential(
      nn.Conv2d(18, 18, 7, stride=1, padding=6, dilation=2, bias=False),
      nn.ReLU(True))

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    return x
