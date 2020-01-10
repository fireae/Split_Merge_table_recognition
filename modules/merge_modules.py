# -*- coding: utf-8 -*-
# Author: craig.li(solitaire10@163.com)
# Defines several modules to build up Merge model

import torch
import torch.nn as nn


class MergeModel(nn.Module):
  """
  Merge model refer to ICDAR 2019 Deep Splitting and Merging for Table Structure Decomposition
  """

  def __init__(self, input_channels):
    """
    Initialization of merge model
    Args:
      input_channels(int): The number of input data channel
    """
    super(MergeModel, self).__init__()
    # shared full conv net
    self.sfcn = SharedFCN(input_channels)
    # four branches: up, down, left, right
    self.rpn1 = ProjectionNet(18, True, 0.3)
    self.rpn2 = ProjectionNet(36, False, 0)
    self.rpn3 = ProjectionNet(36, True, 0.3)

    self.dpn1 = ProjectionNet(18, True, 0.3)
    self.dpn2 = ProjectionNet(36, False, 0)
    self.dpn3 = ProjectionNet(36, True, 0.3)

    self.upn1 = ProjectionNet(18, True, 0.3)
    self.upn2 = ProjectionNet(36, False, 0)
    self.upn3 = ProjectionNet(36, True, 0.3)

    self.lpn1 = ProjectionNet(18, True, 0.3)
    self.lpn2 = ProjectionNet(36, False, 0)
    self.lpn3 = ProjectionNet(36, True, 0.3)

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

  def forward(self, x, arc):
    """
    Forward pass of the merge model
    Args:
      x(torch.tensor): Input tensor with shape (b,c,h,w)
      arc(list(torch.tensor)): Grid architecture for grid project pooling,
          contains two lists of tensor indicates coordinates of horizontal and
          vertical lines
    Return:
      output(list(torch.tensor)): output of the merge model, a list contains four matrices
          indicates four direction
    """
    feature = self.sfcn(x)

    right_feature, r3 = self.rpn1(feature,
                                  arc)  # self.rpn3(self.rpn2(self.rpn1(feature, arc), arc), arc)

    left_feature, l3 = self.lpn1(feature,
                                 arc)  # self.lpn3(self.lpn2(self.lpn1(feature, arc), arc), arc)

    up_feature, u3 = self.upn1(feature,
                               arc)  # self.upn3(self.upn2(self.upn1(feature, arc), arc), arc)

    down_feature, d3 = self.dpn1(feature,
                                 arc)  # self.dpn3(self.dpn2(self.dpn1(feature, arc), arc), arc)

    output = [u3.squeeze(1), d3.squeeze(1), l3.squeeze(1), r3.squeeze(1)]
    return output


class GridProjectPooling(nn.Module):
  """
  Grid project pooling, every pixel location replaces its value with the average of all
    pixels within its grid element:
    $$ \hat F_{ij} = \frac {1}{\lvert\Omega(i,j)\rvert} \sum_{i',j' \in \Omega (i,j)} F_i',j' $$
  """

  def __init__(self):
    """
    Initialization of grid project pooling module
    """
    super(GridProjectPooling, self).__init__()

  def forward(self, x, architecture):
    """
    Forward pass of this module
    Args:
      x(torch.tensor): Input tensor with shape (b, c, h, w)
      architecture(list(torch.tensor)): Grid architecture for grid project pooling,
          contains two lists of tensor indicates coordinates of horizontal and
          vertical lines
    Return:
      output(torch.tensor): Output tensor of this module, the shape is same with
          input tensor
      matrix(torch.tensor): A M x N matrix, where M and N indicates the number of
          lines in horizontal and vertical directions.
    """
    b, c, h, w = x.size()
    h_line, v_line = architecture
    self.h_line = [torch.Tensor([0]).type(
      torch.DoubleTensor).cuda()] + h_line + [
                    torch.Tensor([1]).type(torch.DoubleTensor).cuda()]
    self.v_line = [torch.Tensor([0]).type(
      torch.DoubleTensor).cuda()] + v_line + [
                    torch.Tensor([1]).type(torch.DoubleTensor).cuda()]
    self.h_line = [(h * x).round().type(torch.IntTensor) for x in self.h_line]
    self.v_line = [(w * x).round().type(torch.IntTensor) for x in self.v_line]

    rows = [self.h_line[i + 1] - self.h_line[i] for i in
            range(len(self.h_line) - 1)]
    columns = [self.v_line[i + 1] - self.v_line[i] for i in
               range(len(self.v_line) - 1)]

    slices = torch.split(x, rows, 2)
    means = [torch.mean(y, 2).unsqueeze(2) for y in slices]
    matrix = torch.cat(means, 2)
    blocks = [means[i].repeat(1, 1, rows[i], 1) for i in range(len(means))]
    block = torch.cat(blocks, 2)

    means = [torch.mean(y, 3).unsqueeze(3) for y in
             torch.split(matrix, columns, 3)]
    matrix = torch.cat(means, 3)

    block_mean = [torch.mean(y, 3).unsqueeze(3) for y in
                  torch.split(block, columns, 3)]

    blocks = [block_mean[i].repeat(1, 1, 1, columns[i]) for i in
              range(len(block_mean))]
    output = torch.cat(blocks, 3)
    """
    Old version Grid pooling
    v_blocks = []
    matrix = torch.from_numpy(np.ones([b, c, len(self.h_line) - 1, len(self.v_line) - 1])).type(
        torch.FloatTensor).cuda()
    for i in range(len(self.h_line) - 1):
        h_blocks = []
        for j in range(len(self.v_line) - 1):
            output_block = torch.from_numpy(
                np.ones([b, c, self.h_line[i + 1] - self.h_line[i], self.v_line[j + 1] - self.v_line[j]])).type(
                torch.FloatTensor).cuda()
            mean = torch.mean(
                torch.mean(x[:, :, self.h_line[i]:self.h_line[i + 1], self.v_line[j]:self.v_line[j + 1]], 2),
                2).cuda()
            matrix[:, :, i, j] = mean
            h_blocks.append(mean.unsqueeze(0).transpose(0, 2) * output_block)
        h_block = torch.cat(h_blocks, 3)
        v_blocks.append(h_block)
    output = torch.cat(v_blocks, 2)
    """
    return output, matrix


class ProjectionNet(nn.Module):
  """
  Projection Module contains three parallel conv layers with dilation factor 1,2,3, followed by
    a grid project pooling module
  """

  def __init__(self, input_channels, sigmoid=False, dropout=0.5):
    """
    Initialization of Project module
    Args:
      input_channels(int): The number of input channels of the module
      sigmoid(bool): If need to ge the output matrix
      dropout(float): Drop out ratio
    """
    super(ProjectionNet, self).__init__()
    self.conv_branch1 = nn.Sequential(
      nn.Conv2d(input_channels, 6, 3, stride=1, padding=1, dilation=1),
      nn.GroupNorm(3, 6), nn.ReLU(True))
    self.conv_branch2 = nn.Sequential(
      nn.Conv2d(input_channels, 6, 3, stride=1, padding=2, dilation=2),
      nn.GroupNorm(3, 6), nn.ReLU(True))
    self.conv_branch3 = nn.Sequential(
      nn.Conv2d(input_channels, 6, 3, stride=1, padding=3, dilation=3),
      nn.GroupNorm(3, 6), nn.ReLU(True))
    self.sigmoid = sigmoid
    self.project_module = ProjectionModule(18, sigmoid, dropout=dropout)

  def forward(self, x, arc):
    """
    Forward pass of Project module
    Args:
      x(torch.tensor): Input tensor with shape (b,c,h,w)
      arc(list(torch.tensor)): Grid architecture for grid project pooling,
          contains two lists of tensor indicates coordinates of horizontal and
          vertical lines
    Return:
      output(torch.tensor): Output tensor of this module, the shape is same with
          input tensor
      matrix(torch.tensor): A M x N matrix, where M and N indicates the number of
          lines in horizontal and vertical directions.
    """
    conv_out = torch.cat(
      [m(x) for m in [self.conv_branch1, self.conv_branch2, self.conv_branch3]],
      1)
    output, matrix = self.project_module(conv_out, arc)
    if self.sigmoid:
      return output, matrix
    else:
      return output


class ProjectionModule(nn.Module):
  """
  Projection block
  """

  def __init__(self, input_channels, sigmoid=False, dropout=0.5):
    """
    Initialization of Project module
    Args:
      input_channels(int): The number of input channels of the module
      sigmoid(bool): If need to ge the output matrix
      dropout(float): Drop out ratio
    """
    super(ProjectionModule, self).__init__()
    self.sigmoid = sigmoid

    self.feature_conv = nn.Sequential(
      nn.Conv2d(input_channels, input_channels, 1, bias=False)
      , nn.GroupNorm(6, input_channels), nn.ReLU(True))
    self.prediction_conv = nn.Sequential(nn.Dropout2d(p=dropout),
                                         nn.Conv2d(input_channels, 1, 1,
                                                   bias=False))

    self.feature_project = GridProjectPooling()
    self.prediction_project = GridProjectPooling()
    self.sigmoid_layer = nn.Sigmoid()

  def forward(self, x, arch):
    """
    Forward pass of Project module
    Args:
      x(torch.tensor): Input tensor with shape (b,c,h,w)
      arch(list(torch.tensor)): Grid architecture for grid project pooling,
          contains two lists of tensor indicates coordinates of horizontal and
          vertical lines
    Return:
      output(torch.tensor): Output tensor of this module, the shape is same with
          input tensor
      matrix(torch.tensor): A M x N matrix, where M and N indicates the number of
          lines in horizontal and vertical directions.
    """
    base_input = x
    feature = self.feature_conv(base_input)
    feature, _ = self.feature_project(feature, arch)
    tensors = [base_input, feature]
    if self.sigmoid:
      prediction = self.prediction_conv(base_input)
      prediction, matrix = self.prediction_project(prediction, arch)
      prediction = self.sigmoid_layer(prediction)
      matrix = self.sigmoid_layer(matrix)
      tensors.append(prediction)
      output = torch.cat(tensors, 1)
      return output, matrix
    else:
      output = torch.cat(tensors, 1)
      return output, None


class SharedFCN(nn.Module):
  """Shared fully convolution module"""

  def __init__(self, input_channels):
    """
    Initialization of SFCN instance
    Args:
      input_channels(int): The number of input channels of the module
    """
    super(SharedFCN, self).__init__()
    self.conv = nn.Sequential(
      nn.Sequential(
        nn.Conv2d(input_channels, 18, 7, stride=1, padding=3, bias=False),
        nn.ReLU(True)),
      nn.Sequential(nn.Conv2d(18, 18, 7, stride=1, padding=3, bias=False),
                    nn.ReLU(True)),
      nn.MaxPool2d((2, 2)),
      nn.Sequential(nn.Conv2d(18, 18, 7, stride=1, padding=3, bias=False),
                    nn.ReLU(True)),
      nn.Sequential(nn.Conv2d(18, 18, 7, stride=1, padding=3, bias=False),
                    nn.ReLU(True)),
      nn.MaxPool2d((2, 2))
    )

  def forward(self, x):
    x = self.conv(x)
    return x
