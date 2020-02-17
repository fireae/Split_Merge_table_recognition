# -*- coding: utf-8 -*-
# Author: craig.li(solitaire10@163.com)
# Script for generating data of Split model..


import cv2
import json
import numpy as np
import os
import xml.etree.ElementTree as ET

from functools import reduce
from PIL import Image, ImageDraw


def make_split_data(file, lines, img_dir, out_img_dir, out_mask_dir, length_threshold=20):
    """
    Generate data for training Split model, crops and saves image blocks of table from original image,
        returns labels of image blocks

    Args:
        file(str): File name of original image file.
        img_dir(str): The directory contains image files.
        out_img_dir(str): A directory to save croped image blocks.
        out_mask_dir(str): A directory to save masks of croped image blocks.
        length_threshold(int):Threshold of filtering blocks with short width or height.
    Returns:
        label_dict(dict): A directory, for each item, key is the file id and value contains 'rows' and 'columns',
            which are two vectors, 1 indicates there is a line in corresponding row or column.
    """
    # loads original image
    im_path = os.path.join(img_dir, file)
    im = Image.open(im_path)
    im = im.convert('L')
    im_array = np.array(im)
    # init blank images for drawing masks.
    masks = [Image.new('L', im.size, (0,)) for i in range(4)]
    draws = [ImageDraw.Draw(x) for x in masks]
    # Calculates angle of the image according to labeled boxes.
    thetas = []
    for line in lines:
        draws[int(line['type'])].polygon(line['coordinates'], fill=1)
        if line['type'] == 'h':  # line['type'] should be 'v' or 'h' for vertical and horizontal lines
            up_left = line['coordinates'][:2]
            up_right = line['coordinates'][2:4]
            theta = np.arctan((up_right[1] - up_left[1]) / (up_right[0] - up_left[0]))
            thetas.append(theta)
    theta = np.average(thetas)
    matrix = np.array([[np.cos(-theta), -np.sin(-theta), 0],
                       [np.sin(-theta), np.cos(-theta), 0]])
    # rotates mask images and original images.
    masks = [cv2.warpAffine(np.array(x), matrix, np.array(x).shape[::-1]) for x in masks]
    mask_data = np.array([np.array(x) for x in masks])
    im_array = np.array(cv2.warpAffine(im_array, matrix, im_array.shape[::-1]))
    mask_img = cv2.bitwise_or(cv2.bitwise_or(mask_data[0], mask_data[2]),
                              cv2.bitwise_or(mask_data[1], mask_data[3])) * 255.
    mask_img = np.array(mask_img, dtype=np.uint8)

    # Splits different tables by connected components.
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_img, 4, cv2.CV_32S)
    label_dict = {}
    for i in range(1, num):
        image, contours = cv2.findContours(np.array((labels == i) * 255., dtype=np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(image[0])
        if w < length_threshold or h < length_threshold:
            continue
        row_start = int(y)
        row_end = int(y + h)
        column_start = int(x)
        column_end = int(x + w)
        # filters blocks
        if (row_end - row_start) <= 0 or (column_end - column_start) <= 0:
            continue
        row_table = cv2.bitwise_or(mask_data[0][row_start:row_end, column_start:column_end],
                                   mask_data[2][row_start:row_end, column_start:column_end])
        column_table = cv2.bitwise_or(mask_data[1][row_start:row_end, column_start:column_end],
                                      mask_data[3][row_start:row_end, column_start:column_end])
        # makes column and row labels.
        row_label = []
        for row in range(row_table.shape[0]):
            row_label.append(1 if np.sum(row_table[row, :]) > 0 else 0)
        column_lable = []
        for column in range(column_table.shape[1]):
            column_lable.append(1 if np.sum(column_table[:, column]) > 0 else 0)
        key = file[:-4] + '_' + str(i)
        label_dict[key] = {'row': row_label, 'column': column_lable}
        # crops image blocks of tables.
        mask_table = mask_img[row_start:row_end, column_start:column_end]
        original_img = im_array[row_start:row_end, column_start:column_end]
        mask_table = Image.fromarray(np.array(mask_table, dtype=np.uint8))
        original_img = Image.fromarray(np.array(original_img, dtype=np.uint8))
        # saves cropped blocks.
        original_img.save(os.path.join(out_img_dir, key + '.jpg'))
        mask_table.save(os.path.join(out_mask_dir, key + '.jpg'))
    return label_dict


def extract_lines_from_xml(root_dir, file):
    """
    Extracts labels of lines from original xml file.
    You can rewrite this function to extract lines from your original label.

    Args:
        root_dir(str): The directory to the folder which contains the file.
        file(str): The file name of the xml file.
    Returns:
         lines(list): A list of lines' coordinates, the coordinates should be a list of 8 integers, indicates x, y
            coordinates of a corner in the flowing order up_left->up_right->down_right->down_left. Type should be
            'v' or 'h' for vertical and horizontal lines.
    """
    # extracts data from xml file
    tree = ET.parse(os.path.join(root_dir, file))
    root = tree.getroot()
    elements = root.findall('object')
    # Constructs a list contains lines.
    lines = []
    for element in elements:
        category = element.find('name').text
        bbox = element.find('bndbox')
        coordinates = [] + bbox.find('leftup').text.split(',') + bbox.find('rightup').text.split(',') + bbox.find(
            'rightdown').text.split(',') + bbox.find('leftdown').text.split(',')
        coordinates = [int(x) for x in coordinates]
        lines.append({'type': category, 'coordinates': coordinates})
    return lines


def merge_dicts(d0, d1):
    """
    Merges two directories.
    """
    for k, v in d1.items():
        d0[k] = v
    return d0


if __name__ == "__main__":
    root_dir = 'root_dir'
    img_dir = 'img_dir'
    out_img_dir = 'test_out_img_dir'
    out_mask_dir = 'test_out_mask_dir'
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    if not os.path.exists(out_mask_dir):
        os.mkdir(out_mask_dir)
    files = os.listdir(img_dir)
    json_path = 'test_labels.json'
    ids = [x[:-4] for x in files]
    label_dicts = []
    for id in ids:
        file = id + '.xml'
        lines = extract_lines_from_xml(root_dir, file)
        img_name = id + '.jpg'
        label_dict = make_split_data(img_name, lines, img_dir, out_img_dir, out_mask_dir)
        label_dicts.append(label_dict)
    labels = reduce(merge_dicts, label_dicts)
    with open(json_path, 'w') as f:
        f.write(json.dumps(labels))
