# -*- coding: utf-8 -*-
# Author: craig.li(solitaire10@163.com)
# Script for generating data of Merge model..


import cv2
import json
import numpy as np
import os

from PIL import Image


def make_merge_data(label, mask_img, threshold=5):
    """
    Generates merge labels

    Args:
        label(list): Table row and column vector, same as labels of Split model.
        mask_img(ndarray): Mask drawed with labeled line.
        threshold(int): Threshold .
    Returns:
        h_matrix(ndarray): Label of Merge model data in horizontal direction.
        v_matrix(ndarray): Label of Merge model data in vertical direction.
        columns(list): Position of vertical lines.
        rows(list): Position of horizontal lines.
    """
    h_line, v_line = label
    rows = find_connected_line(h_line)
    columns = find_connected_line(v_line)

    h = len(h_line)
    w = len(v_line)
    mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_AREA)
    append_columns = [0] + columns + [w]
    append_rows = [0] + rows + [h]

    h_matrix = np.zeros((len(rows), len(columns) + 1))
    v_matrix = np.zeros((len(rows) + 1, len(columns)))

    for i in range(len(rows)):
        for j in range(len(append_columns) - 1):
            if np.count_nonzero(mask_img[rows[i], append_columns[j]:append_columns[j + 1]] < 10) > threshold:
                h_matrix[i, j] = 1
            else:
                h_matrix[i, j] = 0
    for i in range(len(append_rows) - 1):
        for j in range(len(columns)):
            if np.count_nonzero(mask_img[append_rows[i]:append_rows[i + 1], columns[j]] < 10) > threshold:
                v_matrix[i, j] = 1
            else:
                v_matrix[i, j] = 0
    rows = [x / h for x in rows]
    columns = [x / w for x in columns]
    return h_matrix, v_matrix, rows, columns


def find_connected_line(lines, threshold=5):
    """
    Gets center of lines.

    Args:
        lines(list): A vector indicates position of lines.
        threshold(int): Threshold for filtering lines that too close to the border.
    """
    length = len(lines)
    i = 0
    blocks = []

    def find_end(start):
        end = length - 1
        for j in range(start, length - 1):
            if lines[j + 1] == 0:
                end = j
                break
        return end

    while i < length:
        if lines[i] == 0:
            i += 1
        else:
            end = find_end(i)
            blocks.append((i, end))
            i = end + 1
    if len(blocks) > 0:
        if blocks[0][0] <= threshold:
            blocks.pop(0)
    if len(blocks) > 0:
        if (length - blocks[-1][1]) <= threshold:
            blocks.pop(-1)
    lines_position = [int((x[0] + x[1]) / 2) for x in blocks]
    return lines_position


if __name__ == "__main__":
    mask_img_dir = 'test_out_mask_dir'
    with open('test_labels.json', 'r') as f:
        dataset = json.load(f)
    merge_dict = {}
    for id, label in dataset.items():
        labels = [np.array(label['row']), np.array(label['column'])]
        mask_img = np.array(Image.open(os.path.join(mask_img_dir, id + '.jpg')))
        h_matrix, v_matrix, rows, columns = make_merge_data(labels, mask_img)
        merge_dict[id] = {'h_matrix': [list(x) for x in list(h_matrix)],
                          'v_matrix': [list(x) for x in list(v_matrix)], 'rows': rows, 'columns': columns}
    with open('merge_dict.json', 'w') as f:
        f.write(json.dumps(merge_dict))
