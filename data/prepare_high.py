# coding: utf-8

import glob
import os
from scipy.ndimage.filters import gaussian_filter
import cv2
import h5py
import numpy as np
import torch
from PIL import Image
import argparse
import json
parser = argparse.ArgumentParser(description='CLTR')
parser.add_argument('--data_path', type=str, default='D:/high_for_PET',
                    help='the data path of high')

args = parser.parse_args()
root = args.data_path

train = root + '/train_data/images/'
# val = root + '/val/images/'
test = root + '/test_data/images/'

'''mkdir directories'''
if not os.path.exists(train.replace('images', 'images_2048')):
    os.makedirs(train.replace('images', 'images_2048'))
if not os.path.exists(train.replace('images', 'gt_detr_map_2048')):
    os.makedirs(train.replace('images', 'gt_detr_map_2048'))
if not os.path.exists(train.replace('images', 'gt_show')):
    os.makedirs(train.replace('images', 'gt_show'))

# if not os.path.exists(val.replace('images', 'images_2048')):
#     os.makedirs(val.replace('images', 'images_2048'))
# if not os.path.exists(val.replace('images', 'gt_detr_map_2048')):
#     os.makedirs(val.replace('images', 'gt_detr_map_2048'))
# if not os.path.exists(val.replace('images', 'gt_show')):
#     os.makedirs(val.replace('images', 'gt_show'))

if not os.path.exists(test.replace('images', 'images_2048')):
    os.makedirs(test.replace('images', 'images_2048'))
if not os.path.exists(test.replace('images', 'gt_detr_map_2048')):
    os.makedirs(test.replace('images', 'gt_detr_map_2048'))
if not os.path.exists(test.replace('images', 'gt_show')):
    os.makedirs(test.replace('images', 'gt_show'))

path_sets = [train, test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()

for img_path in img_paths:
    print(os.path.exists(img_path))
    img = cv2.imread(img_path)
    Img_data_pil = Image.open(img_path).convert('RGB')

    print(img_path)
    rate = 1
    rate1 = 1
    rate2 = 1
    if img.shape[1] >= img.shape[0] and img.shape[1] >= 2048: # 原始文件中shapa[1] 为宽度，[0] 为高度 ,shape为[h,w,c]
        rate1 = 2048.0 / img.shape[1]
    elif img.shape[0] >= img.shape[1] and img.shape[0] >= 2048:
        rate1 = 2048.0 / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=rate1, fy=rate1, interpolation=cv2.INTER_CUBIC)
    Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]) ,Image.Resampling.LANCZOS) # 使用 Image.Resampling.LANCZOS 来替代 Image.ANTIALIAS

    min_shape = 512.0
    if img.shape[1] <= img.shape[0] and img.shape[1] <= min_shape:
        rate2 = min_shape / img.shape[1]
    elif img.shape[0] <= img.shape[1] and img.shape[0] <= min_shape:
        rate2 = min_shape / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=rate2, fy=rate2, interpolation=cv2.INTER_CUBIC)
    Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]), Image.Resampling.LANCZOS)

    rate = rate1 * rate2

    k = np.zeros((img.shape[0], img.shape[1]))
    # gt_file = np.loadtxt(img_path.replace('images', 'ground-truth').replace('jpg', 'json'))
    gt_path = img_path.replace('images', 'ground-truth').replace('jpg', 'json')
    with open(gt_path, 'r') as f:
        mat = json.load(f)
    points = []
    for item in mat['shapes']:
        points.extend(item['points'])
    gt_file = np.array(points)# gt_file[1]是高度,
    # gt_file = gt_file[:, [1, 0]]# 交换，[0]是高，[1]是宽
    fname = img_path.split('/')[-1]

    try:
        y = gt_file[:, 0] * rate #实际上是宽度
        x = gt_file[:, 1] * rate #实际上是高度
        for i in range(0, len(x)):
            if int(x[i]) < img.shape[0] and int(y[i]) < img.shape[1]:
                k[int(x[i]), int(y[i])] += 1 # k[0]高度，K[1]宽度
    except Exception:
        try:
            y = gt_file[0] * rate
            x = gt_file[1] * rate

            for i in range(0, 1):
                if int(x) < img.shape[0] and int(y) < img.shape[1]:
                    k[int(x), int(y)] += 1
        except Exception:
            ''' this image without person'''
            k = np.zeros((img.shape[0], img.shape[1]))

    kpoint = k.copy()
    kpoint = kpoint.astype(np.uint8)

    with h5py.File(img_path.replace('images', 'gt_detr_map_2048').replace('jpg', 'h5'), 'w') as hf:
        hf['kpoint'] = kpoint
        hf['image'] = Img_data_pil

    cv2.imwrite(img_path.replace('images', 'images_2048'), img)

    # for gt_show
    # k show on the img
    # 假设 k 和 img 是已知的
    # k 是一个大小为 (h, w) 的二值图， img 是一个大小为 (h, w, c) 的图像

    # 创建 img_show 副本，避免修改原始图像
    img_show = img.copy()

    # 遍历 k 中的每个点，如果 k 中对应位置的值为 1，则在 img_show 上标记该点
    for y in range(k.shape[0]):
        for x in range(k.shape[1]):
            if k[y, x] == 1:
                # 在 img_show 上标记该点，可以选择颜色，示例用红色
                img_show = cv2.circle(img_show, (x, y), 2, (0, 0, 255), -1)  # 5 是圆的半径，(0, 0, 255) 是红色

    # 保存 img_show 图像
    cv2.imwrite(img_path.replace('images', 'gt_show'), img_show)
print("end")
