#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pprint import pprint
import logging
from datetime import datetime
import time
import math
import os
from functools import partial
from var_dump import var_dump
import json
import matplotlib.pyplot as plt

import pandas as pd
from PIL import Image
import math

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 300)

OUT_DIR = 'ds/step1/output/'

logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s %(name)-1s %(levelname)-3s %(message)s',
    # format='%(levelname)-3s %(message)s',
    format='%(levelname)-3s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d %H:%M:%S',
    handlers=[
        # logging.FileHandler(FILENAME_LOGGING),
        logging.StreamHandler()
    ]
)

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]
YELLOW = [0, 243, 255]
PINK = [189, 0, 255]
PURPLE = [255, 0, 205]

RGB_RED=[255, 0, 0]


def ins(v):
    print("ins>>>")
    print('>dir:')
    print(dir(v))
    print('>type:')
    print(type(v))
    print('>print:')
    print(v)
    print("ins<<<")







def get_box_resized(origin, ratio):
    box = origin.copy()
    for i in range(len(box)):
        box[i] = int(box[i] * ratio)
    return box


def get_shape_resized(shape, ratio):
    width = int(shape[1] * ratio)
    height = int(shape[0] * ratio)
    return width, height


def get_cutted_filename(name):
    part = name.split('_')
    return part[0] + '_' + part[1] + '_' + part[3]


def random_color():
    color = np.random.randint(0, 255, size=(3,))
    return int(color[0]), int(color[1]), int(color[2])


def flaw_name(num):
    map_type = {
        0: "0背景",
        1: "1边异常",
        2: "2角异常",
        3: "3白色点瑕疵",
        4: "4浅色块瑕疵",
        5: "5深色点块瑕疵",
        6: "6光圈瑕疵"
    }
    map_type1 = {
        0: "0 background",
        1: "1 edge weird",
        2: "2 corner weird",
        3: "3 white spot",
        4: "4 light block",
        5: "5 dark block",
        6: "6 halo"
    }
    return map_type1[num]


def flaw_color(num):
    map_color = {
        1: [255, 0, 0],
        0: [255, 165, 0],
        2: [255, 255, 0],
        3: [0, 255, 0],
        4: [0, 0, 255],
        5: [75, 0, 130],
        6: [238, 130, 238]
    }
    return map_color[num]


def convertToRelativeValues(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    cx = (box[1] + box[0]) / 2.0
    cy = (box[3] + box[2]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx * dw
    y = cy * dh
    w = w * dw
    h = h * dh
    # x,y => (bounding_box_center)/width_of_the_image
    # w => bounding_box_width / width_of_the_image
    # h => bounding_box_height / height_of_the_image
    return (x, y, w, h)


def convertToAbsoluteValues(size, box):
    # w_box = round(size[0] * box[2])
    # h_box = round(size[1] * box[3])
    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])
    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
    return (xIn, yIn, xEnd, yEnd)


def aaa(t):
    x = (t['bbox'][0] + t['bbox'][2]) / 2 / t['image_width']
    y = (t['bbox'][1] + t['bbox'][3]) / 2 / t['image_height']
    w = (t['bbox'][2] - t['bbox'][0]) / t['image_width']
    h = (t['bbox'][3] - t['bbox'][1]) / t['image_height']
    return x, y, w, h


def xywh(size, box):
    x = box[0]
    y = box[1]
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x, y, w, h


def draw_img(df, dir_name, filename):
    file_full_path = os.path.join(dir_name, filename)
    img_labels = df[df.name == filename]
    logging.info(img_labels)
    resize_ratio = 1

    img = cv2.imread(file_full_path)
    shape_resized = get_shape_resized(img.shape, resize_ratio)
    img_resized = cv2.resize(img, shape_resized)
    print(img.shape)
    print(img_resized.shape)

    i = 0
    for index, row in img_labels.iterrows():
        bbox_resized = get_box_resized(row['bbox'], resize_ratio)
        print(row['bbox'])
        print(bbox_resized)

        x, y, w, h = xywh(shape_resized, bbox_resized)
        category = row['category']
        color = flaw_color(category)

        cv2.rectangle(img_resized, (x, y, w, h), color, 1)
        cv2.putText(img_resized, flaw_name(category), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite(OUT_DIR + row['name'], img_resized)

    # cv2.imshow(get_cutted_filename(filename) + '_' + str(i), img_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # exit()


def draw_cross(img, loc, color=BLUE, line_length=10):
    (x, y) = loc
    cv2.line(img, (x, y - line_length), (x, y + line_length), color, 1)
    cv2.line(img, (x - line_length, y), (x + line_length, y), color, 1)


# 85 ,70 ,90
THRESHOLD_OF_BINARY = 90
# 10
CONTOUR_MINIMAL_AREA = 10
# 420
CENTER_CIRCULE_RADIUS = 205
# 0
DEBUG_IMSHOW = 0

import time

START_TIME = time.time()


def hit():
    print("hitted --- %s seconds ---" % (time.time() - START_TIME))




def get_region(img):
    ## Threshold in grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_three = cv2.merge([gray, gray, gray])


    # return gray_three



    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    exit()

    retval, threshed = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    ## Find wathc region by counting the projector
    h, w = img.shape[:2]
    x = np.sum(threshed, axis=0)
    y = np.sum(threshed, axis=1)
    yy = np.nonzero(y > (w / 5 * 255))[0]
    xx = np.nonzero(x > (h / 5 * 255))[0]
    region = img[yy[0]:yy[-1], xx[0]:xx[-1]]
    cv2.imshow("region.png" + get_date(), region) if DEBUG_IMSHOW else 1

    return region




def get_date():
	return str(int(round(time.time() * 1000)))[-4:]

DEBUG_IMSHOW = 1


def see(img):
    see1(img)


def see0():
    see1(img)
    see2(img)
    see3(img)

# PIL call sys
def see1(img):
    print('>>> see1, or see, PIL call sys ')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img2)
    im_pil.show()

# matploit
def see2(img):
    print('>>> see2 ,matploit ')
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    plt.xticks([]), plt.yticks([])
    plt.show()

# opencv
def see3(img):
    print('>>> see3 ,opencv ')
    cv2.imshow( get_date() , img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_date():
	return str(int(round(time.time() * 1000)))[-4:]

def get_contours(region):
    ## Change to LAB space
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    imglab = np.hstack((l, a, b))
    plt.imshow(imglab)
    plt.xticks([]), plt.yticks([])
    plt.show()
    exit()

    cv2.imshow("region_lab.png" + get_date(), imglab) if DEBUG_IMSHOW else 1

    ## normalized the a channel to all dynamic range
    na = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow("a_normalized.png" + get_date(), na) if DEBUG_IMSHOW else 1

    ## Threshold to binary
    retval, threshed = cv2.threshold(na, thresh=THRESHOLD_OF_BINARY, maxval=255, type=cv2.THRESH_BINARY)

    ## Do morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
    res = np.hstack((threshed, opened))
    cv2.imshow("a_binary.png" + get_date(), res) if DEBUG_IMSHOW else 1

    ## Find contours
    contours = cv2.findContours(opened, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[-2]

    return contours
