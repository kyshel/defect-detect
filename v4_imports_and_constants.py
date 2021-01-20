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
from time import sleep
from tqdm import tqdm







pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 300)





WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]
YELLOW = [0, 243, 255]
PINK = [189, 0, 255]
PURPLE = [255, 0, 205]
RGB_RED=[255, 0, 0]

CROP_SIZE = 512
OUT_DIR = 'ds/step1/output/'

# same as yolo train dir
CROPS_DIR = 'ds/v4/images/train/'

# same as yolo label dir
TXT_DIR = 'ds/v4/labels/train/'








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