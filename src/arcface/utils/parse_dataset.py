#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import cv2
import math
import json
import numpy as np
import scipy.io as sio
from scipy import linalg
from collections import defaultdict

from utils.path_tools import *


class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)


def get_face_recognition_dataset(path, num_class=50, min_num_imgs=10, max_num_imgs=100, sort=False, shuffle=False, img_type=['jpg'], seed=814):
    dataset_classed = []
    dataset_labeled = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    if sort:
        classes.sort()
    elif shuffle:
        np.random.seed(seed)
        np.random.shuffle(classes)
    else:
        pass

    nrof_classes = len(classes)
    class_counter = 0
    label_list, img_path_list = [], []
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = []
        image_paths_all_type = [glob_file(facedir, typ, recursive=True) for typ in img_type]
        for paths in image_paths_all_type:
            image_paths += paths
        # if len(image_paths) < min_num_imgs:
        #     continue
        image_paths = image_paths[:max_num_imgs]
        num_img = len(image_paths)
        label_list += num_img * [class_counter]
        img_path_list += image_paths
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(image_paths)
        dataset_classed.append(ImageClass(class_name, image_paths))
        class_counter += 1
        if class_counter == num_class:
            break
    dataset_labeled = (img_path_list, label_list)
    return dataset_labeled, dataset_classed



