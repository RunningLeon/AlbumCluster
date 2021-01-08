#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import cv2
import glob
import numpy as np
import argparse
import fnmatch
import shutil
import time


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from utils.path_tools import *

from utils.parse_dataset import get_face_recognition_dataset


def main(args):
    CHECK_EXIST(args.input, typ='d')
    dataset_labeled, dataset_classed = get_face_recognition_dataset(args.input, args.num_class, args.min_nrof_images_per_class, args.max_nrof_images_per_class, sort=False, shuffle=True)

    np.random.seed(814)
    img_path_list, label_list = dataset_labeled
    nrof_class = len(set(label_list))
    nrof_image = len(img_path_list)

    print('Totall %5d classes and %5d images'%(nrof_class, nrof_image))
    if nrof_class != args.num_class:
        raise ValueError('Found %d classes, not required %d classes in directory: %s'%(nrof_class, args.num_class, args.input))

    indices = [i for i in range(nrof_image)]
    np.random.shuffle(indices)
    MAKE_EXIST(args.output, 'f')
    with open(args.output, 'w') as f:
        for idx in indices:
            img_path, label = img_path_list[idx], label_list[idx]
            line = '1\t%s\t%d\n'%(img_path, label)
            f.write(line)
    print('Saving to ', args.output)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, 
        help='Input directory that has jpg images.')
    parser.add_argument('-o', '--output', type=str, default='./datasets/train.lst', 
        help='Output train.lst ')
    parser.add_argument('--num-class', type=int, default=2000, 
        help='Number of classes chosed to train classifier model.')
    parser.add_argument('--min_nrof_images_per_class', type=int, default=1,
        help='Only include classes with at least this number of images in the dataset')
    parser.add_argument('--max_nrof_images_per_class', type=int, default=1000,
        help='Choose maximum number of images from each class.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

