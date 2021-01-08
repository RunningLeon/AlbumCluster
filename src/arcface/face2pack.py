#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import cv2
import numpy as np
import argparse
import time
import pickle

import mxnet as mx
from mxnet import ndarray as nd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from utils.path_tools import *
from utils.parse_dataset import get_face_recognition_dataset

from compare import gen_same_diff_idx


def main(args):
    CHECK_EXIST(args.input, typ='d')
    dataset_labeled, dataset_classed = get_face_recognition_dataset(args.input, args.num_class, args.min_nrof_images_per_class, 
        args.max_nrof_images_per_class, sort=False, shuffle=False)

    np.random.seed(814)
    img_path_list, label_list = dataset_labeled

    nrof_image = len(img_path_list)
    nrof_class = len(set(label_list))
    print('Totall %5d classes and %5d images'%(nrof_class, nrof_image))

    same_pair_indices, diff_pair_indices = gen_same_diff_idx(label_list, nrof_pair_choose=args.nrof_pair, seed=814)
    nrof_same, nrof_diff = [len(i) for i in [same_pair_indices, diff_pair_indices]]

    same_pair_paths = [(img_path_list[p[0]], img_path_list[p[1]]) for p in same_pair_indices[:, :2]]
    diff_pair_paths = [(img_path_list[p[0]], img_path_list[p[1]]) for p in diff_pair_indices[:, :2]]

    same_pair_paths_flat = list(np.array(same_pair_paths).reshape(-1))
    diff_pair_paths_flat = list(np.array(diff_pair_paths).reshape(-1))
    
    paths_list_flat = same_pair_paths_flat + diff_pair_paths_flat
    issame_list = (nrof_same * [True]) + (nrof_diff * [False])
    print(np.sum(np.int32(issame_list)))
    nrof_path = len(paths_list_flat)
    nrof_label = len(issame_list)
    if nrof_path != 2*nrof_label:
        raise ValueError('paths_list_flat length(%d) != issame_list length(%d)'%(nrof_path, 2*nrof_label))

    data_li = []
    for idx, img_path in enumerate(paths_list_flat, start=1):
        img_bgr = cv2.imread(img_path, 1)
        img_bgr = cv2.resize(img_bgr, (args.image_width, args.image_height), interpolation=cv2.INTER_LINEAR)
        _, img_encoded = cv2.imencode('.jpg', img_bgr)
        data_li.append(img_encoded)
        ###img = mx.image.imdecode(_bin)
        ###img = nd.transpose(img, axes=(2, 0, 1))
        if not idx % 1000 or idx >= nrof_path-1:
            print('Processing %5d/%5d ...'%(idx, nrof_path))

    MAKE_EXIST(args.output, 'f')
    with open(args.output, 'wb') as f:
        pickle.dump((data_li, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saving to ', args.output)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, 
        help='Input directory that has jpg images.')
    parser.add_argument('-o', '--output', type=str, default='./datasets/val.bin', 
        help='Output bin file path.')
    parser.add_argument('-ih', '--image-height', type=int, default='112',
        help='Image height')
    parser.add_argument('-iw', '--image-width', type=int, default='112',
        help='Image width')
    parser.add_argument('-n', '--nrof-pair', type=int, default=50000,
        help='Number of same_pairs and diff_pairs, totally = n * 2')
    parser.add_argument('--num-class', type=int, default=2000, 
        help='Number of classes chosed to train classifier model.')
    parser.add_argument('--min_nrof_images_per_class', type=int, default=2,
        help='Only include classes with at least this number of images in the dataset')
    parser.add_argument('--max_nrof_images_per_class', type=int, default=1000,
        help='Choose maximum number of images from each class.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

