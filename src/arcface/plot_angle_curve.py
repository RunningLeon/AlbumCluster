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
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import json
from scipy import interpolate

import argparse
from collections import defaultdict
import fnmatch
import shutil
from sklearn import metrics
from scipy.spatial.distance import cosine
from sklearn.metrics import recall_score, precision_score, accuracy_score
import functools
import itertools
import multiprocessing
import pickle
from itertools import combinations, permutations
import time


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.path_tools import *
from utils.visualize import plot_roc
from utils.parse_dataset import get_face_recognition_dataset

from compare import *


def cal_angle(embeddings1, embeddings2, is_deg=True, dtype=np.float32):
    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
    norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
    similarity = dot / norm
    angle = np.arccos(similarity)
    if is_deg:
        angle *= (180.0 / np.pi)
    angle = angle.astype(dtype)
    return angle


def cal_feat_angle(feat_li, index_pairs, batch=1000, is_deg=True):
    nrof_pair = len(index_pairs)

    left_indices = list(index_pairs[:, 0])
    right_indices = list(index_pairs[:, 1])

    angle_li = []
    nrof_batch = nrof_pair // batch
    nrof_batch = nrof_batch if nrof_batch > 0 else 1
    for idx in range(nrof_batch):
        start_idx = batch * idx
        end_idx = min(nrof_pair, batch * (idx+1))
        left_feats = np.array([feat_li[i]for i in left_indices[start_idx:end_idx]]).reshape(-1, 512)
        right_feats = np.array([feat_li[i]for i in right_indices[start_idx:end_idx]]).reshape(-1, 512)
        angles = list(cal_angle(left_feats, right_feats, is_deg))
        min_angle, max_angle = np.min(angles), np.max(angles)
        # print('min_angle=%.3f, max_angle=%.3f'%(min_angle, max_angle))
        angle_li += angles

    return angle_li


def plot_hist(data_li, label_li, xlabel='feat_angle', ylabel='Number', bins=None, save_path='', title='', block=True):
    plt.rcParams['figure.figsize'] = (20, 16)
    plt.rcParams['font.size'] = 22
    for data, label in zip(data_li, label_li):
        x_min, x_max = min(data), max(data)
        if bins is None:
            bins = int(x_max - x_min) + 1
        plt.hist(data, bins=bins, label=label, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    x_min, x_max = int(np.min(np.array(data_li))), int(np.max(np.array(data_li))) + 1

    x_space, y_space = 2, 500
    xticks = [i for i in range(x_min, x_max, x_space) ]
    plt.xlim(x_min, x_max)
    plt.axes().xaxis.set_minor_locator(MultipleLocator(x_space))
    plt.axes().xaxis.set_tick_params(which='minor', right = 'off')
    plt.axes().yaxis.set_minor_locator(MultipleLocator(500))
    plt.axes().yaxis.set_tick_params(which='minor', right = 'off')
    # plt.ylim(, 1.01)
    plt.xticks(xticks, fontsize=20, rotation=90)
    plt.yticks(fontsize=20)
    plt.grid(which='both')
    plt.title('Hist of %s'%(title))
    plt.legend()
    if save_path:
        MAKE_EXIST(save_path)
        plt.savefig(save_path)
        print('Saving to ', save_path)

    plt.show(block=block)


def main(args):
    CHECK_EXIST(args.input, typ='d')
    if not args.output:
        args.output = args.input

    dataset_labeled, dataset_classed = get_face_recognition_dataset(args.input, args.num_class, args.min_nrof_images_per_class, 
        args.max_nrof_images_per_class, sort=False, shuffle=True, img_type=[args.img_ext])

    nrof_class = len(dataset_classed)
    nrof_image = sum([len(x.image_paths) for x in dataset_classed])
    print('Num classes: %4d, num images: %4d' % (nrof_class, nrof_image))
    if nrof_class == 0 or nrof_image == 0 :
        print('nrof_class != nrof_image')
        return

    feat_li, label_li = read_all_feats(dataset_classed, args.img_ext, args.feat_ext)
    nrof_feat = len(feat_li)
    print('Totally %5d feats,'%(nrof_feat))
    same_pairs, diff_pairs = gen_same_diff_idx(label_li, nrof_pair_choose=args.nrof_pair)
    print('Finshing generate pairs, nrof_same_pair=%5d, nrof_diff_pair=%5d'%(len(same_pairs), len(diff_pairs)))
    if args.nrof_pair is None:
        args.nrof_pair = len(same_pairs)
    same_pair_angles, diff_pair_angles = cal_feat_angle(feat_li, same_pairs), cal_feat_angle(feat_li, diff_pairs)
    filename = 'feat-angle-pair%d-cls%d-min%d-max%d.png'%(args.nrof_pair, nrof_class, args.min_nrof_images_per_class, args.max_nrof_images_per_class)

    output_path = os.path.join(args.output, filename)
    MAKE_EXIST(output_path, 'f')
    plot_hist([same_pair_angles, diff_pair_angles], ['same', 'diff'], bins=None, save_path=output_path, title= args.title +' feat-pair angle in deg', block=args.visualize)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, 
        help='Input directory that has jpg images.')
    parser.add_argument('-o', '--output', type=str, default='', 
        help='Directory to save plotted images.')
    parser.add_argument('--feat-ext', default='.feat.npy', 
        help='File extenstion of output feat file.')
    parser.add_argument('--img-ext', default='jpg',
        help='images file extenstions.')
    parser.add_argument('--num-class', type=int, default=1000, 
        help='Number of classes chosed to train classifier model.')
    parser.add_argument('-min', '--min_nrof_images_per_class', type=int, default=1,
        help='Only include classes with at least this number of images in the dataset')
    parser.add_argument('-max', '--max_nrof_images_per_class', type=int, default=200,
        help='Choose maximum number of images from each class.')
    parser.add_argument('-n', '--nrof-pair', type=int, default=None,
        help='Number of same_pairs and diff_pairs, totally = n * 2')
    parser.add_argument('-v', '--visualize', action='store_true', 
        help='Whether to visualize plotted image while running.')
    parser.add_argument('--title', default='',
        help='Title to add ')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

