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


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.path_tools import *
from utils.visualize import plot_roc
from utils.parse_dataset import get_face_recognition_dataset

class Timer(object):
    """docstring for Timer"""
    def __init__(self):
        super(Timer, self).__init__()
        self._start_time=None
        self._last_time=None


    def tic(self):
        this_time = time.time()
   

def cal_distance_matrix(X):
    out = 1 - metrics.pairwise_distances(X, metric="cosine")
    out = np.arccos(out)/np.pi
    return out


def cal_distance(embeddings1, embeddings2, distance_metric=1):
    from sklearn.metrics.pairwise import cosine_distances
    dist = cosine_distances(embeddings1, embeddings2)
    print(dist)
    return dist
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        min_dist = np.min(similarity)
        dist = 1 - similarity
        # dist = np.arccos(similarity) / np.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    return dist


def read_all_feats(dataset, img_ext, feat_ext, classed_dataset_flag=True, return_img_path=False):
    label_li = []
    feat_li = []
    img_path_li = []

    if classed_dataset_flag:
        nrof_classes = len(dataset)
        for idx, img_cls_i in enumerate(dataset):
            nrof_sample = len(img_cls_i.image_paths)
            for ii, img_path_i in enumerate(img_cls_i.image_paths):
                pi = img_path_i[:-len(img_ext)] + feat_ext
                if not os.path.exists(pi):
                    print('Not exists ', pi)
                    continue
                feat = np.load(pi)
                feat_li.append(feat)
                label_li.append(idx)
                img_path_li.append(img_path_i)
                # print('%5d/%5d, %5d/%5d'%(idx, nrof_classes, ii, nrof_sample))
    else:
        img_path_list, label_list = dataset
        for idx, (img_path, label) in enumerate(zip(img_path_list, label_list), start=1):
            feat_path = img_path[:-len(img_ext)] + feat_ext
            if not os.path.exists(feat_path):
                print('Feat file not exists: ', feat_path)
                continue
            feat = np.load(feat_path)
            feat_li.append(feat)
            label_li.append(label)
            img_path_li.append(img_path)
    if return_img_path:
        return feat_li, label_li, img_path_li

    return feat_li, label_li


def gen_same_diff_idx(label_li, nrof_pair_choose=None, seed=814, nrof_max_image_per_person=25):
    labels = list(set(label_li))
    dic ={}
    for l in labels:
        dic[l] = [idx for idx, cls_id in enumerate(label_li) if cls_id == l]

    same_pairs = []
    tic = time.time()
    for cls_id, indices_li in dic.items():
        comb = list(combinations(indices_li, 2))
        nrof_comb = len(comb)
        pairs = [[idx[0], idx[1], cls_id, cls_id] for idx in comb]
        same_pairs += pairs
    diff_sec = time.time() - tic
    print('Finshing calucate same pairs indices, time=%.6fs'%(diff_sec))
    
    diff_pairs = []
    nrof_class = len(dic)
    time_start = time.time()
    for idx_i, (cls_id_i, indices_i) in enumerate(dic.items(), start=1):
        tic = time.time()
        for idx_j, (cls_id_j, indices_j) in enumerate(dic.items(), start=1):
            if cls_id_i >= cls_id_j:
                continue
            comb = list(itertools.product(indices_i, indices_j))
            pairs = [[idx[0], idx[1], cls_id_i, cls_id_j] for idx in comb]
            diff_pairs += pairs
            toc = time.time()
            diff_sec = toc - tic 
            total_sec = toc - time_start
            # print('Processing outer %5d/%5d, inner %5d/%5d, time_per_inner_loop=%.6fs, totall_time_till_now=%.6fs'%(idx_i, nrof_class, idx_j, nrof_class, diff_sec, total_sec))

    nrof_same_pair = len(same_pairs)
    nrof_diff_pair = len(diff_pairs)
    print('nrof_same_pair=%d, nrof_diff_pair=%2d'%(nrof_same_pair, nrof_diff_pair))
    np.random.seed(seed)
    np.random.shuffle(same_pairs)
    np.random.shuffle(diff_pairs)

    if nrof_pair_choose is None:
        nrof_pair_choose = nrof_same_pair
    nrof_pair_choose = min(nrof_pair_choose, nrof_same_pair)

    same_pairs, diff_pairs = [np.asarray(i).reshape(-1, 4) for i in [same_pairs, diff_pairs]]
    same_pairs = same_pairs[:nrof_pair_choose, :]
    diff_pairs = diff_pairs[:nrof_pair_choose, :]
    return same_pairs, diff_pairs


def cal_all_dist(feat_li, same_pairs, diff_pairs, distance_metric=1, batch=1000, eval_same=True):

    nrof_same, nrof_diff = [len(i) for i in [same_pairs, diff_pairs]]
    if eval_same:
        issame_label, isdiff_label = 1, 0
    else:
        issame_label, isdiff_label = 0, 1

    left_indices = list(same_pairs[:, 0]) + list(diff_pairs[:, 0])
    right_indices = list(same_pairs[:, 1]) + list(diff_pairs[:, 1])
    nrof_all = len(left_indices)

    dist_li = []
    nrof_batch = nrof_all // batch
    nrof_batch = nrof_batch if nrof_batch > 0 else 1
    for idx in range(nrof_batch):
        start_idx = batch * idx
        end_idx = min(nrof_all, batch * (idx+1))
        left_feats = np.array([feat_li[i]for i in left_indices[start_idx:end_idx]]).reshape(-1, 512)
        right_feats = np.array([feat_li[i]for i in right_indices[start_idx:end_idx]]).reshape(-1, 512)
        dists = list(cal_distance(left_feats, right_feats, distance_metric=1))
        min_dist, max_dist = np.min(dists), np.max(dists)
        # print('min_dist=%.3f, max_dist=%.3f'%(min_dist, max_dist))
        dist_li += dists

    nrof_dist = len(dist_li)
    if nrof_dist != nrof_all:
        raise ValueError('nrof_dist=%d != nrof_all=%d'%(nrof_dist, nrof_all))

    max_idx = np.argmax(dist_li)
    max_dist = dist_li[max_idx]

    issame_li = nrof_same * [issame_label] + nrof_diff * [isdiff_label]
    max_label = issame_li[max_idx]
    # print('max_idx=%s, max_dist=%.3f, max_label=%s'%(max_idx, max_dist, max_label))

    return np.array(dist_li), np.array(issame_li)


def gen_roc(dist_li, issame_li, thresholds, eval_same=True):

    precision_li, recall_li = [], []
    nrof_th = len(thresholds)
    for idx, th in enumerate(thresholds, start=1):
        if eval_same:
            pred_same = np.int32(dist_li < th)
        else:
            pred_same = np.int32(dist_li > th)

        pre = precision_score(issame_li, pred_same)
        rec = recall_score(issame_li, pred_same)
        precision_li.append(pre)
        recall_li.append(rec)
        # print('Cal %5d/%5d pre=%.2f, rec=%.2f'%(idx, nrof_th, pre, rec))
    return precision_li, recall_li

def write_txt(index_pairs, output_txt, img_path_li, root_dir):
    with open(output_txt, 'w') as f:
        for i, j, _, _ in index_pairs:
            pi, pj = [img_path_li[x].replace(root_dir, '').lstrip('/') for x in [i, j]]
            f.write('%s %s\n'%(pi, pj))
    print('Saving to ', output_txt)


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

    feat_li, label_li, img_path_li = read_all_feats(dataset_classed, args.img_ext, args.feat_ext, return_img_path=True)
    nrof_feat = len(feat_li)
    print('Totally %5d feats,'%(nrof_feat))
    same_pairs, diff_pairs = gen_same_diff_idx(label_li, nrof_pair_choose=args.nrof_pair)
    print('Finshing generate pairs, nrof_same_pair=%5d, nrof_diff_pair=%5d'%(len(same_pairs), len(diff_pairs)))
    if args.write_sample:
        same_txt, diff_txt = [os.path.join(args.output, '%s-pair.txt'%_) for _ in ['same', 'diff']]
        write_txt(same_pairs, same_txt, img_path_li, args.input)
        write_txt(diff_pairs, diff_txt, img_path_li, args.input)

    eval_choices = (True, False)
    for eval_type in eval_choices:
        dist_li, issame_li = cal_all_dist(feat_li, same_pairs, diff_pairs, eval_same=eval_type)
        print('Finshing cal dist')
        
        min_dist, max_dist = np.min(dist_li), np.max(dist_li)
        threshold_li = np.arange(min_dist + 0.01, max_dist + 0.01, 0.01)
        precision_li, recall_li = gen_roc(dist_li, issame_li, threshold_li, eval_same=eval_type)

        if eval_type:
            title = 'Similarity'
            filename = 'is_same-pair%d-cls%d-min%d-max%d.png'%(args.nrof_pair, nrof_class, args.min_nrof_images_per_class, args.max_nrof_images_per_class)
        else:
            title = 'Difference'
            filename = 'is_diff-pair%d-cls%d-min%d-max%d.png'%(args.nrof_pair, nrof_class, args.min_nrof_images_per_class, args.max_nrof_images_per_class)

        output_roc_path = os.path.join(args.output, filename)
        MAKE_EXIST(output_roc_path, 'f')

        df = plot_roc(threshold_li, precision_li, recall_li, save_path=output_roc_path, block=args.visualize, title=title)
        print(df.head())


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, 
        help='Input directory that has jpg images.')
    parser.add_argument('-o', '--output', type=str, default='', 
        help='Directory to save plotted images.')
    parser.add_argument('--feat-ext', default='feat.npy', 
        help='File extenstion of output feat file.')
    parser.add_argument('--img-ext', default='jpg',
        help='images file extenstions.')
    parser.add_argument('--num-class', type=int, default=100, 
        help='Number of classes chosed to train classifier model.')
    parser.add_argument('--min_nrof_images_per_class', type=int, default=2,
        help='Only include classes with at least this number of images in the dataset')
    parser.add_argument('--max_nrof_images_per_class', type=int, default=200,
        help='Choose maximum number of images from each class.')
    parser.add_argument('-n', '--nrof-pair', type=int, default=50000,
        help='Number of same_pairs and diff_pairs, totally = n * 2')
    parser.add_argument('-v', '--visualize', action='store_true', 
        help='Whether to visualize plotted image while running.')
    parser.add_argument('--write-sample', action='store_true',
        help='Write samples name to txt')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

