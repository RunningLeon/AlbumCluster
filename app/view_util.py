#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import cv2
import math
import numpy as np
import pandas as pd
import scipy.io as sio


def draw_bbox(img_bgr, bbox, is_ratio=False, label='', color=(0, 255, 0), text_size=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = img_bgr.shape[:2]
    left, top, right, bottom = [int(i) for i in bbox[:4]]
    if is_ratio:
        left, right = [int(i * width) for i in [left, right]]
        top, bottom = [int(i * height) for i in [top, bottom]]
    confidience = ''
    if len(bbox) == 5:
        confidience = '%.2f'%bbox[-1]

    cv2.rectangle(img_bgr, (left, top), (right, bottom), color, 3)
    text = '%s %s'%(label, confidience)
    pos = (max(0, left + 2 * len(text)), max(0, bottom - 2))
    cv2.rectangle(img_bgr, (pos[0], bottom - 12), (right, bottom), color, 1)
    cv2.putText(img_bgr, text, pos, font, text_size, (0, 0, 0), 2)

    return img_bgr


def draw_landmark(img_bgr, landmarks, is_ratio=False, color=(0, 255, 255), draw_order=False):
    height, width = img_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    nrof_landmark = len(landmarks)
    if not nrof_landmark: # no landmarks
        return img_bgr

    for i in range(nrof_landmark):
        pos = np.squeeze(landmarks[i, ])
        if is_ratio:
            pos = (width * pos[0], height * pos[1])
        pos = tuple([int(p) for p in pos])
        cv2.circle(img_bgr, pos , 3, color, -1)
        ### check landmarks order
        if draw_order:
            cv2.putText(img_bgr, '%d'%i,(pos[0]-2, pos[1] - 2), font, 1,(0, 0, 255),1 ,cv2.LINE_AA)

    return img_bgr


def view_image(img_bgr, name='image', wait_key=True, position_x=0, position_y=0, win_width=640, win_height=480):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, (win_width, win_height))
    cv2.moveWindow(name, position_x, position_y)
    cv2.imshow(name, img_bgr)
    if wait_key:
        parse_key()


def parse_key():
    good = True
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        sys.exit(-1)
    elif key & 0xFF == ord('d'):
        good = False
    return good