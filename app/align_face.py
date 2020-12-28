#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Sequence, Tuple, Dict

import cv2
import numpy as np
from skimage import transform as trans


def calibrate_norm_landmark(landmarks: np.ndarray,
                            img_shape: tuple,
                            face_height=112,
                            face_width=112,
                            is_ratio=False):
    """calibrate and norm landmark to [0,1]

    Args:
        landmarks ([type]): [description]
        img_shape ([type]): [description]
        face_height (int, optional): [description]. Defaults to 112.
        face_width (int, optional): [description]. Defaults to 112.
        is_ratio (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    landmarks_out = landmarks.copy()

    height, width = img_shape[:2]
    if is_ratio:
        landmarks_out[:, 0] *= width
        landmarks_out[:, 1] *= height
    ratio_x = face_width / float(width)
    ratio_y = face_height / float(height)
    landmarks_out[:, 0] *= ratio_x
    landmarks_out[:, 1] *= ratio_y
    return landmarks_out


def align_face(face_img,
               landmarks,
               mean_landmarks=None,
               method=0,
               img_size=112):
    """align cropped face image with landmarks and reference landmarks.

    Args:
        face_img (np.ndarray): cropped face image
        landmarks (np.ndarray): [description]
        mean_landmarks (np.ndarray, optional): [description]. Defaults to None.
        method (int, optional): [description]. Defaults to 0.
        img_size (int, optional): [description]. Defaults to 112.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if mean_landmarks is None:
        mean_landmarks = np.array(
            [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
             [33.5493, 92.3655], [62.7299, 92.2041]],
            dtype=np.float32)
    method_choices = (0, 1, 2)
    if method not in method_choices:
        raise ValueError('Input argument method(%s) must be in choices=%s ' %
                         (method, method_choices))

    src = mean_landmarks.copy()
    dst = landmarks.copy()
    ratio = cal_face_ratio(src, dst)
    # ### update src
    src[:, 0] *= ratio
    src[:, 1] *= ratio

    src[:, 0] += (dst[2, 0] - src[2, 0])
    src[:, 1] += (dst[2, 1] - src[2, 1])

    height, width, _ = face_img.shape

    if method == 0:
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        mat = tform.params[:2, :]
    elif method == 1:
        mat, _ = cv2.estimateAffinePartial2D(
            dst.reshape(1, -1, 2), src.reshape(1, -1, 2), False)
    else:
        mat, _ = cv2.estimateAffinePartial2D(
            dst.reshape(1, -1, 2), src.reshape(1, -1, 2), True)

    # if norm_affine is True:
    four_pts_boundary = np.array([[0, 0, 1], [width - 1, 0, 1],
                                  [width - 1, height - 1, 1],
                                  [0, height - 1, 1]]).T
    four_pts_affined = np.dot(mat, four_pts_boundary).T
    min_xy = np.min(four_pts_affined, axis=0)
    max_xy = np.max(four_pts_affined, axis=0)
    new_width, new_height = [int(i) for i in max_xy - min_xy]

    mat[1, 2] -= min_xy[1]
    mat[0, 2] -= min_xy[0]

    warped = cv2.warpAffine(
        face_img.copy(),
        mat, (new_width, new_height),
        borderValue=0.0,
        flags=cv2.INTER_LINEAR)

    if img_size is None:
        warped = cv2.resize(
            warped, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return warped


def cal_face_ratio(src_landmarks, dest_landmarks):
    """caculate mean ratio between source and destination landmarks.

    Args:
        src_landmarks ([type]): [description]
        dest_landmarks ([type]): [description]

    Returns:
        [type]: [description]
    """
    ratios = []
    num_points = src_landmarks.reshape(-1, 2).shape[0]
    for i in range(num_points - 1):
        for j in range(i + 1, num_points):
            ratio = np.linalg.norm(dest_landmarks[j, :] - dest_landmarks[i, :]) / \
                    np.linalg.norm(src_landmarks[j, :] - src_landmarks[i, :])
            ratios.append(ratio)
    ratio_mean = np.mean(ratios)
    return ratio_mean


def expand_bbox(img_width: int,
                     img_height: int,
                     bbox: Sequence[int],
                     margin=44) -> Sequence[int]:
    """
    expand_bbox

    Args:
        img_width (int): [description]
        img_height (int): [description]
        bbox (Sequence[int]): [description]
        margin (int, optional): [description]. Defaults to 44.

    Returns:
        Sequence[int]: [description]
    """
    bbox = np.squeeze(bbox).astype(np.int32)
    bbox[0] = np.maximum(bbox[0] - margin / 2, 0)
    bbox[1] = np.maximum(bbox[1] - margin / 2, 0)
    bbox[2] = np.minimum(bbox[2] + margin / 2, img_width)
    bbox[3] = np.minimum(bbox[3] + margin / 2, img_height)
    return bbox


def crop_image(image: np.ndarray, bbox: Sequence[int])->np.ndarray:
    """
    crop_image 

    Args:
        image (np.ndarray): [description]
        bbox (Sequence[int]): [description]

    Returns:
        np.ndarray: [description]
    """
    cropped = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    return cropped