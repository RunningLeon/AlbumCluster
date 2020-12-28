import cv2
import sys
import os
import glob

import numpy as np

from .retinaface import RetinaFace


class FaceDetector(object):
    """RetinaFace detect face boxes and five facial landmarks.
    """

    def __init__(self, model_str, gpu_id=0):
        """Init instance

        Args:
            model_str ([type]): string of model checkpoint by `prefix,epoch`
            gpu_id (int, optional): GPU id. Defaults to 0.
        """
        prefix, epoch = model_str.split(',')
        epoch = int(epoch)
        self.nms_threshold = 0.8
        self._detector = RetinaFace(prefix, epoch, gpu_id, 'net3')
        self.scales = (1024, 1980)

    def predict(self, img: np.ndarray)->tuple:
        """
        predict

        Args:
            img (np.ndarray): [description]

        Returns:
            tuple: [description]
        """
        scales = self._cal_scales(img)
        faces, landmarks = self._detector.detect(
            img, self.nms_threshold, scales=scales, do_flip=False)
        if faces is not None:
            faces = faces.astype(np.int32)
        if landmarks is not None:
            landmarks = landmarks.astype(np.int32)
        return faces, landmarks

    def __call__(self, img: np.ndarray):
        return self.predict(img)

    def _cal_scales(self, img):
        im_shape = img.shape
        target_size = self.scales[0]
        max_size = self.scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        #im_scale = 1.0
        #if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        # print('im_scale', im_scale)

        scales = [im_scale]
        return scales

    def show_result(self, img, faces, landmarks):
        if faces is not None:
            # print('find', faces.shape[0], 'faces')
            for i in range(faces.shape[0]):
                box = faces[i]
                color = (0, 0, 255)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color,
                              2)
                if landmarks is not None:
                    landmark5 = landmarks[i]
                    #print(landmark.shape)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1,
                                   color, 2)
        return img

if __name__ == '__main__':
    image_path = 'images/friends'