import cv2

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
        scales = [im_scale]
        return scales


if __name__ == '__main__':
    import os
    image_path = 'images/friends.jpg'
    assert os.path.exists(image_path), f'File not exists: {image_path}'
    from app.model_config import cfg
    from app.view_util import view_image, draw_bbox, draw_landmark

    detector = FaceDetector(cfg.MODEL_RETINAFACE)
    image = cv2.imread(image_path)
    boxes, landmarks = detector(image.copy())
    if boxes is not None:
        for bb in boxes:
            image = draw_bbox(image, bb)
    if landmarks is not None:
        for pts in landmarks:
            image = draw_landmark(image, pts)
    view_image(image, name='RetinaFace detection', wait_key=True)
