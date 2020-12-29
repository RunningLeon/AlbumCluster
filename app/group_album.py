import os

import cv2
import numpy as np
from tqdm import tqdm

from .align_face import align_face, crop_image, expand_bbox, get_mean_landmarks
from .detect_face import FaceDetector
from .extract_feat import FaceFeatureExtractor
from .face_clustering import FaceClusteringAlgo
from .model_config import cfg
from .view_util import draw_bbox, draw_landmark, view_image


class FaceInfo(object):
    """
    FaceInfo class

    Args:
        object ([type]): [description]
    """
    def __init__(self, bbox, landmarks, feature):
        """
        init

        Args:
            bbox ([type]): [description]
            landmarks ([type]): [description]
            feature ([type]): [description]
        """
        # core members
        self.bbox = bbox
        self.landmarks = landmarks
        self.feature = feature
        self.face_id = None
        self.image_path = None
        self.person_id = None

        # for debug
        self.expand_bbox = None
        self.align_face = None
        self.crop_face = None

    def create_id(self, image_id):
        _id = '#'.join([str(i) for i in self.bbox[:4]])
        _id = image_id + _id
        self.face_id = _id
        return self.face_id


class GroupAlbum(object):
    """
    GroupAlbum class

    Args:
        object ([type]): [description]
    """
    def __init__(self, save_dir, debug=False):
        """
        init

        Args:
            save_dir ([type]): [description]
            debug (bool, optional): [description]. Defaults to False.
        """
        super(GroupAlbum, self).__init__()

        self._cluster_algo = FaceClusteringAlgo()
        self._face_detector = FaceDetector(cfg.MODEL_RETINAFACE)
        self._face_extractor = FaceFeatureExtractor(cfg.MODEL_INSIGHTFACE)
        self._mean_landmarks = get_mean_landmarks()
        self.process_data = {}
        self.save_dir = save_dir
        self.debug = debug

    def _process(self, image: np.ndarray, image_id, image_path):
        """
        _process 

        Args:
            image (np.ndarray): [description]
            image_id ([type]): [description]
            image_path ([type]): [description]

        Returns:
            [type]: [description]
        """
        faceinfo_li = []
        height, width, _ = image.shape
        bboxes, landmarks = self._face_detector(image)
        if self.debug:
            drawn = image.copy()
        if bboxes is not None and landmarks is not None:
            for bbox, pts in zip(bboxes, landmarks):
                # expand bbox and crop face from original image
                exp_bbox = expand_bbox(width, height, bbox)
                cropped_face = crop_image(image, exp_bbox)

                # transform landmarks to crop face coordinate and align face
                pts_in_face = pts.copy()
                pts_in_face[:, 0] -= exp_bbox[0]
                pts_in_face[:, 1] -= exp_bbox[1]
                aligned_face = align_face(cropped_face, pts_in_face,
                                          self._mean_landmarks)

                # extract face feature
                feature = self._face_extractor(aligned_face)

                # update face_info
                face_info = FaceInfo(bbox, pts, feature)
                face_info.image_path = image_path
                face_id = face_info.create_id(image_id)
                self.process_data[face_id] = face_info

                if self.debug:
                    cropped_face = draw_landmark(cropped_face, pts_in_face, draw_order=True)
                    face_info.expand_bbox = exp_bbox
                    face_info.align_face = aligned_face
                    face_info.crop_face = cropped_face

                    view_image(cropped_face,
                               name='cropped_face',
                               wait_key=False)
                    view_image(aligned_face,
                               name='aligned_face',
                               wait_key=False)
                    drawn = draw_bbox(drawn, bbox)
                    drawn = draw_landmark(drawn, pts)

                faceinfo_li.append(face_info)

        if self.debug:
            pass
            view_image(drawn, wait_key=False)

        return faceinfo_li

    def run(self, filename_li):
        """
        run 

        Args:
            filename_li ([type]): [description]

        Returns:
            [type]: [description]
        """
        faceinfo_li = []
        pbar = tqdm(range(len(filename_li)), desc='Extracting feature: ')
        for i in pbar:
            image_path = filename_li[i]
            _, filename = os.path.split(image_path)
            image_id, _ = os.path.splitext(filename)
            if not os.path.exists(image_path):
                continue
            image = cv2.imread(image_path)
            faceinfo_li += self._process(image, image_id, image_path)

        cluster_results = self._cluster_algo.process(faceinfo_li)
        for face_id, person_id in cluster_results.items():
            if not face_id in self.process_data:
                print(f'warning: face_id {face_id} not found')
                continue
            self.process_data[face_id].person_id = person_id

        return self.process_data

    def __call__(self, filename_li):
        return self.run(filename_li)
        
    def save(self):
        """
        save results
        """
        face_id_li = tqdm(self.process_data.keys(), desc='Save image')
        for face_id in face_id_li:
            face_info = self.process_data[face_id]
            person_id = face_info.person_id
            person_dir = os.path.join(self.save_dir, f'person-{person_id}')
            os.makedirs(person_dir, exist_ok=True)
            src_image_path = face_info.image_path
            _, filename = os.path.split(src_image_path)
            dst_image_path = os.path.join(person_dir, filename)
            if self.debug:
                image = cv2.imread(src_image_path)
                image = draw_bbox(image, face_info.bbox)
                image = draw_landmark(image, face_info.landmarks)
                cv2.imwrite(dst_image_path, image)
                crop_face_path = os.path.join(person_dir, face_info.face_id + '_crop.png')
                align_face_path = os.path.join(person_dir, face_info.face_id + '_align.png')
                cv2.imwrite(crop_face_path, face_info.crop_face)
                cv2.imwrite(align_face_path, face_info.align_face)
            else:
                os.symlink(src_image_path, dst_image_path)

        print(f'Saved to {self.save_dir}')
