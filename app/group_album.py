import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from .align_face import align_face, crop_image, expand_bbox, get_mean_landmarks
from .detect_face import FaceDetector
from .extract_feat import FaceFeatureExtractor
from .face_clustering import FaceClusteringAlgo, FaceClusteringDBSCAN
from .model_config import cfg
from .view_util import draw_bbox, draw_landmark, view_image


class FaceInfo(object):
    """
    保存人脸信息的类
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
    相册归档主处理类
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
        # self._cluster_algo = FaceClusteringDBSCAN()
        self._face_detector = FaceDetector(cfg.MODEL_RETINAFACE)
        self._face_extractor = FaceFeatureExtractor(cfg.MODEL_INSIGHTFACE)
        self._mean_landmarks = get_mean_landmarks()
        self.process_data = {}
        self.save_dir = save_dir
        self.debug = debug
        self.feat_batch_size = 5

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
        aligned_face_li = []
        height, width, _ = image.shape
        bboxes, landmarks = self._face_detector(image)
        if self.debug:
            drawn = image.copy()
        if bboxes is not None and landmarks is not None:
            for bbox, pts in zip(bboxes, landmarks):
                # expand bbox and crop face from original image
                exp_bbox = expand_bbox(width, height, bbox, margin=12)
                cropped_face = crop_image(image, exp_bbox)

                # transform landmarks to crop face coordinate and align face
                pts_in_face = pts.copy()
                pts_in_face[:, 0] -= exp_bbox[0]
                pts_in_face[:, 1] -= exp_bbox[1]
                aligned_face = align_face(cropped_face, pts_in_face,
                                          self._mean_landmarks)

                if self.feat_batch_size == 1:
                    # extract face feature
                    feature = self._face_extractor(aligned_face)
                else:
                    # will extract face feature by batch
                    aligned_face_li.append(self._face_extractor._get_input(aligned_face))
                    feature = None # temp

                # update face_info
                face_info = FaceInfo(bbox, pts, feature)
                face_info.image_path = image_path
                face_id = face_info.create_id(image_id)
                self.process_data[face_id] = face_info

                if self.debug:
                    cropped_face = draw_landmark(cropped_face, pts_in_face)
                    face_info.expand_bbox = exp_bbox
                    face_info.align_face = aligned_face
                    face_info.crop_face = cropped_face

                    # view_image(cropped_face,
                    #            name='cropped_face',
                    #            wait_key=False)
                    # view_image(aligned_face,
                    #            name='aligned_face',
                    #            wait_key=False)
                    drawn = draw_bbox(drawn, bbox)
                    drawn = draw_landmark(drawn, pts)

                faceinfo_li.append(face_info)

        # # extract face feature by batch
        # if aligned_face_li:
        #     feat_li = self._face_extractor.predict(aligned_face_li)
        #     for i in range(len(faceinfo_li)):
        #         faceinfo_li[i].feature = feat_li[i]
        if self.debug:
            pass
            # view_image(drawn, wait_key=False)

        return faceinfo_li, aligned_face_li

    def run(self, filename_li):
        """
        主处理函数

        Args:
            filename_li (list[str]): 输入图片文件名列表

        Returns:
            list[FaceInfo]: FaceInfo 列表
        """
        ## save results for debug
        import pickle
        current_dir = os.path.dirname(__file__)
        pkl_path = os.path.join(current_dir, '../data/data.pkl')
        data_dir, _ = os.path.split(pkl_path)
        os.makedirs(data_dir, exist_ok=True)
        if self.debug and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.process_data, faceinfo_li = pickle.load(f)
        else:
            faceinfo_li = []
            aligned_face_li = []
            pbar = tqdm(range(len(filename_li)), desc='Detecting face: ')
            for i in pbar:
                image_path = filename_li[i]
                _, filename = os.path.split(image_path)
                image_id, _ = os.path.splitext(filename)
                if not os.path.exists(image_path):
                    continue
                image = cv2.imread(image_path)
                cur_faceinfo_li, cur_align_li = self._process(image, image_id, image_path)
                faceinfo_li += cur_faceinfo_li
                aligned_face_li += cur_align_li

            # extract feature by batch
            if aligned_face_li:
                feat_li = self._face_extractor.predict(aligned_face_li)
                for i in range(len(faceinfo_li)):
                    faceinfo_li[i].feature = feat_li[i]

            # debug
            if self.debug:
                with open(pkl_path, 'wb') as f:
                    dat = (self.process_data, faceinfo_li)
                    return pickle.dump(dat, f)
        ## cluster faces 
        cluster_results = self._cluster_algo.process(faceinfo_li)
        for face_id, person_id in cluster_results.items():
            if not face_id in self.process_data:
                print(f'warning: face_id {face_id} not found')
                continue
            self.process_data[face_id].person_id = person_id

    def __call__(self, filename_li):
        return self.run(filename_li)
        
    def save(self):
        """
        保存聚类结果
        """
        face_id_li = tqdm(self.process_data.keys(), desc='Saving result')
        for face_id in face_id_li:
            face_info = self.process_data[face_id]
            person_id = face_info.person_id
            person_dir = os.path.join(self.save_dir, f'person-{person_id}')
            os.makedirs(person_dir, exist_ok=True)
            src_image_path = face_info.image_path
            src_image_path = os.path.abspath(src_image_path)
            _, filename = os.path.split(src_image_path)
            dst_image_path = os.path.join(person_dir, filename)
            if os.path.exists(dst_image_path):
                image = cv2.imread(dst_image_path)
            else:
                image = cv2.imread(src_image_path)
            image = draw_bbox(image, face_info.bbox)
            image = draw_landmark(image, face_info.landmarks)
            cv2.imwrite(dst_image_path, image)
            if self.debug:
                crop_face_path = os.path.join(person_dir, face_info.face_id + '_crop.png')
                align_face_path = os.path.join(person_dir, face_info.face_id + '_align.png')
                cv2.imwrite(crop_face_path, face_info.crop_face)
                cv2.imwrite(align_face_path, face_info.align_face)
                # os.symlink(src_image_path, dst_image_path)

        print(f'Saved to {self.save_dir}')

    def show(self):
        """
        show 可视化显示聚类结果
        """
        group_dic = defaultdict(list)
        for face_info in self.process_data.values():
            group_dic[face_info.person_id].append(face_info)
        
        group_li = sorted(group_dic.items(), key=lambda x:len(x[1]), reverse=True)
        nrof_max_face_per_person = len(group_li[0][1])
        nrof_person = len(group_li)
        pad = 12
        image_size = 224
        image_size_pad = image_size + pad
        nrof_rows = image_size_pad * nrof_person
        nrof_cols = image_size_pad * nrof_max_face_per_person
        shape = (nrof_rows, nrof_cols, 3)
        image_matrix = np.zeros(shape) + 255
        image_matrix = image_matrix.astype(np.uint8)
        for i, (cid, group) in enumerate(group_li):
            row_start = i * image_size_pad
            for j, face_info in enumerate(group):
                col_start = j * image_size_pad
                # face_img = cv2.resize(face_info.crop_face, (image_size, image_size))
                face_img = cv2.resize(face_info.align_face, (image_size, image_size))
                image_matrix[row_start:(row_start+image_size), col_start:(col_start+image_size), :] \
                    = face_img
        view_image(image_matrix, name='face clustering', wait_key=True)
