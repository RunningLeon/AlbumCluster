from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import mxnet as mx
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing

from .info import is_cuda_available

__all__ = ['FaceFeatureExtractor']


def get_model(image_size, model_str, layer, batch_size=1):
    """
    get_model Load model from checkpoint

    Args:
        image_size ([type]): [description]
        model_str ([type]): [description]
        layer ([type]): [description]
        batch_size (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    context = mx.gpu() if is_cuda_available() else mx.cpu()
    model = mx.mod.Module(symbol=sym, context=context, label_names=None)
    model.bind(data_shapes=[('data', (batch_size, 3, image_size[0],
                                      image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceFeatureExtractor(object):
    """
    FaceFeatureExtractor 图像特征提取模块
    """

    def __init__(self, modelfile, batch_size=1):
        super(FaceFeatureExtractor, self).__init__()
        _vec = (112, 112)  # image height, width
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        self.ga_model = None
        self.model = get_model(image_size, modelfile, 'fc1', batch_size)
        self.image_size = image_size
        self.batch_size = batch_size

    def _get_input(self, face_img):
        face_img = cv2.resize(face_img, self.image_size, interpolation=cv2.INTER_LINEAR)
        nimg = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        aligned = np.expand_dims(aligned, axis=0)
        return aligned

    def _get_feature(self, aligned):
        """
        _get_feature 输入与处理后人脸图片, 提取特征

        Args:
            aligned ([type]): [description]

        Returns:
            [type]: [description]
        """
        data = mx.nd.array(aligned)
        db = mx.io.DataBatch(data=(data, ))
        self.model.forward(db, is_train=False)
        embeddings = self.model.get_outputs()
        embeddings = [_.asnumpy().reshape(1, -1) for _ in embeddings]
        embeddings = np.vstack(embeddings)
        embeddings = preprocessing.normalize(embeddings, axis=1)
        return embeddings

    def predict(self, image):
        """
        输入对齐后的人脸图片, 输出 512-D 人脸特征

        Args:
            image (np.ndarray or list): 输入人脸图片

        Returns:
            np.ndarray: 512-D 人脸特征
        """
        if isinstance(image, np.ndarray):
            img_processed = self._get_input(image)
            feat = self._get_feature(img_processed).reshape(-1)
        elif isinstance(image, list):
            nrof_image = len(image)
            nrof_batch = nrof_image // self.batch_size
            nrof_rest = nrof_image % nrof_batch
            if nrof_rest:
                nrof_batch += 1
                image += nrof_rest * [image[-1]]
            feat_li = []
            pbar = tqdm(range(nrof_batch), desc='Extracting feature: ')
            for i in pbar:
                start_idx = i * self.batch_size
                end_idx = (i+1) * self.batch_size
                img_li = image[start_idx:end_idx]
                img_processed = np.vstack(img_li)
                feat_ret = self._get_feature(img_processed)
                feat_li += [feat_ret[_].reshape(-1) for _ in range(len(feat_ret))]
            feat = feat_li[:nrof_image]

        return feat

    def __call__(self, image):
        return self.predict(image)


if __name__ == '__main__':
    import os
    import sys
    np.set_printoptions(precision=4)
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '.'))
    sys.path.append(os.path.join(current_dir, '..'))

    from app.model_config import cfg

    face_path = os.path.join(current_dir, '../images/align_face.png')
    assert os.path.exists(face_path), f'File not exists: {face_path}'
    face_img = cv2.imread(face_path)
    extractor = FaceFeatureExtractor(cfg.MODEL_INSIGHTFACE)
    feature = extractor.predict(face_img)
    print('Extracted face feature: ')
    print(feature.shape, feature.dtype)
    print(feature)


