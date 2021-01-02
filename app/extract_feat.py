from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import mxnet as mx
import numpy as np
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
    FaceFeatureExtractor 人脸识别特征提取模块
    """

    def __init__(self, modelfile, batch_size=1):
        super(FaceFeatureExtractor, self).__init__()
        _vec = (112, 112)  # image height, width
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        self.ga_model = None
        self.model = get_model(image_size, modelfile, 'fc1')
        self.image_size = image_size
        self.batch_size = batch_size

    @staticmethod
    def _get_input(face_img):
        nimg = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        return aligned

    def _get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data, ))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding).flatten()
        return embedding

    def predict(self, image: np.ndarray):
        """
        输入一张对齐后的人脸图片, 输出 512-D 人脸特征

        Args:
            image (np.ndarray): 输入人脸图片

        Returns:
            np.ndarray: 512-D 人脸特征
        """
        src_img = cv2.resize(
            image, self.image_size, interpolation=cv2.INTER_LINEAR)
        img_processed = self._get_input(src_img)
        feat = self._get_feature(img_processed)
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


