import os

__all__ = ['cfg']


class ModelCFG(object):
    """ModelCFG config file.

    Args:
        object ([type]): [description]
    """
    
    MODEL_RETINAFACE = os.path.join(os.path.dirname(__file__), '../models/retinaface-R50/R50,0')
    MODEL_INSIGHTFACE = os.path.join(os.path.dirname(__file__), '../models/model-r50-am-lfw/model,0')

    def __init__(self):
        self.__dict__.update(self.__class__.__dict__)
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            model_file = getattr(self, k)
            if k in ('MODEL_RETINAFACE', 'MODEL_INSIGHTFACE'):
                self.__class__.__assert_mxnet_file(self, model_file)
            else:
                if isinstance(v, str):
                    self.__class__.__assert_file(self, v)
                elif hasattr(v, "__iter__"):
                    for i in v:
                        self.__class__.__assert_file(self, i)

    def __assert_mxnet_file(self, model_file):
        path_, epoch = model_file.split(',')
        params_file = path_ + "-%04d.params" % int(epoch)
        symbol_file = path_ + "-symbol.json"
        self.__class__.__assert_file(self, params_file)
        self.__class__.__assert_file(self, symbol_file)

    def __assert_file(self, model_file):
        assert os.path.exists(model_file), "<%s> not exists!" % model_file


cfg = ModelCFG()
