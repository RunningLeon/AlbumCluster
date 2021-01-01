import mxnet as mx

def is_cuda_available():
    """Check if mxnet with cuda is installed.

    Returns:
        [bool]: whether cuda is available.
    """
    has_cuda = False
    try:
        _ = mx.nd.array(1,ctx=mx.gpu(0))
        has_cuda = True
    except:
        pass
    return has_cuda
