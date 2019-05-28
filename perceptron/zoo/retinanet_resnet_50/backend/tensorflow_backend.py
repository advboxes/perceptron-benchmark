"""rensorflow backend"""
import tensorflow


def ones(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/
    api_docs/python/tf/ones .
    """
    return tensorflow.ones(*args, **kwargs)


def transpose(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/
    api_docs/python/tf/transpose .
    """
    return tensorflow.transpose(*args, **kwargs)


def map_fn(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/
    api_docs/python/tf/map_fn .
    """
    return tensorflow.map_fn(*args, **kwargs)


def pad(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/
    api_docs/python/tf/pad .
    """
    return tensorflow.pad(*args, **kwargs)


def top_k(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/
    api_docs/python/tf/nn/top_k .
    """
    return tensorflow.nn.top_k(*args, **kwargs)


def clip_by_value(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/
    api_docs/python/tf/clip_by_value .
    """
    return tensorflow.clip_by_value(*args, **kwargs)


def resize_images(images, size, method='bilinear', align_corners=False):
    """ See https://www.tensorflow.org/versions/master/
    api_docs/python/tf/image/resize_images .

    Parameters
    ----------
    method : str
        The method used for interpolation. One of ('bilinear', 'nearest',
         'bicubic', 'area').
    """
    methods = {
        'bilinear': tensorflow.image.ResizeMethod.BILINEAR,
        'nearest': tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic': tensorflow.image.ResizeMethod.BICUBIC,
        'area': tensorflow.image.ResizeMethod.AREA,
    }
    return tensorflow.image.resize_images(
        images, size, methods[method], align_corners)


def non_max_suppression(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/
    tf/image/non_max_suppression .
    """
    return tensorflow.image.non_max_suppression(*args, **kwargs)


def range(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/
    tf/range .
    """
    return tensorflow.range(*args, **kwargs)


def scatter_nd(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/
    tf/scatter_nd .
    """
    return tensorflow.scatter_nd(*args, **kwargs)


def gather_nd(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/
    tf/gather_nd .
    """
    return tensorflow.gather_nd(*args, **kwargs)


def meshgrid(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/
    tf/meshgrid .
    """
    return tensorflow.meshgrid(*args, **kwargs)


def where(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/
    tf/where .
    """
    return tensorflow.where(*args, **kwargs)
