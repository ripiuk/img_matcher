"""
Example:

>>> import img_matcher
>>> my_image = img_matcher.open_image('images/first_img.jpg')
>>> my_image.get_hash()
7ff0fef0fc787c787e787c7878380018001000000008003e7ffe7ffe3ffe1ffc
>>> my_image.get_hash(convert_to_bytes=True)
b'x\xda-\xc6\xd1\t\x00P\x08B\xd1\x95\x0c?l\x9f\xc8\xfdGx\xf5H8xe\xc3=J\xa9\xd5\xf7\xc9\x04b\xe16\xcd\x96\xfdq\x84]\x0f(x\x11b'
>>> my_image.get_histogram()
[1.6558431e-01 1.5567668e-02 3.3445659e-03 2.5470156e-04 0.0000000e+00
 0.0000000e+00 0.0000000e+00 0.0000000e+00 2.0376125e-03 2.9146604e-02
 1.6694529e-02 3.9568786e-03 2.1868314e-04 0.0000000e+00 0.0000000e+00
 ...
 3.7176134e-03 1.6480992e-02]
>>> my_image.get_features()
[[ 0.  0.  0. ...  2.  3. 27.]
 [ 0.  0.  0. ...  0.  0.  7.]
 [40.  2.  0. ... 11.  4. 48.]
 ...
 [ 0.  0.  0. ...  0.  0.  2.]
 [ 8.  0.  0. ...  0.  0.  0.]
 [ 5.  0.  0. ...  9.  1.  0.]]
>>> other_image = img_matcher.open_image('images/second_img.jpg')
>>> img_matcher.match_by_hashes(my_image.get_hash(), other_image.get_hash())
True
>>> img_matcher.match_by_hashes(my_image.get_hash(), other_image.get_hash(), get_raw_result=True)
62
>>> img_matcher.match_by_histograms(my_image.get_histogram(convert_to_str=True), other_image.get_histogram(),
... get_raw_result=True)
0.5763813126029974
>>> img_matcher.match_by_features(my_image.get_features(), other_image.get_features(convert_to_bytes=True),
... get_raw_result=True)
[[<DMatch 0x7fafb66533b0>, <DMatch 0x7fafb6653390>], [<DMatch 0x7fafb6653450>, <DMatch 0x7fafb66534d0>],
[<DMatch 0x7fafb6653630>, <DMatch 0x7fafb66535d0>], [<DMatch 0x7fafb6653710>, <DMatch 0x7fafb6653530>],
...
[<DMatch 0x7fafb6653730>, <DMatch 0x7fafb66536b0>], [<DMatch 0x7fafb6653750>, <DMatch 0x7fafb6653770>],
[<DMatch 0x7fafb59f9490>, <DMatch 0x7fafb59f94b0>], [<DMatch 0x7fafb59f94d0>, <DMatch 0x7fafb59f94f0>]]
"""

import json
import zlib

import cv2
import numpy
import imagehash
from PIL import Image


class ImageRepresentation:

    def __init__(self, path: str):
        """
        :param path: path to the picture
            (e.g '/home/username/picture.jpg', 'img/picture2.png')
        """
        self.image_path = path

    def get_hash(self, convert_to_str: bool=False, convert_to_bytes: bool=False,
                 hash_size: int=16) -> (imagehash.ImageHash, str, bytes):
        """ Get a wavelet hash of an image
        :param convert_to_str: bool (return a string instance)
        :param convert_to_bytes: bool (return a compressed bytes instance)
        :param hash_size: the size of the hash (power of 2) e.g 8, 16, 32
        :return: imagehash.ImageHash instance e.g <class 'imagehash.ImageHash'>
            or str e.g '7ff0fef0fc787c787e787c7878380018001000000008003e7ffe7ffe3ffe1ffc'
            or bytes e.g b'x\xda+\xce\xcfMU(I\xad(\x01\x00\x11\xe8\x03\x9a'
        """
        image = Image.open(self.image_path)

        the_hash = imagehash.whash(image, hash_size=hash_size)
        if convert_to_str or convert_to_bytes:
            the_hash = str(the_hash)
            if convert_to_bytes:
                the_hash = _compress_str_to_byte(the_hash)
        return the_hash

    def get_histogram(self, convert_to_str: bool=False, convert_to_bytes: bool=False) -> (numpy.ndarray, str, bytes):
        """ Calculate histogram of blue, green or red channel respectively
        :param convert_to_str: bool (return a string instance)
        :param convert_to_bytes: bool (return a compressed bytes instance)
        :return: numpy.ndarray instance e.g [1.6558431e-01 1.5567668e-02 3.3445659e-03 ... 1.6480992e-02]
            or str e.g '[0.16558431088924408, 0.015567667782306671, 0.00334456586278975, ..., 0.01648099161684513]'
            or bytes e.g b'x\xda+\xce\xcfMU(I\xad(\x01\x00\x11\xe8\x03\x9a'
        """
        image = cv2.imread(self.image_path)

        # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        #
        # images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
        # channels : it is also given in square brackets. It is the index of channel for which we calculate histogram.
        #   For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2]
        #   to calculate histogram of blue, green or red channel respectively.
        # mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram
        #   of particular region of image, you have to create a mask image for that and give it as mask.
        # histSize : this represents our BIN count. Need to be given in square brackets. For full scale - [256].
        # ranges : this is our RANGE. Normally, it is [0,256].
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        hist = cv2.normalize(hist, hist).flatten()
        if convert_to_str or convert_to_bytes:
            hist = json.dumps(hist.tolist())
            if convert_to_bytes:
                hist = _compress_str_to_byte(hist)
        return hist

    def get_features(self, convert_to_str: bool=False, convert_to_bytes: bool=False) -> (numpy.ndarray, str, bytes):
        """ Directly find keypoints and descriptors in a single step
        :param convert_to_str: bool (return a string instance)
        :param convert_to_bytes: bool (return a compressed bytes instance)
        :return: numpy.ndarray instance e.g [[ 0.  0.  0. ...  2.  3. 27.] ... [ 5.  0.  0. ...  9.  1.  0.]]
            or str e.g '[[0.0, 0.0, 0.0, ... 2.0, 3.0, 27.0], ... [5.0, 0.0, 0.0, ... 9.0, 1.0, 0.0]]'
            or bytes e.g b'x\xda+\xce\xcfMU(I\xad(\x01\x00\x11\xe8\x03\x9a'
        """
        image = cv2.imread(self.image_path, 0)

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        _, des1 = sift.detectAndCompute(image, None)
        if convert_to_str or convert_to_bytes:
            des1 = json.dumps(des1.tolist())
            if convert_to_bytes:
                des1 = _compress_str_to_byte(des1)
        return des1


def open_image(path: str) -> ImageRepresentation:
    """
    :param path: path to the picture
        (e.g '/home/username/picture.jpg', 'img/picture2.png')
    :return: instance of an ImageRepresentation class
    """
    return ImageRepresentation(path)


def match_by_hashes(first_hash: (imagehash.ImageHash, str, bytes), second_hash: (imagehash.ImageHash, str, bytes),
                    get_raw_result: bool=False, max_diff_value: int=5) -> (int, bool):
    """ Match two images using hashes of the same shape (hash size) using Hamming distance of two hashes.
    :param first_hash: e.g <class 'imagehash.ImageHash'> or e.g 'fcec6e4400027f7e'
        or e.g b'x\xda+\xce\xcfMU(I\xad(\x01\x00\x11\xe8\x03\x9a'
    :param second_hash: e.g <class 'imagehash.ImageHash'> or e.g 'acec604b050f787c'
        e.g b'x\xda+\xce\xcfMU(I\xad(\x01\x00\x11\xe8\x03\x9a'
    :param get_raw_result: return integer value of Hamming distance of two hashes
    :param max_diff_value: the boundary value of difference between two images
    :return: Hamming distance of two hashes, or bool - are these the same images
    """
    if isinstance(first_hash, bytes):
        first_hash = _decompress_byte_to_str(first_hash)
    if isinstance(second_hash, bytes):
        second_hash = _decompress_byte_to_str(second_hash)
    if isinstance(first_hash, str):
        first_hash = _from_str_to_imagehash(first_hash)
    if isinstance(second_hash, str):
        second_hash = _from_str_to_imagehash(second_hash)
    diff = first_hash - second_hash
    if get_raw_result:
        return diff
    return diff < max_diff_value


def match_by_histograms(first_hist: (numpy.ndarray, str, bytes), second_hist: (numpy.ndarray, str, bytes),
                        get_raw_result: bool=False, max_diff_value: float=2.0) -> (float, bool):
    """ Match two images using histograms (Chi-Squared method)
    :param first_hist: e.g <class 'numpy.ndarray'>
        or e.g '[0.16558431088924408, 0.015567667782306671, 0.00334456586278975, ..., 0.01648099161684513]'
        or e.g b'x\xda+\xce\xcfMU(I\xad(\x01\x00\x11\xe8\x03\x9a'
    :param second_hist: e.g <class 'numpy.ndarray'>
        or e.g '[0.16558431088924408, 0.015567667782306671, 0.00334456586278975, ..., 0.01648099161684513]'
        or e.g b'x\xda+\xce\xcfMU(I\xad(\x01\x00\x11\xe8\x03\x9a'
    :param get_raw_result: return difference between two hashes using Chi-Squared method
    :param max_diff_value: the boundary value of difference between two images
    :return: difference between two hashes using Chi-Squared method, or bool - are these the same images
    """
    if isinstance(first_hist, bytes):
        first_hist = _decompress_byte_to_str(first_hist)
    if isinstance(second_hist, bytes):
        second_hist = _decompress_byte_to_str(second_hist)
    if isinstance(first_hist, str):
        first_hist = _from_str_to_ndarray(first_hist)
    if isinstance(second_hist, str):
        second_hist = _from_str_to_ndarray(second_hist)
    diff = cv2.compareHist(first_hist, second_hist, cv2.HISTCMP_CHISQR)
    if get_raw_result:
        return diff
    return diff < max_diff_value


def match_by_features(first_features: (numpy.ndarray, str, bytes),
                      second_features: (numpy.ndarray, str, bytes), get_raw_result: bool=False,
                      min_match_count: int=10, match_coefficient: float=0.6) -> (list, bool):
    """ Match two images using features (FLANN based Matcher)
    :param first_features: e.g <class 'numpy.ndarray'>
        or e.g '[[0.0, 0.0, 0.0, ... 2.0, 3.0, 27.0], ... [5.0, 0.0, 0.0, ... 9.0, 1.0, 0.0]]'
        or e.g b'x\xda+\xce\xcfMU(I\xad(\x01\x00\x11\xe8\x03\x9a'
    :param second_features:  e.g <class 'numpy.ndarray'>
        or e.g '[[0.0, 0.0, 0.0, ... 2.0, 3.0, 27.0], ... [5.0, 0.0, 0.0, ... 9.0, 1.0, 0.0]]'
        or e.g b'x\xda+\xce\xcfMU(I\xad(\x01\x00\x11\xe8\x03\x9a'
    :param get_raw_result: return feature difference using FLANN based Matcher
        e.g [[<DMatch 0x7f48ad7de3b0>, <DMatch 0x7f48ad7de390>], ... <DMatch 0x7f48acb874f0>]]
    :param min_match_count: the min amount of matches by features between two images
    :param match_coefficient: coefficient of coincidence between two images
        e.g (0.0 ... 1.0)
    :return: feature difference using FLANN based Matcher, or bool - are these the same images
    """
    if isinstance(first_features, bytes):
        first_features = _decompress_byte_to_str(first_features)
    if isinstance(second_features, bytes):
        second_features = _decompress_byte_to_str(second_features)
    if isinstance(first_features, str):
        first_features = _from_str_to_ndarray(first_features)
    if isinstance(second_features, str):
        second_features = _from_str_to_ndarray(second_features)
    flann_index_kdtree = 0
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(first_features, second_features, k=2)
    if get_raw_result:
        return matches
    good = []
    for m, n in matches:
        if m.distance < match_coefficient * n.distance:
            good.append(m)
    return len(good) > min_match_count


def _from_str_to_imagehash(element: str) -> imagehash.ImageHash:
    """ Convert string to imagehash.ImageHash instance
    :param element: e.g 'fcec6e4400027f7e'
    :return: e.g <class 'imagehash.ImageHash'>
    """
    return imagehash.hex_to_hash(element)


def _from_str_to_ndarray(element: str) -> numpy.ndarray:
    """ Convert string to numpy.ndarray instance
    :param element: e.g '[[0.0, 0.0, 0.0, ... 2.0, 3.0, 27.0], ... [5.0, 0.0, 0.0, ... 9.0, 1.0, 0.0]]'
    :return: e.g <class 'numpy.ndarray'> [[ 0.  0.  0. ...  2.  3. 27.] ... [ 5.  0.  0. ...  9.  1.  0.]]
    """
    return numpy.asarray(json.loads(element)).astype(numpy.float32)


def _compress_str_to_byte(data: str) -> bytes:
    """ Compress data from string to bytes
    :param data: string
    :return: a compressed string in bytes
    >>> data = 'some text'
    >>> print(_compress_str_to_byte(data))
    >>> b'x\xda+\xce\xcfMU(I\xad(\x01\x00\x11\xe8\x03\x9a'
    """
    return zlib.compress(data.encode('utf-8'), level=9)


def _decompress_byte_to_str(data: bytes) -> str:
    """ Decompress data from compressed bytes to string
    :param data: a compressed string in bytes
    :return: decompressed string
    >>> data = b'x\xda+\xce\xcfMU(I\xad(\x01\x00\x11\xe8\x03\x9a'
    >>> print(_decompress_byte_to_str(data))
    >>> 'some text'
    """
    return zlib.decompress(data).decode('utf-8')
