## Image Matcher

An image matcher library written in Python. img_matcher supports:
* transform an image to a wavelet hash, comparison images using Hamming distance of two hashes;
* transform an image to a histogram of blue, green or red channel respectively, 
comparison images using Chi-Squared method;
* transform an image to a keypoints and descriptors, match two images using features (FLANN based Matcher);

## Getting Started

* Install python
~~~bash
$ make install
~~~
* Initialize virtualenvironment
~~~bash
$ make venv_init
~~~

## Basic usage
~~~pydocstring
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
~~~