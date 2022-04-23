# import and configure modules(모듈 구성 및 임포트)
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# from tensorflow.python.keras.preprocessing import image as kp_image
# from tensorflow.python.keras import models
# from tensorflow.python.keras import losses
# from tensorflow.python.keras import layers
# from tensorflow.python.keras import backend as K

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

## Set up some global values here (예제와 다르게 나만의 사진으로 구현)
content_path = 'C:/Users/74seh/python/WorkSpace/opencv/project/image/Tuebingen_Neckarfront.jpg'
style_path = 'C:/Users/74seh/python/WorkSpace/opencv/project/image/Arbre_en_fleur,_Gustave_Caillebotte,_1882.jpg'

# Visualize the input(입력 시각화)


# def load_img(path_to_img):
#     max_dim = 512
#     img = Image.open(path_to_img)
#     long = max(img.size)
#     scale = max_dim/long
#     img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
#
#     img = kp_image.img_to_array(img)
#
#   # We need to broadcast the image array such that it has a batch dimension
#     img = np.expand_dims(img, axis=0)
#     return img
#
# def imshow(img, title=None):
#     # Remove the batch dimension
#     out = np.squeeze(img, axis=0)
#     # Normalize for display
#     out = out.astype('uint8')
#     plt.imshow(out)
#     if title is not None:
#         plt.title(title)
#     plt.imshow(out)
#
# plt.figure(figsize= (10, 10))
#
# content = load_img(content_path).astype('uint8')
# style = load_img(style_path).astype('uint8')
#
# plt.subplot(1, 2, 1)
# imshow(content, 'Content Image')
#
# plt.subplot(1, 2, 2)
# imshow(style, "Style Image")
# print(plt.show)

def load_img(path_to_img):  # 이미지를 불러오는 함수를 정의하고, 최대 이미지 크기를 512개의 픽셀로 제한
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim/long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize_images(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):    # 이미지를 출력하기 위한 간단한 함수를 정의
    if len(image.shape) > 3 :
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title is not None:
        plt.title(title)


content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, "Style Image")

# 콘텐츠와 스타일 표현 정의
# VGG19 모델을 불러오고, 작동여부를 확인하기 위해 이미지에 적용
x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.vgg19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape


