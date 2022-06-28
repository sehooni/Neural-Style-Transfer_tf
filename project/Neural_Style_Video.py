# import and configure modules(모듈 구성 및 임포트)
import matplotlib.pyplot as plt
import matplotlib as mpl
import IPython.display

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools

import tensorflow_hub as hub
import cv2
import os

import tensorflow as tf

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K


# Configure modules
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


# Visualize the input(입력 시각화)
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# def load_img(img_path):     # 이미지를 불러오는 함수를 정의하고, 최대 이미지 크기를 512개의 픽셀로 제한
#     max_dim = 512
#     img = Image.open(img_path)
#     long = max(img.size)
#     scale = max_dim / long
#     img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
#
#     img = kp_image.img_to_array(img)
#
#     # In this case, we need to broadcast the image array such that it has a batch dimension.
#     img = np.expand_dims(img, axis=0)
#     return img


def show_img(img, title=None):    # 이미지를 출력하기 위한 간단한 함수를 정의
    # Remove the batch dimension
    out = np.squeeze(img, axis=0)
    # Normalize for display
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)

# Way to get images from video.
def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.jpg') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()


# Save results of process
def save_result_video(saving_point):
    try:
        video_dir = saving_point
        # video.save('{}/{}'.format(video_dir, video_name))
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        print("Sucessfully saved the output video!")
        return video_dir

    except IOError:
        print("An error occured when saving...")
        pass


def Neural_Style_Video(video_path, content_path, style_path, saving_point):
    print("--------- Neural Style Transfer to Video ---------\n")
    print("Eager execution: {}".format(tf.executing_eagerly()))
    # # when the tensorflow version is 1.x
    # tf.enable_eager_execution()
    print("Provide the file path to a content image and style image below:")

    ## Set up some global values here
    content_path = content_path
    style_path = style_path
    # base_img_path = './image/starry_night.jpg'
    # style_img_path = './image/starry_night.jpg'

    plt.figure(figsize=(10, 10))

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    plt.subplot(1, 2, 1)
    show_img(content_image, 'Content Image')

    plt.subplot(1, 2, 2)
    show_img(style_image, "Style Image")
    plt.show

    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    tensor_to_image(stylized_image)

    # Way to get images from video.
    # print(cv2.__version__)
    video_Dir = video_path
    img_Dir = content_path
    video_to_frames(video_Dir, img_Dir)
    # vidcap = cv2.VideoCapture(video_Dir + "lalaland1.mp4")
    # print(vidcap.read())
    # success, image = vidcap.read()
    # count = 0
    # success = True
    # while success:
    #     success, image = vidcap.read()
    #     print('Read a new frame: ', success)
    #     cv2.imwrite(img_Dir + "frame%d.jpg" % count, image)  # save frame as JPG file
    #     count += 1
    #     if count == 800:  # Estimated Based on 30 FPS X time of video (seconds)
    #         break


    # Style Transfer Image Frames using TF-Hub and Save images to styled_image folder
    content_path = content_path
    style_path = './styled_image/'
    frames_list = os.listdir(content_path)
    for frame in frames_list:
        idx = frame[5:frame.find('.jpg')]
        # if int(idx)%2==0:
        print('process images: ', frame)
        content_image = load_img(os.path.join(content_path, frame))
        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        img = tensor_to_image(stylized_image)
        img.save('./styled_image/' + frame)


    # Assemble Stylized Images into a Video
    image_folder = style_path
    # video_name = image_folder + '/lalaland2.avi'
    # video_results =

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape



    # Save Result video
    while True:
        cmd = input("Would you like to save the output image? (y/n): ")
        if cmd == 'y' or cmd == 'Y':
            output_video_name = input("File name for out file: ")
            # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            # c1 = save_result_video() + output_video_name +'.avi'
            # video = cv2.VideoWriter('{}'.format(save_result_video() + output_video_name +'.avi'), 0, 20, (width, height))  # FPS = 50
            video_result_dir = saving_point
            video = cv2.VideoWriter(save_result_video(video_result_dir) + output_video_name + '.avi', 0, 20, (width, height))  # FPS = 20
            for image in images:
                video.write(cv2.imread(os.path.join(image_folder, image)))

            cv2.destroyAllWindows()
            video.release()
            break
        elif cmd == 'n' or cmd =='N':
            break
        else:
            print("It's invalid command!")
    print("Conclude & Thank U!")


if __name__ == "__main__":
    video_Dir = './Video/Travel_in_Europe.mp4'
    img_Dir = './image/content_img/'
    style_path = './styled_image/'
    saving_point = './video_results/'
    Neural_Style_Video(video_Dir, img_Dir, style_path, saving_point)
