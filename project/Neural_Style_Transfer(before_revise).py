# Download Images
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
img_dir = 'D:/WorkSpace/Neural-Style-Transfer_tf/project/image/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

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

#
# import tensorflow.compat.v1 as tf

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

# # when the tensorflow version is 1.x
# tf.enable_eager_execution()

# when the tensorflow version is 2.x
tf.executing_eagerly()
print("Eager execution: {}".format(tf.executing_eagerly()))

## Set up some global values here (예제와 다르게 나만의 사진으로 구현)
content_path = './image/Tuebingen_Neckarfront.jpg'
style_path = './image/Effel_Tower.jpg'


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
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def imshow(image, title=None):  # 이미지를 출력하기 위한 간단한 함수를 정의
    if len(image.shape) > 3:
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
plt.show


# 콘텐츠와 스타일 표현 정의
# prepare the data
def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "demension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


# # Define content and style representations
# VGG19 모델을 불러오고, 작동여부를 확인하기 위해 이미지에 적용
x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
print(prediction_probabilities.shape)

predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]

# # 분류층을 제외한 VGG19 모델을 불러오고, 각 층의 이름을 출력
# vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# print()
# for layer in vgg.layers:
#     print(layer.name)
#
# Content layer where we will pull our feature maps
content_layers = ['block5_conv2']

# Style_layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# Build the Model
def get_model():
    """ Creates our model with access to intermediate layers.

      This function will load the VGG19 model and access the intermediate layers.
      These layers will then be used to create a new model that will take input image
      and return the outputs from these intermediate layers from the VGG model.

      Returns:
          returns a keras model that takes image inputs and outputs the style and
          content intermediate layers.
      """
    # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model
    return models.Model(vgg.input, model_outputs)

# def vgg_layers(layer_names):
#     """ 중간층의 출력값을 배열로 반환하는 vgg 모델을 만든다. """
#     # Load our model. We load pretrained VGG, trained on imagenet data
#     vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
#     vgg.trainable = False
#
#     outputs = [vgg.get_layer(name).output for name in layer_names]
#
#     model = tf.keras.Model([vgg.input], outputs)
#     return model

#
# style_extractor = get_model(style_layers)
# # style_extractor = vgg_layers(style_layers)
# style_outputs = style_extractor(style_image*255)

# # 각 층의 출력에 대한 통계량을 살펴보면
# for name, output in zip(style_layers, style_outputs):
#     print(name)
#     print(" 크기: ", output.numpy().shape)
#     print(" 최솟값: ", output.numpy().min())
#     print(" 최댓값: ", output.numpy().max())
#     print(" 평균: ", output.numpy().mean())
#     print()
#
# # 스타일 계산하기
# def gram_matrix(input_tensor):
#     result = tf.linalg.einsum('bijc,bijd→bcd', input_tensor, input_tensor)
#     input_shape = tf.shape(input_tensor)
#     num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
#     return result/(num_locations)
#
#
# # 스타일과 콘텐츠 추출하기
# class StyleContentModel(tf.keras.models.Model):
#     def __init__(self, style_layers, content_layers):
#         super(StyleContentModel, self).__init__()
#         self.vgg = vgg_layers(style_layers + content_layers)
#         self.style_layers = style_layers
#         self.content_layers = content_layers
#         self.num_style_layers = len(style_layers)
#         self.vgg.trainable = False
#
#     def call(self, inputs):
#         "[0, 1] 사이의 실수 값을 입력으로 받습니다."
#         inputs = inputs * 255.0
#         preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
#         outputs = self.vgg(preprocessed_input)
#         style_outputs, content_outputs = (outputs[:self.num_style_layers],
#                                           outputs[self.num_style_layers:])
#
#         style_outputs = [gram_matrix(style_output)
#                          for style_output in style_outputs]
#
#         content_dict = {content_name:value
#                         for content_name, value
#                         in zip(self.content_layers, content_outputs)}
#
#         style_dict = {style_name:value
#                       for style_name, value
#                       in zip(self.style_layers, style_outputs)}
#
#         return {'content': content_dict, 'style': style_dict}
#
# extractor = StyleContentModel(style_layers, content_layers)
#
# results = extractor(tf.constant(content_image))
#
# print('스타일:')
# for name, output in sorted(results['style'].items()):
#     print("  ", name)
#     print("    크기: ", output.numpy().shape)
#     print("    최솟값: ", output.numpy().min())
#     print("    최댓값: ", output.numpy().max())
#     print("    평균: ", output.numpy().mean())
#     print()
#
# print("콘텐츠:")
# for name, output in sorted(results['content'].items()):
#     print("  ", name)
#     print("    크기: ", output.numpy().shape)
#     print("    최솟값: ", output.numpy().min())
#     print("    최댓값: ", output.numpy().max())
#     print("    평균: ", output.numpy().mean())

# Computing content loss
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Computing style loss
def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))  # / (4. * (channels ** 2) * (width * height) ** 2)

# Apply style transfer to our images
def get_feature_representations(model, content_path, style_path):
    """Helper function to compute our content and style feature representations.

    This function will simply load and preprocess both the content and style
    images from their path. Then it will feed them through the network to obtain
    the outputs of the intermediate layers.

    Arguments:
      model: The model that we are using.
      content_path: The path to the content image.
      style_path: The path to the style image

    Returns:
      returns the style features and the content features.
    """
    # Load our images in
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

# Computing the loss and gradients
# def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
#   """This function will compute the loss total loss.

#   Arguments:
#     model: The model that will give us access to the intermediate layers.
#     loss_weights: The weights of each contribution of each loss function.
#       (style weight, content weight, and total variation weight)
#     init_image: Our initial base image. This image is what we are updating with our optimization process. We apply the gradients wrt the loss we are calculating to this image.
#     gram_style_features: Precomputed gram matrices corresponding to the defined style layers of interest.
#     content_features: Precomputed outputs from defined content layers of interest.

#   Returns:
#     returns the total loss, style loss, content loss, and total variational loss
#   """
#   style_weight, content_weight = loss_weights

#   # Feed our init image through our model. This will give us the content and style representations at our desired layers. Since we're using eager our model is callable just like any other functions!
#   model_outputs = model(init_image)

#   style_output_features = model_outputs[:num_style_layers]
#   content_output_features = model_outputs[num_style_layers:]

#   style_score = 0
#   content_score = 0

#   # Accumulate style losses from all layers
#   # Here, we equally weight each contribution of each loss layer
#   weight_per_style_layer = 1.0 / float(num_style_layers)
#   for target_style, comb_style in zip(gram_style_features, style_output_features):
#     style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

#   # Accumulate content losses from all layers
#   weight_per_content_layer = 1.0 / float(num_content_layers)
#   for target_content, comb_content in zip(content_features, content_output_features):
#     content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)

#   style_score *= style_weight
#   content_score *= content_weight

#   # Get total loss
#   loss = style_score + content_score
#   return loss, style_score, content_score

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    """This function will compute the loss total loss.

    Arguments:
      model: The model that will give us access to the intermediate layers
      loss_weights: The weights of each contribution of each loss function.
        (style weight, content weight, and total variation weight)
      init_image: Our initial base image. This image is what we are updating with
        our optimization process. We apply the gradients wrt the loss we are
        calculating to this image.
      gram_style_features: Precomputed gram matrices corresponding to the
        defined style layers of interest.
      content_features: Precomputed outputs from defined content layers of
        interest.

    Returns:
      returns the total loss, style loss, content loss, and total variational loss
    """
    style_weight, content_weight = loss_weights

    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score

# def compute_grads(cfg):
#   with tf.GradientTape() as tape:
#     all_loss = compute_loss(**cfg)
#   # Compute gradients wrt input image
#   total_loss = all_loss[0]
#   return tape.gradient(total_loss, cfg['init_image']), all_loss

def compute_grads(cfg):
    with tf.GradientTape() as tape:
      all_loss = compute_loss(**cfg)
    # Compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


# Optimization loop
import IPython.display

def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    # We don't need to (or want to) train any layers of our model, so we set their trainable to false.
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    # Create our optimizer
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

    # For displaying intermediate images
    iter_count = 1

    # Store our best result
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
          'model' : model,
          'loss_weights': loss_weights,
          'init_image': init_image,
          'gram_style_features': gram_style_features,
          'content_features': content_features
    }

    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        if i % display_interval == 0:
            start_time = time.time()

            # Use the .numpy() method to get the concrete numpy array
            plot_img = init_image.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            IPython.display.clear_output(wait=True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration : {}'.format(i))
            print('Total loss: {:.4e}, '
                'style loss: {:.4e}, '
                'content loss: {:.4e}, '
                'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14,4))
    for i,img in enumerate(imgs):
        plt.subplot(num_rows, num_cols,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    return best_img, best_loss

best, best_loss = run_style_transfer(content_path,
                                     style_path, num_iterations=1000)
print(Image.fromarray(best))


# Visualize Outputs
def show_results(best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_path)
    style = load_img(style_path)

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')

    if show_large_final:
        plt.figure(figsize=(10, 10))

        plt.imshow(best_img)
        plt.title('Output Image')
        plt.show()

show_results(best, content_path, style_path)