from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.layers import Layer, Lambda, Multiply
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import tensorflow.keras
import matplotlib.pyplot as plt
import sys
import cv2


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def load_image(path):
    # img_path = sys.argv[1]
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# 对ReLU层进行自定义，实现梯度修改
@tf.custom_gradient
def gg_relu(x):
    result = tf.keras.activations.relu(x)

    def guided_grad(grad):
        dtype = x.dtype
        return grad * tf.cast(grad > 0., dtype) * tf.cast(x > 0., dtype)

    return result, guided_grad


class GGReLuLayer(Layer):
    def __init__(self):
        super(GGReLuLayer, self).__init__()

    def call(self, x):
        return gg_relu(x)


# 实现获取class saliency的功能。提示：class saliency的本质是损失函数对谁的梯度？
def get_class_saliency(model, image, category_index):
    tf_image = K.cast(image, dtype='float32')
    with tf.GradientTape() as tape:
        tape.watch(tf_image)
        predictions = model(tf_image)
        loss = predictions[:, category_index]

    saliency = tape.gradient(loss, tf_image)[0]
    saliency = normalize(saliency)
    return saliency


# 实现获取saliency的功能，注意梯度计算与class saliency的区别
def get_saliency(model, image, layer_name):
    grad_model = tf.keras.models.Model(model.inputs, model.get_layer(layer_name).output)

    tf_image = K.cast(image, dtype='float32')
    with tf.GradientTape() as tape:
        tape.watch(tf_image)
        output = grad_model(tf_image)
        loss = tf.reduce_mean(output)

    grads = tape.gradient(loss, tf_image)
    saliency = normalize(grads[0])
    return saliency


def replace_activation_layer_in_keras(model, replaced_layer, new_layer):
    layers = [l for l in model.layers]
    new_model = model
    for i in range(1, len(layers)):
        if hasattr(layers[i], 'activation'):
            if layers[i].activation == replaced_layer:
                config = layers[i].get_config()
                layerClass = getattr(tf.keras.layers, layers[i].__class__.__name__)
                copyLayer = layerClass.from_config(config)
                new_model.layers[i] = copyLayer
                new_model.layers[i].set_weights(layers[i].get_weights())
                new_model.layers[i].activation = new_layer()

    return new_model


def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 建立用于获取中间层输出和梯度的模型。注意后续代码中的线索，此模型有两个输出，其中一个是根据'layer_name'定位的层输出，另一个是模型最终的输出
def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 1000

    grad_model = tf.keras.models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        print('`output` has type {0}'.format(type(predictions)))
        loss = predictions[:, category_index]

    grads = tape.gradient(loss, conv_outputs)
    grads_val = normalize(grads[0])

    output = conv_outputs[0]

    # 怎么给各通道加权？
    weights = np.mean(grads_val.numpy(), axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    # 计算conv_output各通道的加权和
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def test():
    preprocessed_input = load_image("./img/cat_dog.png")

    model = VGG16(weights=None)
    model.load_weights('./model/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    predictions = model.predict(preprocessed_input)

    top_k = 5
    top_k_idx = np.argsort(predictions[0])[::-1][0:top_k]

    top_1 = decode_predictions(predictions)[0][0]
    print('Predicted class:')
    print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

    predicted_class = top_k_idx[2]
    predicted_tensor = K.one_hot([predicted_class], 1000)

    cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
    cv2.imwrite("./test_figure/step1/gradcam.jpg", cam)

    guided_model = replace_activation_layer_in_keras(model, tf.keras.activations.relu, GGReLuLayer)

    saliency = get_saliency(guided_model, preprocessed_input, "block5_conv3")
    gradcam = saliency * heatmap[..., np.newaxis]
    save_img("./test_figure/step1/guided_gradcam.jpg", deprocess_image(gradcam.numpy()))

    saliency = get_class_saliency(guided_model, preprocessed_input, predicted_class)
    gradcam = saliency * heatmap[..., np.newaxis]
    save_img("./test_figure/step1/guided_gradcam_class.jpg", deprocess_image(gradcam.numpy()))