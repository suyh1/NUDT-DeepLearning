from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.layers import Lambda, Multiply
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


# TODO 1: 对ReLU层进行自定义，实现梯度修改
'''
TODO 1
根据guided backpropagation公式，搜到了相关的实现：
https://www.coderskitchen.com/guided-backpropagation-with-pytorch-and-tensorflow/
@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,tf.float32)  * tf.cast(x>0,tf.float32) * dy
    return tf.nn.relu(x), grad
很相似啊很相似，改一改就能用力
'''


@tf.custom_gradient
def gg_relu(x):
    result = tf.keras.activations.relu(x)

    def guided_grad(grad):
        dtype = x.dtype
        return grad * tf.cast(grad > 0., dtype) * tf.cast(x > 0., dtype)

    return result, guided_grad


# TODO1 End


class GGReLuLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GGReLuLayer, self).__init__()

    def call(self, x):
        return gg_relu(x)  # you don't need to explicitly define the custom gradient
        # as long as you registered it with the previous method


# TODO2.1： 实现获取class saliency的功能。提示：class saliency的本质是损失函数对谁的梯度？
'''
TODO 2.1
class saliency 本质上是损失函数对输入图像的梯度。通过自定义梯度的方式实现。
老师给的代码首先将输入的图像转换为 float32 类型，然后使用 tf.GradientTape 记录前向计算过程。
补全代码predictions:直接对模型进行调用，获得输出。
然后老师的代码将损失函数设为预测结果中指定类别的概率，并计算关于输入图像的梯度。
最后，对梯度进行归一化处理，并返回结果。
'''


def get_class_saliency(model, image, category_index):
    tf_image = K.cast(image, dtype='float32')
    with tf.GradientTape() as tape:
        tape.watch(tf_image)  # watch谁？当然是tf_image，都给出来了，这不闭着眼填
        predictions = model(tf_image)
        loss = predictions[:, category_index]

    saliency = tape.gradient(loss, tf_image)[0]  # 如何得到saliency图像？
    # 实际上考察的是tape.gradient的用法。tape.gradient(loss, model.trainable_variables)
    saliency = normalize(saliency)
    return saliency


# TODO2.1 End

# TODO2.2： 实现获取saliency的功能，注意梯度计算与class saliency的区别
'''
TODO 2.2
梯度计算与class saliency的区别:本质上是损失函数对指定层输出的梯度
老师的代码首先使用 tf.keras.models.Model 构造一个新的模型
这个模型的输入为原始模型的输入，输出为指定层的输出和原始模型的最终预测结果
在 tf.GradientTape 中，计算指定层输出对整个模型输出的求和，作为损失函数
最后，计算损失函数关于输入图像的梯度，并归一化
'''


def get_saliency(model, image, layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    tf_image = K.cast(image, dtype='float32')
    with tf.GradientTape() as tape:
        tape.watch(tf_image)
        conv_outputs, predictions = grad_model(tf_image)
        max_output = K.sum(K.max(conv_outputs, axis=3))

    saliency = tape.gradient(max_output, tf_image)[0]
    saliency = normalize(saliency)
    return saliency


# TODO2.2 End

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
    if K.image_data_format() == 'channels_first':  # if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


'''
TODO 3 4
补全代码：用 tf.keras.models.Model 构造一个新的模型
和TODO2.2类似，模型的输入为原始模型的输入，输出为指定层的输出和原始模型的最终预测结果
在 tf.GradientTape 中，计算损失函数关于指定层输出的梯度，并归一化(其实上面的补全就是抄的这里的...)
然后，使用 np.mean 函数来计算均值，将其作为每个通道的权重
用for循环遍历所有通道，与相应的权重相乘后累加到 cam 中
计算完毕后，老师的代码对 cam 进行了裁剪、归一化等一系列处理
'''


def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 1000

    # TODO3: 建立用于获取中间层输出和梯度的模型。注意后续代码中的线索，此模型有两个输出，其中一个是根据'layer_name'定位的层输出，另一个是模型最终的输出
    grad_model = tf.keras.models.Model([input_model.inputs],
                                       [input_model.get_layer(layer_name).output, input_model.output])
    # TODO3: End

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        print('`output` has type {0}'.format(type(predictions)))
        loss = predictions[:, category_index]

    grads = tape.gradient(loss, conv_outputs)
    grads_val = normalize(grads[0])

    output = conv_outputs[0]

    # TODO4: using the weights to sum the channels of conv_outputs
    # 怎么给各通道加权？
    weights = np.mean(grads_val.numpy(), axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    # 计算conv_output各通道的加权和
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

        # TODO4 End

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
    # preprocessed_input = load_image(sys.argv[1])
    preprocessed_input = load_image("./img/cat_dog.png")

    model = VGG16(weights=None)
    model.load_weights('./model/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    predictions = model.predict(preprocessed_input)

    top_k = 5
    top_k_idx = np.argsort(predictions[0])[::-1][0:top_k]

    top_1 = decode_predictions(predictions)[0][0]
    print('Predicted class:')
    print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

    predicted_class = top_k_idx[2]  # np.argmax(predictions)
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


'''
import matplotlib.image as mpimg
from grad_cam_tf2 import test
import logging
logging.disable(30)

test()

#用图片对比的方法测试网络结构是否正确
test_img = mpimg.imread('./test_figure/step1/gradcam.jpg')
answer_img= mpimg.imread('./answer/gradcam.jpg')
assert((answer_img == test_img).all())
print('Grad-cam image pass!')

test_img = mpimg.imread('./test_figure/step1/guided_gradcam.jpg')
answer_img= mpimg.imread('./answer/guided_gradcam.jpg')
assert((answer_img == test_img).all())
print('guided_gradcam pass!')

test_img = mpimg.imread('./test_figure/step1/guided_gradcam_class.jpg')
answer_img= mpimg.imread('./answer/guided_gradcam_class.jpg')
assert((answer_img == test_img).all())
print('guided_gradcam_class pass!')

print('Success!')
'''
