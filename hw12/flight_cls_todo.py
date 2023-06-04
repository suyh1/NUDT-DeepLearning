# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import cv2
import os
import os.path as osp
import math
import random
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 160  # All images will be resized to 160x160
ORIGINAL_IMG_SIZE = 255
scale_f = IMG_SIZE / ORIGINAL_IMG_SIZE
LM_POINTS = ['Nose', 'Fuselage', 'Empennage', 'FLwing', 'FRwing', 'BLwing', 'BRwing']

data_root = pathlib.Path('./database/')
print(data_root)


def mk_ap_dataset(data_root):
    # 列出可用的类别标签：
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

    # 为每个类别标签分配索引：
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    index_to_label = dict((index, name) for index, name in enumerate(label_names))

    all_image_paths = list(data_root.glob('*/*jpg'))
    all_anno_paths = list(data_root.glob('*/*txt'))
    anno_to_img = dict(
        (p_img, p_anno) for p_img in all_image_paths for p_anno in all_anno_paths if p_img.stem == p_anno.stem)

    c_lables = np.zeros(len(anno_to_img))
    lm_labels = np.zeros((len(anno_to_img), 14))
    index = 0
    for img, anno in anno_to_img.items():
        with open(anno) as f_anno:
            lines = f_anno.readlines()
            c_lables[index] = label_to_index[img.parent.name]
            id_line = 0
            for line in lines:
                name_id, x, y = line.split()
                lm_labels[index][id_line * 2] = int(int(x) * scale_f)
                lm_labels[index][id_line * 2 + 1] = int(int(y) * scale_f)
                id_line += 1
        index += 1
    all_image_paths = [str(path) for path in all_image_paths]
    return all_image_paths, c_lables, lm_labels, index_to_label


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 128.0 - 1  # normalize to [-1,1] range

    return image


def deprocess_image(img):
    img = (img + 1.0) * 128
    return img


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def test():
    all_image_paths, all_cls_labels, all_lm_labels, index_to_label = mk_ap_dataset(data_root)

    # 可以在创建数据集之前就先打乱原数据
    # random shuffle the img and the label
    state = np.random.get_state()
    np.random.shuffle(all_image_paths)
    np.random.set_state(state)
    np.random.shuffle(all_cls_labels)
    np.random.set_state(state)
    np.random.shuffle(all_lm_labels)
    len_data = len(all_image_paths)

    def map_samples(path, cls_labels, lm_labels):
        return load_and_preprocess_image(path), {'c_pred': cls_labels, 'lm_pred': lm_labels}

    train_ds = tf.data.Dataset.from_tensor_slices((all_image_paths[0:int(len_data * 0.8)],
                                                   all_cls_labels[0:int(len_data * 0.8)],
                                                   all_lm_labels[0:int(len_data * 0.8)])).map(map_samples)
    val_ds = tf.data.Dataset.from_tensor_slices((all_image_paths[int(len_data * 0.8):int(len_data * 0.9)],
                                                 all_cls_labels[int(len_data * 0.8):int(len_data * 0.9)],
                                                 all_lm_labels[int(len_data * 0.8):int(len_data * 0.9)])).map(
        map_samples)
    test_ds = tf.data.Dataset.from_tensor_slices((all_image_paths[int(len_data * 0.9):],
                                                  all_cls_labels[int(len_data * 0.9):],
                                                  all_lm_labels[int(len_data * 0.9):])).map(map_samples)

    BATCH_SIZE = 8
    SHUFFLE_BUFFER_SIZE = 200

    # TODO 1:数据准备
    # 利用加载好的train_ds, val_ds 和 test_ds, 建立训练和评估用得train_batches、val_batches和test_batches
    # 提示：利用repeat\shuffle\batch\prefetch等进行配置
    train_batches = train_ds.repeat().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_batches = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_batches = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights=None)
    mobile_net.load_weights('./model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5')
    # mobile_net.summary() # 本语句不要在线上运行！！！
    mobile_net.trainable = False

    # TODO 2:优化迁移多任务学习网络结构
    # 一个简单的多任务迁移学习model构建例子，可以尝试图片分类、关键点标注效果，
    # 但是效果不一定很好，考虑修改自己的模型结构，提升分类和关键点标注效果
    layer_name = "block_10_expand_relu"  # 选择一个中间层用于进行关键点位置回归预测。中间层的layer_name可以通过mobile_net.summary()查看，任务说明中也已经给出。由于在线平台调试信息显示行数的限制，不推荐在线使用mobile_net.summary()，可以在自己的设备上进行测试
    x = mobile_net.outputs[0]

    # 注意，我们采用得mobile_net结构中，include_top选项是false。什么是include_top? no top意味着什么？ x = mobile_net.outputs[0]之后应该追加什么类型的层？
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # 之后可以使用一个或多个全连接层， 一般建议单层，节点数少于128，否则可能出现内存不足的问题
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    c_pred = tf.keras.layers.Dense(3, activation="softmax", name="c_pred")(x)  # 分类层，共有三类飞行器

    y = mobile_net.get_layer(layer_name).output

    # 同样，获取中间层输出后，应该如何处理？
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    # 之后可以使用一个或多个全连接层， 一般建议单层节点数少于128，否则可能出现内存不足的问题
    y = tf.keras.layers.Dense(128, activation='relu')(y)

    y = tf.keras.layers.Concatenate()([x, y])
    lm_pred = tf.keras.layers.Dense(14, activation='linear', name="lm_pred")(y)  # 关键点位置预测层，每张图片均有7个关键点，每个关键点要预测(x,y)坐标
    mt_model = tf.keras.Model(inputs=mobile_net.inputs, outputs=[c_pred, lm_pred], name='mt_model')
    # mt_model.summary()  # 本语句不要在线上运行！！！

    # TODO 3:完成多任务的model compile，同时请在lr_schedule中设置合理的weight_decay参数
    # initial_learning_rate可以修改，但是推荐使用0.001
    initial_learning_rate = 0.015
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1500,
        decay_rate=1e-5,
        staircase=True)
    mt_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                     loss={
                         'c_pred': 'sparse_categorical_crossentropy',
                         'lm_pred': 'mse'},  # 为两个任务选择合适的loss
                     loss_weights={
                         'c_pred': 1.,
                         'lm_pred': 1.
                     }, metrics={'c_pred': 'accuracy', 'lm_pred': 'mean_squared_error'})
    steps_per_epoch = np.ceil(len_data * 0.8 / BATCH_SIZE)
    v_steps_per_epoch = np.ceil(len_data * 0.1 / BATCH_SIZE)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="flight_mt_logs\\", histogram_freq=1)

    tf.keras.backend.set_learning_phase(1)

    # 由于服务器性能限制，不能训练过多迭代，推荐迭代次数8次以内，否则可能超时
    mt_model.fit(train_batches, epochs=8, validation_data=val_batches,
                 steps_per_epoch=steps_per_epoch, validation_steps=v_steps_per_epoch, callbacks=[tensorboard_callback],
                 verbose=2)
    tf.keras.backend.set_learning_phase(0)

    result = mt_model.evaluate(test_batches)
    result_dict = dict(zip(mt_model.metrics_names, result))

    for image_batch, label_dict in test_batches.take(1):
        imgs = image_batch.numpy()
        output = mt_model(image_batch)
        c_pred = output[0].numpy()
        lm_pred = output[1].numpy()
        for i in range(BATCH_SIZE):
            c_label = label_dict['c_pred'].numpy()
            lm_label = label_dict['lm_pred'].numpy()
            plt.figure()

            ax1 = plt.subplot(1, 2, 1)
            plt.title(index_to_label[int(c_label[i])])
            for j in range(len(LM_POINTS)):
                ax1.scatter(lm_label[i][j * 2], lm_label[i][j * 2 + 1])
                ax1.text(lm_label[i][j * 2] * 1.01, lm_label[i][j * 2 + 1] * 1.01, LM_POINTS[j], fontsize=10,
                         color="r", style="italic", weight="light", verticalalignment='center',
                         horizontalalignment='right', rotation=0)  # 给散点加标签
            plt.imshow(np.uint8(deprocess_image(imgs[i])))

            ax2 = plt.subplot(1, 2, 2)
            plt.title(index_to_label[np.argmax(c_pred[i])])
            for j in range(len(LM_POINTS)):
                ax2.scatter(lm_pred[i][j * 2], lm_pred[i][j * 2 + 1])
                ax2.text(lm_pred[i][j * 2] * 1.01, lm_pred[i][j * 2 + 1] * 1.01, LM_POINTS[j], fontsize=10,
                         color="r", style="italic", weight="light", verticalalignment='center',
                         horizontalalignment='right', rotation=0)  # 给散点加标签
            plt.imshow(np.uint8(deprocess_image(imgs[i])))
            plt.savefig("./test_figure/" + str(i + 1) + ".jpg")

    return result_dict

# 以下为测试代码，请勿解除注释
# coding=utf-8

# result_dict = test()

# if result_dict['c_pred_accuracy'] > 0.95 and result_dict['lm_pred_mean_squared_error'] < 3.0:
#     print("Success!")
# else:
#    print("Something wrong!")

