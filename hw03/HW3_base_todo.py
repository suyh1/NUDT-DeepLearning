import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import time
import os
num_classes = 10
#1. Data preparation
def dataPreparation():
    path = './dataset/mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    h = x_train.shape[1] // 2
    w = x_train.shape[2] // 2

    x_train = np.expand_dims(x_train, axis=-1)
    x_train = tf.image.resize(x_train, [h, w]).numpy()  # if we want to resize
    x_test = np.expand_dims(x_test, axis=-1)
    x_test = tf.image.resize(x_test, [h, w]).numpy()  # if we want to resize

    print(x_train.shape)

    ###################### 任务1. prepare datasets ####################################
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # TODO: Add your codes here
    # (1) 将图片分割为上下两个部分：x_train1, x_train2, x_test1, x_test2
    x_train1, x_train2 = tf.split(x_train, num_or_size_splits=2, axis=1)
    x_test1, x_test2 = tf.split(x_test, num_or_size_splits=2, axis=1)
    # (2) 将标签转换为one_hot格式
    y_train = tf.one_hot(y_train, num_classes)
    y_test = tf.one_hot(y_test, num_classes)
    
    len_train1 = len(x_train1)
    len_test1 = len(x_test1)
    len_train2 = len(x_train2)
    len_test2 = len(x_test2)

    x_train1 = tf.reshape(x_train1, [len_train1, -1])
    x_test1 = tf.reshape(x_test1, [len_test1, -1])
    x_train2 = tf.reshape(x_train2, [len_train2, -1])
    x_test2 = tf.reshape(x_test2, [len_test2, -1])

    # (3) 利用tf.data.Dataset中的工具生成数据集
    train_datasets = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices((x_train1,x_train2)).batch(60000),
            tf.data.Dataset.from_tensor_slices(y_train).batch(60000)
        )
    )
    test_datasets = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices((x_test1,x_test2)).batch(60000),
            tf.data.Dataset.from_tensor_slices(y_test).batch(60000)
        )
    )
    
    ####################### 任务1 end ################################################
    return train_datasets, test_datasets


###################### 任务2. 自定义层建立 ##########################################
class BiasPlusLayer(tf.keras.layers.Layer):
    # TODO: Add your codes here
    def __init__(self, num_outputs, **kwargs):
        super(BiasPlusLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.bias = self.add_variable(name='b',shape=[self.num_outputs],initializer=tf.zeros_initializer())
    def build(self, input_shape):
        super(BiasPlusLayer, self).build(input_shape)
    def call(self, inputs):
        return inputs[0] + inputs[1] + self.bias
######################## 任务2 end ################################################

######################## 任务3. net_build #####################################
def BuildModel():
    # TODO: Add your codes here
    input1 = tf.keras.layers.Input(shape=(98,), name='Input1')
    input2 = tf.keras.layers.Input(shape=(98,), name='Input2')

    share_base = tf.keras.Sequential(
        [
        tf.keras.layers.Input(shape=(98,), name='D1_input'),
        tf.keras.layers.Dense(units=64,activation='softmax',name='D1'),
        ],
        name='seq1')

    s1 = share_base(input1)
    s2 = share_base(input2)

    x = BiasPlusLayer(64,name='BiasPlusLayer')([s1, s2])
    output = tf.keras.layers.Dense(units=10,activation='softmax',name='dense')(x)

    siamese_net = tf.keras.Model(inputs=[input1, input2], outputs=output, name='siamese_net')

    tf.keras.utils.plot_model(siamese_net, to_file='./test_figure/step1/siamese_net.png', show_shapes=True,
                              expand_nested=True)

    return siamese_net
########################## 任务3 end ########################################



def test_fun():
    siamese_net = BuildModel()
    train_datasets, test_datasets = dataPreparation()
####################### 任务4. train and test ################################
    # TODO: Add your codes here
    siamese_net.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits,optimizer=tf.keras.optimizers.Adam(learning_rate=0.2), metrics=['acc'])
    history = siamese_net.fit(train_datasets,batch_size=128,epochs=40,verbose=2,validation_data=(test_datasets))
###################### 任务4 end ###########################################
    return siamese_net, history


##################以下为测试用例，请勿取消注释！#######################
# import warnings
# warnings.filterwarnings("ignore")
# from HW3_base_todo import dataPreparation,test_fun
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# import os
#
# siamese_net, history = test_fun()
#
#1.用图片对比的方法测试网络结构是否正确
# test_img = mpimg.imread('./test_figure/step1/siamese_net.png')
# answer_img= mpimg.imread('./path/to/the/correct/image')
# assert((answer_img == test_img).all())
# print('Network pass!')

#2.测试网络训练是否达标
# print(history.history)
# if history.history['val_acc'][-1] > 0.85:
#    print("Success!")









