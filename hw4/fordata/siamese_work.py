import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib as mpl
import sklearn
from sklearn import metrics
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import random

def dataModelPreparation():

    # 本地读取MNIST数据
    path = './dataset/mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    #为便于评测，图像尺寸缩小为原来的一半
    h = x_train.shape[1] // 2
    w = x_train.shape[2] // 2

    x_train = np.expand_dims(x_train, axis=-1)
    x_train = tf.image.resize(x_train, [h, w]).numpy()  # if we want to resize
    x_test = np.expand_dims(x_test, axis=-1)
    x_test = tf.image.resize(x_test, [h, w]).numpy()  # if we want to resize


    # 图像归一化,易于网络学习
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0


    #WORK1: --------------BEGIN-------------------
    #请补充完整训练集和测试集的产生方法：
    n_classes = 10
    # 请将训练集和测试集的标签转换为one-hot编码
    y_train = 
    y_test = 

    # 请补充建立训练集和测试集的代码
    train_datasets = 

    test_datasets = 
    #WORK1: ---------------END--------------------


    #WORK2: --------------BEGIN-------------------
    #请参考所给网络结构图，补充完整共享参数孪生网络siamese_net的实现：
    #注意，我们用比较图片的方法来评测网络结构是否正确
    #所以网络结构中的参数维度、名称等需和参考图中一致，否则不能通过评测
    inputs = tf.keras.Input(shape=(14, 14, 1), name='data')
    
    #WORK2: ---------------END--------------------  

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='conv_net')
    
    plot_model(model, to_file='./test_figure/step1/conv_net.png', show_shapes=True,expand_nested=True)
    
    return train_datasets, test_datasets, model

#WORK3: --------------BEGIN-------------------
#实例化网络并进行训练    
def test_fun():
    train_datasets, test_datasets, model = dataModelPreparation()
    #3.1 model compile，请选择适用于图像分类任务及one-hot编码的loss
    model.compile(
        optimizer=,
        loss=,
        metrics=['acc']
    )
    #3.2 配置训练参数，开始训练，
    history = model.fit(
        ,
        validation_data=,
        epochs=,
        verbose = 2
    )
    return model, history
#WORK3: ---------------END--------------------      
    
    
########### 以下为测试代码，评测时自动调用，请不要取消注释！！！ #####
# from siamese_work import test_fun
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np

# import os

# model, history = test_fun()

#1.用图片对比的方法测试网络结构是否正确
# test_img = mpimg.imread('./test_figure/step1/conv_net.png')
# answer_img= mpimg.imread('./path/to/answer.png')
# assert((answer_img == test_img).all())
# print('Network pass!')


#2.测试网络训练是否达标
# if history.history['val_acc'][-1] > 0.90:
#     print("Success!")