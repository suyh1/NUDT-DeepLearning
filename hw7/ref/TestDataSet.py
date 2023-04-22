import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
# from tensorflow_core.python.keras.layers import Dense, Flatten, Conv2D, concatenate, Input, Activation,Lambda
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]

def create_pairs(x, labels , size):
    '''
    :param x: 输入数据集
    :param label: 数据标签
    :param size: 目标数据集大小
    :return: 处理完成的数据集
    '''

    '''分别取出10是数字'''
    from_0_to_9 = [[], [], [], [], [], [], [], [], [], []]
    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    for d in range(num_classes):
        size_of_eachnumb = len(digit_indices[d])
        for i in range(size_of_eachnumb):
            from_0_to_9[d].append(x[digit_indices[d][i]])

    image1 = []  # 存在第一个图像
    image2 = []  # 存放第二个图像
    label = []  # 存在label
    count_list_right = {}  # 设计为字典
    count_list_nega = {}  # 设计为字典

    '''下面考虑正例，每个数字占（size/2）的1/10'''
    num_each_right = int(size/2/10)  #每个数字创造的样本数
    num_each_nega = int(size / 2 / 45)
    for i in range(10):  #对每个数字做遍历
        isbreak = 0
        count =0 #当前样本数为0
        len_i = len(from_0_to_9[i]) # 获取当前这个数字对应的样本数量
        for j in  range(len_i): # 对这个数字对应的样本做遍历
            if isbreak:
                break
            for k in range(len_i): # 对这个数字对应的样本进行二重遍历
                if j==k :
                    continue
                image1.append(from_0_to_9[i][j])
                image2.append(from_0_to_9[i][k])
                label.append([0])
                count+=1
                if count>=num_each_right:
                    count_list_right["Positive(%d,%d)"%(i,i)]=count
                    isbreak = 1
                    break
                    #跳出两层for循环
    # print(count_list_right)
    # print(len(count_list_right),len(count_list_right)*num_each_right)

    '''下面考虑负样本'''
    for i in range(10):  # 对0-9遍历
        for j in range(i + 1, 10):  # 仍对0-9遍历
            for count in range(num_each_nega):  # 构造这num_each_nega多个样本数据
                image1.append(from_0_to_9[i][count])
                image2.append(from_0_to_9[j][count])
                label.append([1])
            count_list_nega["Negtive(%d,%d)" % (i, j)] = num_each_nega

    # print(count_list_nega)
    # print(len(count_list_nega), len(count_list_nega)*num_each_nega)

    '''组合数据'''
    data = []
    data.append(image1)
    data.append(image2)
    data.append(label)
    return data, count_list_right, count_list_nega

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

if __name__ =="__main__":
    size = 9000
    test_images = x_test
    test_labels = y_test
    test_data,count_right,count_nega = create_pairs(test_images, test_labels, size)
    num_each_right = int(size / 2 / 10)  # 每个数字创造的样本数
    num_each_nega = int(size / 2 / 45)


    print("Positive Sample:\n")
    for key, value in count_right.items():
        print('{key}:{value}'.format(key=key, value=value))
    print("Pos Total:\n", len(count_right) * num_each_right)

    print("Negtive Sample:\n")
    for key, value in count_nega.items():
        print('{key}:{value}'.format(key=key, value=value))
    print("Neg Total:\n", len(count_nega) * num_each_nega)

    print("Total: ", len(test_data[0]))
