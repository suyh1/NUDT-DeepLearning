import numpy as np
import tensorflow.keras.datasets.mnist as mnist
import random
import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense,Lambda
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from sklearn.metrics import precision_recall_curve,roc_curve

num_classes = 10

def create_pairs_(x, labels , size):
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

def balanced_batch(batch_x,batch_y, num_cls):
    batch_size=len(batch_y)
    pos_per_cls_e=round(batch_size/2/num_cls/2)
    pos_per_cls_e*=2
    index=batch_y.argsort()
    ys_1=batch_y[index]
    #print(ys_1)
    num_class=[]
    pos_samples=[]
    neg_samples=set()
    cur_ind=0
    for item in set(ys_1):
        num_class.append((ys_1==item).sum())
        num_pos=pos_per_cls_e
        while(num_pos>num_class[-1]):
            num_pos-=2
        pos_samples.extend(np.random.choice(index[cur_ind:cur_ind+num_class[-1]],num_pos,replace=False).tolist())
        neg_samples=neg_samples|(set(index[cur_ind:cur_ind+num_class[-1]])-set(list(pos_samples)))
        cur_ind+=num_class[-1]
    neg_samples=list(neg_samples)
    x1_index=pos_samples[::2]
    x2_index=pos_samples[1:len(pos_samples)+1:2]
    x1_index.extend(neg_samples[::2])
    x2_index.extend(neg_samples[1:len(neg_samples)+1:2])
    p_index=np.random.permutation(len(x1_index))
    x1_index=np.array(x1_index)[p_index]
    x2_index=np.array(x2_index)[p_index]
    r_x1_batch=batch_x[x1_index]
    r_x2_batch=batch_x[x2_index]
    r_y_batch=np.array(batch_y[x1_index]!=batch_y[x2_index],dtype=np.float32)
    #return r_x1_batch, r_x2_batch,r_y_batch 输入样本对（ r_x1_batch, r_x2_batch）样本对正负例标签 r_y_batch
    return r_x1_batch,r_x2_batch,r_y_batch

def Creat_Pairs(image_set , label_set , size):
    '''
    :param image_set: 图片数据
    :param label_set: 标签数据
    :param size: 构建的数据集大小
    :return:正反例平衡，带图片是否相同标签的数据集
    '''
    numb_index = [np.where(label_set == i)[0] for i in range(num_classes)] #获取相同数字图片的索引

    pairs = []
    labels = []
    total_count = 0

    count_list_right = {}  # 设计为字典
    count_list_nega = {}  # 设计为字典

    num_each_right = int(size/2/10)  #每个数字创造的样本数
    num_each_nega = int(size/2/45)

    # 打包正例样本集
    flag = 0
    for i in range(num_classes):
        num_right = 0
        if flag == 1:
            break
        size_of_eachnumb = len(numb_index[i])
        for j in range(size_of_eachnumb):
            if j + 1 > num_each_right:
                continue
            index1,index2 = numb_index[i][j] , numb_index[i][j+1]   #相同数字的图片索引
            # pairs += [image_set[index1],image_set[index2]]          #相同数字的图片对
            pairs.append([image_set[index1],image_set[index2]])
            num_right += 1
            count_list_right["Positive(%d,%d)" % (i, i)] = num_right

            labels.append([0])
            total_count += 1
            if total_count >= size/2:
                flag = 1
                break

    # 打包反例样本集
    flag = 0
    for i in range(num_classes):
        if flag == 1:
            break
        size_of_eachnumb = len(numb_index[i])
        for j in range(i+1 , num_classes):
            num_nega = 0
            for k in range(num_each_nega):
                index1,index2 = numb_index[i][k] , numb_index[j][k]     #不同数字的图片索引
                # pairs += [image_set[index1],image_set[index2]]          #不同数字的图片对
                pairs.append([image_set[index1],image_set[index2]])
                # pairs.append(image_set[index1])
                # pairs.append(image_set[index2])
                num_nega += 1
                count_list_nega["Negtive(%d,%d)" % (i, j)] = num_nega

                labels.append([1])
                total_count += 1
                if total_count >= size:
                    flag = 1
                    break

    return np.array(pairs) , np.array(labels , dtype='float') , count_list_right , count_list_nega

def create_net(input_shape):
    input = Input(shape=input_shape)
    x = Dense(500, input_dim=784, activation='relu')(input)
    x = Dense(10, input_dim=500, activation='relu')(x)
    return Model(input, x)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def create_siamese():
    input_shape = (784,)

    base_network = create_net(input_shape)
    input_a = Input(shape=(input_shape))
    input_b = Input(shape=(input_shape))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = K.sqrt(K.sum(tf.square(processed_a - processed_b), axis=1))
    # distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    return model

def loss(label_pair, e_w):
    loss_p = (1 - label_pair) * 2 / Q * e_w ** 2
    loss_n = label_pair * 2 * Q * K.exp(-2.77 / Q * e_w)
    loss = K.mean(loss_p + loss_n)
    return loss

if __name__ == "__main__":
    # hyperparameters
    lr = 0.01
    Q = 5
    epochs = 20

    # data prepare
    (x_train , y_train) , (x_test , y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((60000, - 1))
    x_test = x_test.reshape((10000, - 1))

    # test_pairs , test_labels , test_list_right , test_list_nega = Creat_Pairs(x_test,y_test,9000)
    # train_pairs, train_labels, train_list_right, train_list_nega = Creat_Pairs(x_train, y_train, 27000)

    # r_x1_batch = np.array(train_pairs[:,0])
    # r_x2_batch = np.array(train_pairs[:,1])
    # r_y_batch = np.array(train_labels)
    #
    # e_x1_batch = np.array(test_pairs[:,0])
    # e_x2_batch = np.array(test_pairs[:,1])
    # e_y_batch = np.array(test_labels)

    r_x1_batch,r_x2_batch,r_y_batch = balanced_batch(x_train,y_train,30000)
    e_x1_batch,e_x2_batch,e_y_batch = balanced_batch(x_test,y_test,9000)

    print('r_x1_batch',np.shape(r_x1_batch))
    print('r_x2_batch',np.shape(r_x2_batch))

    print('r_x1_batch',np.shape(e_x1_batch))
    print('r_x2_batch',np.shape(e_x2_batch))

    # create siamese network
    model = create_siamese()

    plot_model(model, 'model.png', show_shapes=True, expand_nested=True, rankdir='TB')
    model.summary()

    Optimizer = tf.keras.optimizers.Adam(lr)
    auc_ = tf.keras.metrics.AUC()
    def auc(label_pair, e_w):
        e_w = tf.keras.layers.Flatten()(e_w)
        e_w = (e_w - K.min(e_w)) / (K.max(e_w) - K.min(e_w))
        label_pair = tf.keras.layers.Flatten()(label_pair)
        return auc_(label_pair, e_w)

    model.compile(loss=loss, optimizer=Optimizer, metrics=[auc])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs\\", histogram_freq=1)

    history = model.fit([r_x1_batch, r_x2_batch], r_y_batch,
                        batch_size=128,
                        epochs=epochs,
                        validation_data=([e_x1_batch, e_x2_batch], e_y_batch),
                        callbacks = [tensorboard_callback]
                        )

    plt.figure(0)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.legend(["auc","val_auc"])
    plt.show()

    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(["loss","val_loss"])
    plt.show()

    y_pred = model.predict([e_x1_batch, e_x2_batch])
    plt.figure(2)
    precision, recall, _thresholds = precision_recall_curve(e_y_batch, y_pred)
    plt.plot(recall, precision)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

