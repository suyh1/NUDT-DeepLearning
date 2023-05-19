import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K


def test():
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

    # 测试也是255归一化的数据，请不要改归一化
    x_train = x_train / 255.
    x_test = x_test / 255.

    # WORK1: --------------BEGIN-------------------
    # 构建数据平衡采样方法：make_batch
    # 参数等都可以自定义
    # 返回值为(input_a, input_b), label
    # input_a形状为(batch_size,14,14,1),input_b形状为(batch_size,14,14,1),label形状为(batch_size,)
    # def make_batch(batch_size, dataset):
    #     return (input_a, input_b), label
    '''
    1. 从dataset中随机采样batch_size个样本，分别为x1_batch和x2_batch
    2. 从x1_batch中随机采样batch_size/2个样本，分别为x1_pos和x2_pos
    3. 从x2_batch中随机采样batch_size/2个样本，分别为x1_neg和x2_neg
    4. 将x1_pos和x2_pos组合成input_a，将x1_neg和x2_neg组合成input_b
    输入不使用老师那么麻烦的东西，直接用x_train,y_train,x_test,y_test 和 数量
    真不明白明明是处理数据，非要给数据封上一层是为啥
    输出和老师一样，返回正反例对和对应的标签(0为正，1为反)
    '''

    def make_batch(batch_x, batch_y, num_cls):
        batch_size = len(batch_y)
        pos_per_cls_e = round(batch_size / 2 / num_cls / 2)
        pos_per_cls_e *= 2
        index = batch_y.argsort()
        ys_1 = batch_y[index]
        # print(ys_1)
        num_class = []
        pos_samples = []
        neg_samples = set()
        cur_ind = 0
        for item in set(ys_1):
            num_class.append((ys_1 == item).sum())
            num_pos = pos_per_cls_e
            while (num_pos > num_class[-1]):
                num_pos -= 2
            pos_samples.extend(
                np.random.choice(index[cur_ind:cur_ind + num_class[-1]], num_pos, replace=False).tolist())
            neg_samples = neg_samples | (set(index[cur_ind:cur_ind + num_class[-1]]) - set(list(pos_samples)))
            cur_ind += num_class[-1]
        neg_samples = list(neg_samples)
        x1_index = pos_samples[::2]
        x2_index = pos_samples[1:len(pos_samples) + 1:2]
        x1_index.extend(neg_samples[::2])
        x2_index.extend(neg_samples[1:len(neg_samples) + 1:2])
        p_index = np.random.permutation(len(x1_index))
        x1_index = np.array(x1_index)[p_index]
        x2_index = np.array(x2_index)[p_index]
        r_x1_batch = batch_x[x1_index]
        r_x2_batch = batch_x[x2_index]
        r_y_batch = np.array(batch_y[x1_index] != batch_y[x2_index], dtype=np.float32)
        return (r_x1_batch, r_x2_batch), r_y_batch

    # WORK1: --------------END-------------------

    # WORK2: --------------BEGIN-------------------
    # 根据make_batch的设计方式，给出相应的train_set、val_set
    # 这两个数据集要作为make_batch(batch_size,dataset)的dataset参数，构成采样数据的来源
    # 忽略老师上面的提示，这里直接生成fit的输入就完事了
    (r_x1_batch, r_x2_batch), r_y_batch = make_batch(x_train, y_train, 30000)
    (e_x1_batch, e_x2_batch), e_y_batch = make_batch(x_test, y_test, 9000)

    # WORK2: --------------END-------------------

    # yield为啥用到这儿没看懂，反正用不到，直接注释掉
    # def data_generator(batch_size, dataset):
    #     while True:
    #         yield make_batch(batch_size, dataset)
    Q = 5

    # WORK3: --------------BEGIN-------------------
    # 实现损失函数
    def loss(y_true, y_pred):
        loss_p = (1 - y_true) * 2 / Q * y_pred ** 2
        loss_n = y_true * 2 * Q * K.exp(-2.77 / Q * y_pred)

        loss = K.mean(loss_p + loss_n)
        return loss

    # WORK3: --------------END-------------------

    # WORK4: --------------BEGIN-------------------
    # 构建siamese模型,输入为[input_a, input_b],输出为distance
    def create_net(input_shape):
        input = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Dense(200, activation='relu', name='dense1')(input)
        x = tf.keras.layers.Dense(10, activation='relu', name='dense2')(x)
        return tf.keras.Model(input, x)

    def build_model():
        # 注意，为防止梯度爆炸，对distance添加保护措施
        base_net = create_net((196,))
        input_a = tf.keras.layers.Input(shape=(14, 14, 1), name='data1')
        reshape_a = tf.keras.layers.Reshape((196,))(input_a)
        input_b = tf.keras.layers.Input(shape=(14, 14, 1), name='data2')
        reshape_b = tf.keras.layers.Reshape((196,))(input_b)

        a = base_net(reshape_a)
        b = base_net(reshape_b)

        distance = K.sqrt(K.sum(tf.square(a - b), axis=1))
        model = tf.keras.Model(inputs=[input_a, input_b], outputs=distance)
        return model

    # WORK4: --------------END-------------------

    model = build_model()
    plot_model(model, to_file='./test_figure/step1/model.png', show_shapes=True, expand_nested=True)

    # 注意，tf.keras.metrics.AUC()中，函数使用时默认正例标签为1，而在我们任务中，正例标签为0
    # 为了让我们定义的正例auc贯穿始终，用1-y_true和1-norm(y_pred)当作auc的标签和概率
    # (在我们的任务中，反例（y_true=1）的距离大，正例（y_true=0）的距离远，
    # 距离归一化norm(y_pred)后刚好符合反例（y_true=1）概率的变换趋势，1-norm(y_pred)就当作正例概率)
    auc_ = tf.keras.metrics.AUC()

    def auc(y_true, y_pred):
        y_pred = tf.keras.layers.Flatten()(y_pred)
        y_pred = 1 - (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred))
        y_true = 1 - tf.keras.layers.Flatten()(y_true)
        return auc_(y_true, y_pred)

    # WORK5: --------------BEGIN-------------------
    # 训练模型，参数可根据自己构建模型选取
    # 对于推荐的两层全连接模型，推荐参数如下：
    # 一般5-8个迭代以内auc可上0.97
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=loss,
                  metrics=[auc])
    model.fit([r_x1_batch, r_x2_batch], r_y_batch,
              validation_data=([e_x1_batch, e_x2_batch], e_y_batch),
              batch_size=128,
              epochs=5,
              verbose=2)
    # WORK5: --------------END-------------------
    return model
