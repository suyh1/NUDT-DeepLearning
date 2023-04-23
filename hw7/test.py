import numpy as np
import tensorflow as tf
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from mycode import test

# 1. 读取MNIST数据
print('start')
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

x_train = x_train / 255.
x_test = x_test / 255.

num_classes = 10

# 2. 对数据进行混洗
s_index = np.arange(len(y_test))
np.random.shuffle(s_index)
s_y_test = y_test[s_index]
s_x_test = x_test[s_index]

# 3. 将MNIST数据按类型连续排列，np.where返回满足条件的多维数组中的元素的坐标，
# 返回坐标的每一维合成一个tuple。标签只有一个维度，固选[0]
index1 = [np.where(s_y_test == i)[0] for i in range(num_classes)]
len_digits = [len(index1[i]) for i in range(num_classes)]

print('load data')


# 4. test_data_gen 产生测试数字对
# 类型0：同一数字，不同图片；标签：0；个数：每个数字800个配对，共800*10=8000个配对
# 产生方案：(index,（index+ran)%len)数字队列自交错配对,ran对每个数字集合一定
# 类型1：不同数字图片；标签：1；个数：每种不同数字组合200个配对，共200*45=9000个配对
# 产生方案：从每个数字图片集合中随机取一个图片配对
def test_data_gen(data, index):
    pairs_l = []
    pairs_r = []
    labels = []

    for c in range(num_classes):
        ran = random.randrange(1, len_digits[c])  # for each class: [1, classnum) rand list
        for i in range(800):
            i1, i2 = index[c][i], index[c][(i + ran) % len_digits[c]]
            pairs_l.append(data[i1])
            pairs_r.append(data[i2])
            labels.append(0)  # add positive samples (overall 800*10)
    for c in range(num_classes):
        for i in range(c):  # change c to num_classes
            if c == i:
                continue

            for _ in range(200):
                ran1 = random.randrange(1, len_digits[c])
                ran2 = random.randrange(1, len_digits[i])
                i1, i2 = index[c][ran1], index[i][ran2]
                pairs_l.append(data[i1])
                pairs_r.append(data[i2])
                labels.append(1)  # add negative samples (overall 200*45???)

    return (np.array(pairs_l), np.array(pairs_r)), np.array(labels).astype('float32')


test_pairs, test_y = test_data_gen(s_x_test, index1)
# s_index=np.arange(len(y_test))
s_index = np.arange(len(test_y))
np.random.shuffle(s_index)
test_pairs = (test_pairs[0][s_index], test_pairs[1][s_index])
test_y = test_y[s_index]

print('make data')
# 5. 读取模型进行测试
model = test()
loss, auc = model.evaluate(test_pairs, test_y, verbose=2, batch_size=64)
test_predictions = model.predict(test_pairs)


# 6. 绘制正例ROC曲线
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions, pos_label=0)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 100.5])
    plt.ylim([-0.5, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend(loc='lower right')
    plt.savefig("./test_figure/step1/roc.png")


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
true_labels = test_y.astype('uint8')
test_scores = 1 - (test_predictions - test_predictions.min()) / (test_predictions.max() - test_predictions.min())
plot_roc("My Model", true_labels, test_scores, color=colors[0])

if auc > 0.97:
    print('Success!')