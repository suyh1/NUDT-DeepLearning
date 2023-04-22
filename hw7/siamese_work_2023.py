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

    #测试也是255归一化的数据，请不要改归一化
    x_train = x_train / 255.
    x_test = x_test/ 255.
	

	#WORK1: --------------BEGIN-------------------
    #构建数据平衡采样方法：make_batch
	#参数等都可以自定义
	#返回值为(input_a, input_b), label
	#input_a形状为(batch_size,14,14,1),input_b形状为(batch_size,14,14,1),label形状为(batch_size,)
	def make_batch(batch_size,dataset):

		return (input_a, input_b), label
    #WORK1: --------------END-------------------
    
	#WORK2: --------------BEGIN-------------------
	#根据make_batch的设计方式，给出相应的train_set、val_set
	#这两个数据集要作为make_batch(batch_size,dataset)的dataset参数，构成采样数据的来源
	train_set = 
    val_set = 
	#WORK2: --------------END-------------------

    def data_generator(batch_size,dataset):
        while True:
            yield make_batch(batch_size,dataset)
    
    Q = 5
    #WORK3: --------------BEGIN-------------------
    # 实现损失函数
    def loss(label_pair, e_w):
        loss_p = (1 - label_pair) * 2 / Q * e_w ** 2
        loss_n = label_pair * 2 * Q * K.exp(-2.77 / Q * e_w)

        loss = K.mean(loss_p + loss_n)
        return loss
    # WORK3: --------------END-------------------

    #WORK4: --------------BEGIN-------------------
	#构建siamese模型,输入为[input_a, input_b],输出为distance
    def build_model():
        
    
        #注意，为防止梯度爆炸，对distance添加保护措施
		distance = K.sqrt(K.sum(tf.square(hidden_a - hidden_b), axis=1))
        model = tf.keras.Model(inputs=, outputs=distance)

        return model 
    #WORK4: --------------END-------------------
    
	model=build_model()
    plot_model(model, to_file='./test_figure/step1/model.png', show_shapes=True,expand_nested=True)
        
    #注意，tf.keras.metrics.AUC()中，函数使用时默认正例标签为1，而在我们任务中，正例标签为0
    #为了让我们定义的正例auc贯穿始终，用1-y_true和1-norm(y_pred)当作auc的标签和概率
    #(在我们的任务中，反例（y_true=1）的距离大，正例（y_true=0）的距离远，
    #距离归一化norm(y_pred)后刚好符合反例（y_true=1）概率的变换趋势，1-norm(y_pred)就当作正例概率)
    auc_ = tf.keras.metrics.AUC()
	def auc(y_true, y_pred):
        y_pred = tf.keras.layers.Flatten()(y_pred)
        y_pred = 1-(y_pred - K.min(y_pred))/(K.max(y_pred) - K.min(y_pred))
        y_true = 1-tf.keras.layers.Flatten()(y_true)
        return auc_(y_true, y_pred)
    
	#WORK5: --------------BEGIN-------------------
	#训练模型，参数可根据自己构建模型选取
	#对于推荐的两层全连接模型，推荐参数如下：
	#一般5-8个迭代以内auc可上0.97
    model.compile(optimizer=, loss=, metrics=[auc])
    model.fit(, verbose = 2)
    #WORK5: --------------END-------------------
    return model
	
'''
import numpy as np
import tensorflow as tf
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from siamese_work import test

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
x_test = x_test/ 255.

num_classes = 10

#2. 对数据进行混洗
s_index=np.arange(len(y_test))
np.random.shuffle(s_index)
s_y_test=y_test[s_index]
s_x_test=x_test[s_index]

#3. 将MNIST数据按类型连续排列，np.where返回满足条件的多维数组中的元素的坐标，
#返回坐标的每一维合成一个tuple。标签只有一个维度，固选[0]
index1 = [np.where(s_y_test == i)[0] for i in range(num_classes)]
len_digits=[len(index1[i]) for i in range(num_classes)]

print('load data')

#4. test_data_gen 产生测试数字对
#类型0：同一数字，不同图片；标签：0；个数：每个数字800个配对，共800*10=8000个配对
#产生方案：(index,（index+ran)%len)数字队列自交错配对,ran对每个数字集合一定
#类型1：不同数字图片；标签：1；个数：每种不同数字组合200个配对，共200*45=9000个配对
#产生方案：从每个数字图片集合中随机取一个图片配对
def test_data_gen(data, index):
    pairs_l = []
    pairs_r = []
    labels = []
 
    for c in range(num_classes):
        ran = random.randrange(1, len_digits[c])# for each class: [1, classnum) rand list
        for i in range(800):
            i1, i2 = index[c][i], index[c][(i + ran)%len_digits[c]]
            pairs_l.append(data[i1])
            pairs_r.append(data[i2])
            labels.append(0) # add positive samples (overall 800*10)
    for c in range(num_classes):
        for i in range(c): # change c to num_classes
            if c == i:
                continue

            for _ in range(200):
                ran1 = random.randrange(1, len_digits[c])
                ran2 = random.randrange(1, len_digits[i])
                i1, i2 = index[c][ran1], index[i][ran2]
                pairs_l.append(data[i1])
                pairs_r.append(data[i2])
                labels.append(1) # add negative samples (overall 200*45???)

    return (np.array(pairs_l),np.array(pairs_r)), np.array(labels).astype('float32')


test_pairs, test_y = test_data_gen(s_x_test, index1)
# s_index=np.arange(len(y_test))
s_index=np.arange(len(test_y))
np.random.shuffle(s_index)
test_pairs=(test_pairs[0][s_index],test_pairs[1][s_index])
test_y=test_y[s_index]

print('make data')
#5. 读取模型进行测试
model=test()
loss,auc = model.evaluate(test_pairs,test_y,verbose=2,batch_size=64)
test_predictions = model.predict(test_pairs)

#6. 绘制正例ROC曲线
def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions, pos_label=0)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,100.5])
  plt.ylim([-0.5,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')
  plt.legend(loc='lower right')
  plt.savefig("./test_figure/step1/roc.png")


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
true_labels=test_y.astype('uint8')
test_scores = 1-(test_predictions - test_predictions.min())/(test_predictions.max() - test_predictions.min())
plot_roc("My Model", true_labels, test_scores, color=colors[0])

if auc>0.97:
    print('Success!')
'''	
