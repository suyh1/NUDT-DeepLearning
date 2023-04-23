### 任务描述

用如下网络结构对mnist数据集进行分类。 网络结构如图所示：

![ 网络结构 ](https://data.educoder.net/api/attachments/2657874) 其将图片样本分为上下两半X1,X2；分别送入input1,input2。后续的两个路径的线性加权模块 share_base(X)=X*W 共享一个变量 name='w'

整个分类模型可描述为 softmax( Dense（share_base(X1)+share_base(X2)+b)） 注意：b 是一个变量，share_base是一层全连接，没有偏置 如果构建模型正确，打印出来的图像应该像这样： ![ 打印出的网络结构 ](https://data.educoder.net/api/attachments/2694167) （如果你是TF 2.1请用from tensorflow.keras.utils import plot_model）

要求：1. 必须实现要求的共享网络结构，图像是上下分半，网络能够打印出类似下方给的网络结构图。2. 训练分类精度0.85以上

### 相关知识

1. 如何分割图片？ 利用tf.split函数：

   ```
   data1, data2 = tf.split(data, axis, num_or_size_splits)
   ```

   其中data为输入数据, axis指定分割维度。num_or_size_splits标明将数据沿指定维度分割成多少份。 需要注意的是，本作业数据集x_train的原始大小为(60000, 14, 14, 1), 因此如果分成上下两半，则得到的两块数据大小均为(60000, 7, 14, 1)

2. 如何将标签转换为one-hot格式？ 利用tf.one_hot函数即可

3. 如何生成数据集？ 利用tf.data.Dataset.from_tensor_slices。需要注意的是，本作业的数据集包括两个训练数据（即图片的上下两部分）+一个标签

4. 怎么实现共享变量的相同网络结构？ 线性加权模块 share_base需定义为一个子模块（sub_model),为了共享变量，实际整个网络中，只实例化了一个share_base对象。因此两个同样结构网络，共享变量W，实际上是只有一个实例化对象这样方式实现。例如：

```
share_base = tf.keras.Sequential([        ...    ], name='seq1')x1=Inputx2=Inputs1=shared_base(x1)s2=shared_base(x2)...
```

1. softmax( Dense（share_base(X1)+share_base(X2)+b)）怎么实现？ 利用代码中提供的自定义层class BiasPlusLayer(tf.keras.layers.Layer)，此自定义层实现了share_base(X1)+share_base(X2)+b的功能。

   ```
   x = BiasPlusLayer(..., name='BiasPlusLayer')([s1, s2])
   ```

   调用此自定义层之后，再追加一个全连接层， 将层定义中的激活函数定义为softmax即可。

2. 怎样得到网络层中的变量并进行可视化？ Keras是一个模型级库，本身不希望你去烦恼管理变量这件事。 但这里面变量其实都有名字，层也可以有名字，模型也可以有名字。你不命名，系统也会给他们一个默认的名字。（多个分层模块的模型，其命名是 模型名/一层模块名/二层模块名/。。。。/层名/变量名 也可以用scope命名空间管理） 我们可以给层和模型命名（name=xxx)用来找到这个层或模型。 例如：

   ```
   train_weights=xxx_net.get_layer('seq1').get_layer('D1').kernel.numpy()
   ```

   就是得到xxx_net模型下，名字为'seq1'的子模块下，名字为'D1'层的kernel变量（前提是我自己知道D1层是全连接Dense，它会有权值变量W，在keras下把全连接Dense下的权值矩阵W定义为Dense的kernel元素）

### 编程要求

网络结构与标准答案相同

精度达到0.85以上 ![img](https://data.educoder.net/api/attachments/668639)

### 测试说明

平台会对你编写的代码进行测试：

测试输出的网络结构图是否和标准答案一致

测试训练结束后的验证集准确率是否高于85%

------

开始你的任务吧，祝你成功！