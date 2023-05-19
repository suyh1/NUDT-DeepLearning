- [任务描述](https://www.educoder.net/tasks/rsfynfjx/860703/6ux3ypt8rv4z?coursesId=rsfynfjx#任务描述)
- 相关知识
  - [Guided Grad-CAM 算法](https://www.educoder.net/tasks/rsfynfjx/860703/6ux3ypt8rv4z?coursesId=rsfynfjx#guided grad-cam 算法)
  - [算子反向传播修改方法](https://www.educoder.net/tasks/rsfynfjx/860703/6ux3ypt8rv4z?coursesId=rsfynfjx#算子反向传播修改方法)
- [编程要求](https://www.educoder.net/tasks/rsfynfjx/860703/6ux3ypt8rv4z?coursesId=rsfynfjx#编程要求)
- [测试说明](https://www.educoder.net/tasks/rsfynfjx/860703/6ux3ypt8rv4z?coursesId=rsfynfjx#测试说明)

------

### 任务描述

使用Guided Grad-CAM算法对预训练分类网络VGG16进行可视化评估，看他们的grad cam和guided grad cam都在图像中指向了哪里？

### 相关知识

为了完成本关任务，你需要掌握：1.Guided Grad-CAM 算法；2. 算子反向传播修改方法。

#### Guided Grad-CAM 算法

Guided Grad-CAM简介：（回看反向传播课程中后半部分相关介绍，相关文章：Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization） Guided Grad-CAM将预训练模型中，某个特征层f 对输入图像img分类为c的图像依据以“热点图”展示出来。例如我们输入一张猫的图像，分类模型预测为“猫”,通过可视化我们可以发现是图像哪些部分支持这个分类结果，尖耳朵？胡须？还是猫脸？这些区域对分类的支持程度怎样？支持越高的区域，其在“热点图”上就越“热”。

![,](https://data.educoder.net/api/attachments/1726565) Guided Grad-CAM分为两部分：1. Guided-backpropagation 梯度图 与 2. Grad-CAM(Class Activation Mapping) 特征图。他们分别从不同的角度描述特征层f对图像分类的依据。其中，Guided-backpropagation 梯度图 从梯度的意义进行解释（输出对输入图像的梯度）：最能引起结果变化的图像区域分布（并且这些区域该“+”还是该“-“）；Grad-CAM 特征图 从feature map对结果的贡献进行解释：对该次分类结果，决策过程对不同feature的采纳程度，这个程度我们视为该类型对该feature map的“权重”。Grad-CAM即是加权的feature map。而Guided Grad-CAM 是 Guided-backpropagation 梯度图 与 Grad-CAM最终融合（相乘）的结果，显示从这两个角度都重要的图像分类依据。

(1)  Guided-backpropagation梯度: 与一般反向传播求梯度的过程唯一不同的地方在于对激活节点ReLU的传播处理, Guided-backpropagation梯度只传播“正”的梯度。注意，下图中f^out是指定输出，f_i^l是ReLU()的输入x, f_i^l+1是ReLU(x)输出。R_i^l+1 是上一层反向传播给ReLU层的梯度grad，也可以理解是ReLU上面一层的节点偏导。下图只表示对于ReLU层的处理。

![,](https://data.educoder.net/api/attachments/1726569) ![,](https://data.educoder.net/api/attachments/1726570) 通俗理解：反向传播梯度包含两部分：有则改之，多多益善。不支持结果的地方，有则改之(例如猫图像上的背景);支持的地方，多多益善（猫图像上的猫）。因此，完整梯度包含的信息比较杂乱。如果我们只关注支持的依据，则只选择梯度中的多多益善部分。如下图所示： ![,](https://data.educoder.net/api/attachments/1726571)

注意，作业中有从分类类型c当输出f_out和从考察的conv feature map最大通道当f_out两种模式。他们各自代表意义、区别和优缺点？

(2)  Grad-CAM 特征图：目的是看特征层f的各个通道f_i综合起来，对于分为类型c的特征贡献效果。Grad-CAM 是特征层f通道加权的特征图，其权重w_i的分配代表通道i对类型c的重要程度。例如，如果看block5_conv3的grad-cam。我们假象block5_conv3之后直接连接一个softmax分类（往往需要做Global average pooling），则分类器上类型c代表的节点连接各个通道的全连接权重，就代表了对类c而言，block5_conv3不同通道的重要程度。而我们感兴趣的网络层往往出些在网络的中间，权重怎么求？考虑梯度的物理意义，对节点的梯度∂c/∂f，即是最后节点对该节点的偏导，反映了该节点对结果改变的重要程度及方向。因此，我们用∂c/∂f_i的均值表示通道i的权重，这样对该层加权求得的热图也就是该层的Grad-CAM。

![,](https://data.educoder.net/api/attachments/1726573)

#### 算子反向传播修改方法

在函数def gg_relu(x)中进行修改，函数的前向传播部分与正常的relu一致，因此可以直接调用tf.keras.activations.relu。反向部分在def guided_grad中定义。需根据上文中给出的guided backpropagation公式，补齐函数中缺失的部分。需要注意的是，为了防止精度损失等问题，建议使用tf.cast函数将梯度的数据类型与输入x的数据类型设定为一致。

完成算子反向传播修改后，在replace_activation_layer_in_keras函数中对activation进行替换即可

### 编程要求

根据提示，在右侧编辑器补充代码（TODO1到TODO4），完成对算子反向传播的修改、获取saliency图像及gradcam图像等操作

### 测试说明

用cat_dog图像做示例 ![,](https://data.educoder.net/api/attachments/1726577) 如果选类型:分类top1，boxer dog,结果如下

![,](https://data.educoder.net/api/attachments/1726578) ![,](https://data.educoder.net/api/attachments/1726579) 如果选类型:分类top3，tigger cat,结果如下 ![,](https://data.educoder.net/api/attachments/1726580) ![,](https://data.educoder.net/api/attachments/1726581)

本次作业评测方法是比对输出图片与标准答案。注意，由于本次作业直接加载并使用预训练模型计算相关数值，不涉及模型训练，所以，只要代码正确，大家的输出应该都是一样的。

------

开始你的任务吧，祝你成功！