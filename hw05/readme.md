### 任务描述

按要求构建SEblock Lenet，达到任务要求的性能 思考SeBlock的权重对卷积特征的影响。

### 相关知识

为了完成本关任务，你需要掌握：1.SEblock Lenet构建和使用方法，2.特征提取方法。

#### SENetBlock构建和使用方法

SENet (Squeeze-Excitation Net),其核心为设计SENetBlock去评估feature各个通道的权重，并用该权重给feature加权，用加权的feature当作实际的feature输出。如下图所示。 ![ senet ](https://data.educoder.net/api/attachments/784870) SENetBlock是一个很通用的模块，结构其实非常容易实现，也易于添加到我们自己的网络结构中去，提升现有效果。我们只需要在相关的层后面，添加一个SENetBlock层就可以了。 例如这个作业，我们就添加一个类似我们课上讲的SENetBlock,如下图所示： ![ senet block ](https://data.educoder.net/api/attachments/784924)

就是实现黄圈的这个分支，他需要经过如下计算：

```
def SeNetBlock(feature,reduction=4):    channels = #得到feature的通道数量c    avg_x = #先对feature的每个通道进行全局平均池化Global Average Pooling 得到通道描述子（Squeeze）    ...    x = #接着做reduction，用int(channels)//reduction个卷积核对 avg_x做1x1的点卷积    x = #接着用int(channels)个卷积核个数对 x做1x1的点卷积，扩展x回到原来的通道个数    cbam_feature = #对x 做 sigmoid 激活得到通道权重    return #返回以cbam_feature 为scale，对feature做拉伸加权的结果（Excitation） 
```

我们把SeNetBlock放在Lenet(基本lenet结构参考第四课卷积网资源里的tensorflow2_mnist.py)第二个卷积层后面（pooling之前），让他对有16个卷积核的第二个卷积层进行加权操作，这样得到的senet_lenet如下图所示: ![ selenet ](https://data.educoder.net/api/attachments/2807566)

#### 特征提取方法

如何得到网络特征xxfeature并作图？ 我们一般用构建特征模型的方式，用特征层作为输出output，原来的输入作为输入input,重新定义一个keras.model,并通过调用该模型前向计算得到特征。

例如对于SeNetBlock中的cbam_feature特征：

1. 在通过函数方式定义SeNetBlock计算图中， def SeNetBlock(feature,reduction=4)返回两个值 return x，cbam_feature。x是和后续层连接的输出，cbam_feature是其中的中间结果，也是我们需要绘制的特征。
2. 接下来，我们另定义一个 cbam_feature 模型 cbam_feature_model=model(input,output=cbam_feature )
3. 然后再组织好数据data， 前向一次cbam_feature_out=cbam_feature_model（data)，得到cbam_feature_out后，再reshape为batchsizex16的矩阵，然后用plt绘图

### 编程要求

根据注释提示，和所给结构图完成模型构建和训练,并输出通道注意力特征。

要求构建的模型与下图完全一致： ![ selenet ](https://data.educoder.net/api/attachments/2807566)

要求正确输出训练得到的通道注意力特征，示例如下： ![ cbamfeature ](https://data.educoder.net/api/attachments/2807572)

### 测试说明

通过图片比对和性能评估的方式进行测试。 输出的模型结构图片需和答案一致。 模型在MNIST测试集的性能须达到90%以上（注意，仅需要训练一个epoch，不要修改训练代码中的epoch数量，否则可能导致超时而无法完成作业） 能够正确输出通道注意力特征

对于有的同学在自己的环境中调试发现plot_model结构图中无法确定不定通道数为"?"或"None"的问题，请参考： https://stackoverflow.com/questions/63101709/keras-plot-model-replace-the-question-mark-by-none-in-the-model-architecture

------

开始你的任务吧，祝你成功！