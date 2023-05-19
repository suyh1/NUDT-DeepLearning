在 TODO1 中，我们需要对 ReLU 层进行自定义以实现梯度修改。这里需要完成自定义梯度函数 `guided_grad`，并在其中实现反向传播时对梯度的修改。

具体地，我们可以通过将输入的梯度 `grad` 中小于等于 0 的部分替换为 0，从而实现对梯度的修正。具体实现如下：

```
Copy Codedef gg_relu(x):
    result = tf.keras.activations.relu(x)

    def guided_grad(grad):
        dtype = x.dtype
        condition = (x > 0)
        zero = tf.zeros_like(grad, dtype=dtype)
        return tf.where(condition, grad, zero)

    return result, guided_grad
```

在 `gg_relu` 中，我们首先计算原本的 ReLU 激活结果，并在内部定义了一个梯度函数 `guided_grad`。该函数以输入的梯度 `grad` 作为参数，返回一个梯度，表示最终输出对中间值的导数。具体地，我们使用 `tf.where` 函数将输入中所有小于等于 0 的位置替换为 0，并返回结果即可。

在 TODO2.1 中，我们需要实现获取 class saliency 的功能。根据定义，class saliency 本质上是损失函数对输入图像的梯度。因此，我们可以通过自定义梯度的方式来实现。

具体地，我们可以通过构造一个新的模型，让其输出 softmax 分类结果中指定类别的概率，然后计算其关于输入图像的梯度并返回。具体实现如下：

```
Copy Codedef get_class_saliency(model, image, category_index):
    tf_image = K.cast(image, dtype='float32')
    with tf.GradientTape() as tape:
        tape.watch(tf_image)
        predictions = model(tf_image)
        loss = predictions[:, category_index]

    saliency = tape.gradient(loss, tf_image)
    saliency = normalize(saliency)
    return saliency
```

在这里，我们首先将输入的图像转换为 `float32` 类型，并使用 `tf.GradientTape` 记录前向计算过程。在其中，我们直接对模型进行调用，以获得输出。然后，我们将损失函数设为预测结果中指定类别的概率，并计算关于输入图像的梯度。最后，我们对梯度进行归一化处理，并返回结果即可。

在 TODO2.2 中，我们需要实现获取 saliency 的功能。注意到这里的 saliency 本质上是损失函数对指定层输出的梯度。因此，我们可以通过类似的方式来实现。

具体地，我们可以构造一个新的模型，使其输出指定层的输出结果，然后计算其关于输入图像的梯度。具体实现如下：

```
Copy Codedef get_saliency(model, image, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output])

    tf_image = K.cast(image, dtype='float32')
    with tf.GradientTape() as tape:
        tape.watch(tf_image)
        conv_outputs, predictions = grad_model(tf_image)
        max_output = K.sum(K.max(conv_outputs, axis=3))

    saliency = tape.gradient(max_output, tf_image)
    saliency = normalize(saliency)
    return saliency
```

在这里，我们首先使用 `tf.keras.models.Model` 构造一个新的模型。该模型的输入为原始模型的输入，输出为指定层的输出和原始模型的最终预测结果。在 `tf.GradientTape` 中，我们计算指定层输出对整个模型输出的求和，并将其作为损失函数。然后，我们计算损失函数关于输入图像的梯度，并归一化处理后返回结果即可。

在 TODO4 中，我们需要对 gradcam 的通道加权进行计算。具体地，我们需要使用梯度的平均值来计算每个通道的权重，然后将每个通道的输出乘以相应的权重并求和。

具体实现如下：

```
Copy Codedef grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 1000

    grad_model = tf.keras.models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, category_index]

    grads = tape.gradient(loss, conv_outputs)
    grads_val = normalize(grads[0])

    output = conv_outputs[0]

    # calculate weights for each channel
    weights = np.mean(grads_val.numpy(), axis=(0, 1))

    cam = np.ones(output.shape[0:2], dtype=np.float32)

    # weighted sum of conv outputs channels
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # return result
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap
```

在这里，我们首先使用 `tf.keras.models.Model` 构造一个新的模型。该模型的输入为原始模型的输入，输出为指定层的输出和原始模型的最终预测结果（与 TODO2.2 类似）。在 `tf.GradientTape` 中，我们计算损失函数关于指定层输出的梯度，并对其进行归一化处理。然后，我们获取指定层输出的所有通道，并计算每个通道的权重。具体地，我们使用 `np.mean` 函数来计算均值，并将其作为每个通道的权重。

最后，在 `for` 循环中，我们遍历所有通道，将其与相应的权重相乘后累加到 `cam` 中。计算完毕后，我们对 `cam` 进行裁剪、归一化等处理，并返回结果即可。