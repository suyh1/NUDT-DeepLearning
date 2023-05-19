from siamese_work import test_fun
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import os

model, history = test_fun()

#1.用图片对比的方法测试网络结构是否正确
# test_img = mpimg.imread('./test_figure/step1/conv_net.png')
# answer_img= mpimg.imread('./path/to/answer.png')
# assert((answer_img == test_img).all())
# print('Network pass!')


#2.测试网络训练是否达标
if history.history['val_acc'][-1] > 0.90:
    print("Success!")