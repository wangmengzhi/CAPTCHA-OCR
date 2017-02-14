# -*- coding: UTF-8 -*-
# 网络配置文件
# created by wmz
# copyright USTC
# 2017-02-14

from paddle.trainer_config_helpers import *

is_predict = get_config_arg("is_predict", bool, False)

if not is_predict:
    define_py_data_sources2(
        train_list='data/batches/train.list',
        test_list='data/batches/test.list',
        module='image_provider',
        obj='processData')

settings(
    batch_size = 128,
    learning_rate = 0.1 / 128.0,
    learning_method = MomentumOptimizer(0.9),
    #权重衰减，防过拟合
    regularization = L2Regularization(0.0005 * 128)
)

img = data_layer(name='image',size=1*28*28)
predict = small_vgg(input_image=img,num_channels=1,num_classes=62)

if not is_predict:
    lbl = data_layer(name="label", size=62)
    outputs(classification_cost(input=predict, label=lbl))
else:
    outputs(predict)