# -*- coding: UTF-8 -*-
from paddle.trainer_config_helpers import *

is_predict = get_config_arg("is_predict", bool, False)

####################Data Configuration ##################
if not is_predict:
    define_py_data_sources2(
        train_list='data/batches/train.list',
        test_list='data/batches/test.list',
        module='image_provider',
        obj='processData')

######################Algorithm Configuration #############
settings(
    batch_size = 128,
    learning_rate = 0.1 / 128.0,
    learning_method = MomentumOptimizer(0.9),
    #权重衰减，防过拟合
    regularization = L2Regularization(0.0005 * 128)
)

#######################Network Configuration #############
img = data_layer(name='image',size=1*28*28)
# small_vgg is predined in trainer_config_helpers.network
predict = small_vgg(input_image=img,num_channels=1,num_classes=62)

if not is_predict:
    lbl = data_layer(name="label", size=62)
    outputs(classification_cost(input=predict, label=lbl))
else:
    #预测网络通常直接输出最后一层的结果而不是像训练时一样以cost layer作为输出
    outputs(predict)