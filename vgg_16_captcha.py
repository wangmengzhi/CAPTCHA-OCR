from paddle.trainer_config_helpers import *

is_predict = get_config_arg("is_predict", bool, False)

####################Data Configuration ##################
if not is_predict:
  data_dir='data/batches/'
  meta_path=data_dir+'batches.meta'

  args = {'meta':meta_path,'mean_img_size': 28,
          'img_size': 28,'num_classes': 10,
          'use_jpeg': 1,'color': 0}

  define_py_data_sources2(train_list=data_dir+"train.list",
                          test_list=data_dir+'test.list',
                          module='image_provider',
                          obj='processData',
                          args=args)

######################Algorithm Configuration #############
settings(
    batch_size = 128,
    learning_rate = 0.1 / 128.0,
    learning_method = MomentumOptimizer(0.9),
    regularization = L2Regularization(0.0005 * 128)
)

#######################Network Configuration #############
data_size=1*28*28
label_size=10
img = data_layer(name='image',
                 size=data_size)
# small_vgg is predined in trainer_config_helpers.network
predict = small_vgg(input_image=img,
                    num_channels=1,
                    num_classes=label_size)

if not is_predict:
    lbl = data_layer(name="label", size=label_size)
    outputs(classification_cost(input=predict, label=lbl))
else:
    outputs(predict)

