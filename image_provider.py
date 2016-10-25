import io
import random

import paddle.utils.image_util as image_util
from paddle.trainer.PyDataProvider2 import *


#
# {'img_size': 28,
# 'settings': <paddle.trainer.PyDataProviderWrapper.Cls instance at 0x7fea27cb6050>,
# 'color': False,
# 'mean_img_size': 28,
# 'meta': './data/batches/batches.meta',
# 'num_classes': 10,
# 'file_list': ('./data/batches/train_batch_000',),
# 'use_jpeg': True}
def hook(settings, img_size, mean_img_size, num_classes, color, meta, use_jpeg,
         is_train, **kwargs):
    settings.mean_img_size = mean_img_size
    settings.img_size = img_size
    settings.num_classes = num_classes
    settings.color = color
    settings.is_train = is_train

    if settings.color:
        settings.img_raw_size = settings.img_size * settings.img_size * 3
    else:
        settings.img_raw_size = settings.img_size * settings.img_size

    settings.meta_path = meta
    settings.use_jpeg = use_jpeg

    settings.img_mean = image_util.load_meta(settings.meta_path,
                                             settings.mean_img_size,
                                             settings.img_size,
                                             settings.color)

    settings.logger.info('Image size: %s', settings.img_size)
    settings.logger.info('Meta path: %s', settings.meta_path)
    settings.input_types = [
        dense_vector(settings.img_raw_size),  # image feature
        integer_value(settings.num_classes)]  # labels

    settings.logger.info('DataProvider Initialization finished')


@provider(init_hook=hook)
def processData(settings, file_name):
    """
    The main function for loading data.
    Load the batch, iterate all the images and labels in this batch.
    file_name: the batch file name.
    """
    data = cPickle.load(io.open(file_name, 'rb'))
    indexes = list(range(len(data['images'])))
    if settings.is_train:
        random.shuffle(indexes)
    for i in indexes:
        if settings.use_jpeg == 1:
            img = image_util.decode_jpeg(data['images'][i])
        else:
            img = data['images'][i]
        img_feat = image_util.preprocess_img(img, settings.img_mean,
                                             settings.img_size, settings.is_train,
                                             settings.color)
        label = data['labels'][i]
        yield img_feat.tolist(), int(label)
