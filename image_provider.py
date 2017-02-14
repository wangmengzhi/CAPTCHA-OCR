# -*- coding: UTF-8 -*-
import io
import random

import paddle.utils.image_util as image_util
from paddle.trainer.PyDataProvider2 import *

@provider(input_types=[
    dense_vector(28 * 28),
    integer_value(62)
])
def processData(settings, file_name):
    """
    加载数据
    迭代每一批的所有图像和标签
    file_name: 批文件名
    """
    #使用pickle类来进行python对象的序列化，而cPickle提供了一个更快速简单的接口，如python文档所说的：“cPickle -- A faster pickle”
    data = cPickle.load(io.open(file_name, 'rb'))
    #list()方法用于将元组转换为列表，元组与列表的区别在于元组的元素值不能修改，元组是放在括号中，列表是放于方括号中。
    indexes = list(range(len(data['images'])))
    random.shuffle(indexes)
    for i in indexes:
        #加载图像，img:(K x H x W) ndarrays
        img = image_util.decode_jpeg(data['images'][i])
        label = data['labels'][i]
        '''
	    包含yield语句的函数会被特地编译成生成器。当函数被调用时，他们返回一个生成器对象
	    不像一般函数生成值后退出，生成器函数生成值后会自动挂起并暂停他们的执行和状态，他的本地变量将保存状态信息，这些信息在函数恢复时将再度有效
	    执行到 yield时，processData 函数就返回一个迭代值，下次迭代时，代码从 yield的下一条语句继续执行
	    '''
        yield img.flatten().tolist(), int(label)
