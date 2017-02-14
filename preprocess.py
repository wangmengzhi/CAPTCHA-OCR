# -*- coding: UTF-8 -*-
from paddle.utils.preprocess_img import ImageClassificationDatasetCreater

#ImageClassificationDatasetCreater参数分别是图片路径，图片大小，图片有无颜色
data_creator = ImageClassificationDatasetCreater('data',28,0)
#每个训练文件包含的图片数
data_creator.num_per_batch = 1000
data_creator.overwrite = True
data_creator.create_batches()
