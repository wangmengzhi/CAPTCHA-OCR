# CAPTCHA-OCR
图像验证码自动识别系统

## 开发准备

* ubuntu操作系统
* [PaddlePaddle文档](http://www.paddlepaddle.org/doc_cn/)

## 搭建环境

### 安装PaddlePaddle

* http://www.paddlepaddle.org/doc_cn/build_and_install/install/ubuntu_install.html

### 下载项目
```
git clone https://github.com/wangmengzhi/CAPTCHA-OCR.git
cd CAPTCHA-OCR
```

### 下载MNIST手写数字训练样本
* 从群共享里下载后解压到data文件夹里

### 把样本转为Paddle格式
```
sh preprocess.sh
```

### 训练
```
sh train.sh
/*
train.sh中的参数意义如下
use_gpu:是否使用gpu
trainer_count:训练线程数
num_passes:训练次数
*/
```

### 预测
```
sh predict.sh
/*
predict.sh中的参数意义如下
model:预测使用的模型
image:要预测的图片位置
*/
```
