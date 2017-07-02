# CAPTCHA-OCR
图像验证码自动识别系统

#
# 开发准备


* ubuntu 14.04

## 搭建环境


### 安装OpenCV
```
sudo apt-get install libcv-dev
```


### 安装PaddlePaddle


* http://www.paddlepaddle.org/doc_cn/build_and_install/install/ubuntu_install.html


### 安装CUDA
```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

### 安装cuDNN


*
https://developer.nvidia.com/cudnn



### 安装MxNet

下载mxnet
```
git clone --recursive https://github.com/dmlc/mxnet
```
将mxnet/make/config.mk复制到mxnet/，修改以下几行：
```
USE_CUDA = 1
USE_CUDA_PATH = /usr/local/cuda
```
在mxnet/目录下编译
```
make -j4
```
安装python支持
```
cd python
python setup.py install
```


### 安装tesseract
```

sudo apt-get install tesseract-ocr
```
### 安装pytesseract
```
sudo pip install pytesseract
```
### 安装pyqt4
```
sudo apt-get install libxext6 libxext-dev libqt4-dev libqt4-gui libqt4-sql qt4-dev-tools qt4-doc qt4-designer qt4-qtconfig "python-qt4-*" python-qt4
```
### 安装selenium
```
pip install selenium
```
### 下载项目

```

git clone https://github.com/wangmengzhi/CAPTCHA-OCR.git

cd CAPTCHA-OCR

```



### 把样本转为Paddle格式

```

python preprocess.py
```


### 训练

```

sh train.sh
```

### 配置tesseract
* 将captcha和captcha_digits拷贝到/usr/share/tesseract-ocr/tessdata/configs/

### 运行
```
python CAPTCHA-OCR.py
```