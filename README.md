# CAPTCHA-OCR
ͼ����֤���Զ�ʶ��ϵͳ

#
# ����׼��


* ubuntu 14.04

## �����


### ��װOpenCV
```
sudo apt-get install libcv-dev
```


### ��װPaddlePaddle


* http://www.paddlepaddle.org/doc_cn/build_and_install/install/ubuntu_install.html


### ��װCUDA
```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

### ��װcuDNN


*
https://developer.nvidia.com/cudnn



### ��װMxNet

����mxnet
```
git clone --recursive https://github.com/dmlc/mxnet
```
��mxnet/make/config.mk���Ƶ�mxnet/���޸����¼��У�
```
USE_CUDA = 1
USE_CUDA_PATH = /usr/local/cuda
```
��mxnet/Ŀ¼�±���
```
make -j4
```
��װpython֧��
```
cd python
python setup.py install
```


### ��װtesseract
```

sudo apt-get install tesseract-ocr
```
### ��װpytesseract
```
sudo pip install pytesseract
```
### ��װpyqt4
```
sudo apt-get install libxext6 libxext-dev libqt4-dev libqt4-gui libqt4-sql qt4-dev-tools qt4-doc qt4-designer qt4-qtconfig "python-qt4-*" python-qt4
```
### ��װselenium
```
pip install selenium
```
### ������Ŀ

```

git clone https://github.com/wangmengzhi/CAPTCHA-OCR.git

cd CAPTCHA-OCR

```



### ������תΪPaddle��ʽ

```

python preprocess.py
```


### ѵ��

```

sh train.sh
```

### ����tesseract
* ��captcha��captcha_digits������/usr/share/tesseract-ocr/tessdata/configs/

### ����
```
python CAPTCHA-OCR.py
```