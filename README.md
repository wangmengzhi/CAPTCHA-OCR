# CAPTCHA-OCR
ͼ����֤���Զ�ʶ��ϵͳ

#
# ����׼��


* ubuntu 14.04

## �����


### ��װPaddlePaddle


* http://www.paddlepaddle.org/doc_cn/build_and_install/install/ubuntu_install.html


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
python test.py
```