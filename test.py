# -*- coding: UTF-8 -*-
from PyQt4 import QtCore, QtGui
from ui import Ui_mainWindow
import cv2
import urllib
import urllib2
import os
import math
import numpy as np
import pytesseract
import PIL
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
import time
import platform
import re
import cookielib
import sys

reload(sys)
sys.setdefaultencoding('utf8')

if platform.system()=="Linux":
    from py_paddle import swig_paddle, DataProviderConverter
    from paddle.trainer.PyDataProvider2 import dense_vector
    from paddle.trainer.config_parser import parse_config
    from paddle.utils import preprocess_util

    label_set = preprocess_util.get_label_set_from_dir('data/test')

    conf = parse_config("vgg_16_captcha.py", "is_predict=1")
    print conf.data_config.load_data_args
    swig_paddle.initPaddle("--use_gpu=0")
    network = swig_paddle.GradientMachine.createFromConfigProto(conf.model_config)
    network.loadParameters("captcha_vgg_model/pass-00009/")
    converter = DataProviderConverter([dense_vector(28*28)])

def predict(image=None):
    img=np.array(PIL.Image.open(image))
    data=[[]]
    data[0].append(img.flatten().tolist())
    inArg = converter(data)
    network.forwardTest(inArg)
    output = network.getLayerOutputs("__fc_layer_1__")
    return output["__fc_layer_1__"][0]

def show(img,name):
    name = name.decode('utf-8').encode('gbk')
    print name
    img = cv2.resize(img,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#将img放在28*28图像的中央
def resize(img):
    #将原图大小控制在20*20内
    height,width = img.shape
    k = 20.0 / max(height,width)
    #CV_INTER_CUBIC双三次插值
    img = cv2.resize(img,None,fx=k,fy=k,interpolation=cv2.INTER_CUBIC)
    height,width = img.shape
    t = np.zeros((28,28),dtype=np.uint8)
    x = (28 - width) / 2
    y = (28 - height) / 2
    #t[y:y+height]表示从t[y]到t[y+height-1]不包括t[y+height]
    t[y:y+height,x:x+width]=img
    return t

#继承的类和生成的ui对象
class Ui(QtGui.QDialog):
    output_path = 'temp/'
    if not(os.path.exists(output_path)):
        os.makedirs(output_path)
    current = -1
    img = None
    bak = None
    tresult = ''#tesseract结果
    vresult = ''#vgg结果
    mode = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    def kdbbs(self):
        self.ui.le_dl.setText('http://bbs.ustc.edu.cn/cgi/bbscaptcha')
        self.ui.le_dt.setText("50")
        self.ui.le_divide_thresh.setText("5")
        self.ui.cb_dm.setCurrentIndex(3)
        self.ui.cb_divide_method.setCurrentIndex(1)
        self.ui.cb_type.setCurrentIndex(1)

    def kdyjs(self):
        self.ui.le_dl.setText('http://yjs.ustc.edu.cn/checkcode.asp')
        self.ui.le_dt.setText("10")
        self.ui.le_divide_thresh.setText("20")
        self.ui.cb_dm.setCurrentIndex(0)
        self.ui.cb_divide_method.setCurrentIndex(0)
        self.ui.cb_type.setCurrentIndex(0)

    def csdn(self):
        self.ui.le_dl.setText('http://passport.csdn.net/ajax/verifyhandler.ashx')
        self.ui.le_divide_thresh.setText("5")
        self.ui.cb_divide_method.setCurrentIndex(1)
        self.ui.cb_type.setCurrentIndex(1)
        
    def iqiyi(self):
        self.ui.le_dl.setText('http://passport.iqiyi.com/register/vcode.php')
        self.ui.cb_type.setCurrentIndex(1)

    def le(self):
        self.ui.le_dl.setText('https://sso.le.com/verify')
        self.ui.cb_type.setCurrentIndex(1)

    def __init__(self, parent=None):
        super(Ui, self).__init__(parent)
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)
        self.ui.rb_dl.setChecked(True)
        self.setFixedSize(800, 600)

    def rec(self):
        self.tresult=self.vresult=''
        if self.ui.rb_csdn.isChecked():
            #self.gray()
            #self.bin()
            s = pytesseract.image_to_string(PIL.Image.open(self.output_path + 'captcha.png'),lang="eng",config="-psm 7 captcha")
            s = s.replace(' ', '')
            self.ui.le_result.setText(s)
            return

        if self.ui.rb_kdbbs.isChecked():
            #不能放大，否则影响去噪效果
            self.denoise()
            self.gray()
            self.bin()
            self.dilate()
            self.medianBlur()

        if self.ui.rb_kdyjs.isChecked():
            #放大，防膨胀粘连,连通域面积为0
            self.zoom()
            self.gray()
            self.bin()
            self.dilate()
            self.denoise()
            self.medianBlur()
            #self.tresult = pytesseract.image_to_string(PIL.Image.open(self.output_path+'captcha.png'),lang="eng",config="-psm 7 captcha_digits")
            #self.tresult=self.tresult.replace(' ','')
            #return

        #普通模式
        self.mode = 1
        while self.divide():
            pass
        self.mode = 0

    def zoom(self):
        self.bak = self.img.copy()
        #长宽各放大2倍
        self.img = cv2.resize(self.img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(self.output_path + 'zoom.png', self.img)
        self.ui.lb_cap.setPixmap(QtGui.QPixmap(self.output_path + "zoom.png"))

    def last(self):
        if not self.ui.rb_lc.isChecked():
            QtGui.QMessageBox.information(self,u"错误",u"本功能仅在读取本地图片时可用！")
            return
        if(self.current < 1):
            QtGui.QMessageBox.information(self,u"错误",u"已经是第一张图片！")
            return
        self.tresult = self.vresult = ''
        self.current-=1
        self.bak = self.img.copy()
        path=str(self.ui.le_lc.text() + '/' + self.files[self.current])
        self.img = cv2.imread(path)
        self.ui.lb_cap.setPixmap(QtGui.QPixmap(path))

    def next(self):
        self.tresult = self.vresult = ''
        self.current+=1
        if self.img is not None:
            self.bak = self.img.copy()
        if self.ui.rb_dl.isChecked():
            #伪装下载的http请求，否则有些站点不响应下载请求
            urllib.URLopener.version = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:51.0) Gecko/20100101 Firefox/51.0'
            path=self.output_path + "captcha"
            urllib.urlretrieve(str(self.ui.le_dl.text()), path)
        elif self.ui.rb_lc.isChecked():
            if(self.current == len(self.files)):
                QtGui.QMessageBox.information(self,u"错误",u"已经是最后一张图片！")
                return
            path=str(self.ui.le_lc.text() + '/' + self.files[self.current])
        self.img = cv2.imread(path)
        cv2.imwrite(self.output_path + 'captcha.png', self.img)
        self.ui.lb_cap.setPixmap(QtGui.QPixmap(self.output_path + 'captcha.png'))

    def gray(self):
        self.bak = self.img.copy()
        #灰度化
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.output_path + 'gray.png', self.img)
        self.ui.lb_cap.setPixmap(QtGui.QPixmap(self.output_path + "gray.png"))

    def bin(self):
        self.bak = self.img.copy()
        #二值化
        thresh,self.img = cv2.threshold(self.img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(self.output_path + 'bin.png', self.img)
        self.ui.lb_cap.setPixmap(QtGui.QPixmap(self.output_path + "bin.png"))
        
    def dilate(self):
        self.bak = self.img.copy()
        #膨胀
        self.img = cv2.dilate(self.img, self.kernel)
        cv2.imwrite(self.output_path + 'dilate.png', self.img)
        self.ui.lb_cap.setPixmap(QtGui.QPixmap(self.output_path + "dilate.png"))
        
    def erode(self):
        self.bak = self.img.copy()
        #腐蚀
        self.img = cv2.erode(self.img, self.kernel)
        cv2.imwrite(self.output_path + 'erode.png', self.img)
        self.ui.lb_cap.setPixmap(QtGui.QPixmap(self.output_path + "erode.png"))

    def equalizeHist(self):
        self.bak = self.img.copy()
        #直方图均衡化使图像更清晰,将比较淡的图像变换为比较深的图像（即增强图像的亮度及对比度）
        self.img = cv2.equalizeHist(self.img)
        cv2.imwrite(self.output_path + 'equalizeHist.png', self.img)
        self.ui.lb_cap.setPixmap(QtGui.QPixmap(self.output_path + "equalizeHist.png"))

    def medianBlur(self):
        self.bak = self.img.copy()
        #中值滤波去除噪点,将每一像素点的灰度值设置为该点某邻域窗口内的所有像素点灰度值的中值
        #ksize：线性直径大小，大于1的奇数，例如：3, 5, 7
        self.img = cv2.medianBlur(self.img,3)
        cv2.imwrite(self.output_path + 'medianBlur.png', self.img)
        self.ui.lb_cap.setPixmap(QtGui.QPixmap(self.output_path + 'medianBlur.png'))

    def undo(self):
        self.tresult = self.vresult = ''
        self.img = self.bak.copy()
        cv2.imwrite(self.output_path + 'backup.png', self.img)
        self.ui.lb_cap.setPixmap(QtGui.QPixmap(self.output_path + 'backup.png'))

    def denoise(self):
        self.bak = self.img.copy()
        if str(self.ui.cb_dm.currentText()) == '去小点':
            #cv2.findContours会修改原图，cv2.RETR_LIST检测的轮廓不建立等级关系
            contours, hierarchy = cv2.findContours(self.img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                if cv2.contourArea(contours[i]) < int(self.ui.le_dt.text()):
                    #0表示画黑色，-1表示填充，若为正数表示轮廓粗细
                    cv2.drawContours(self.bak,contours,i,0,-1)
            self.img = self.bak.copy()
            cv2.imwrite(self.output_path + 'denoise.png', self.img)
            self.ui.lb_cap.setPixmap(QtGui.QPixmap(self.output_path + "denoise.png"))
            return
        
        height,width,channel = self.img.shape
        dict = {}
        for i in range(width):
            for j in range(height):
                if self.img[j,i,0]!=255 or self.img[j,i,1]!=255 or self.img[j,i,2]!=255:
                    key = '%03d' % self.img[j,i,0] + '%03d' % self.img[j,i,1] + '%03d' % self.img[j,i,2]
                    if dict.has_key(key):
                        dict[key]+=1
                    else:
                        dict[key] = 1
        #按value从小到大排序
        dict= sorted(dict.items(), key=lambda d:d[1])
        #去少像素1是找出最少的像素，将与它相近的颜色去掉
        #去少像素2是找出最多的像素，将与它差异较大的颜色去掉
        #去多像素是找出最多的像素，将与它差异较小的颜色去掉
        #放大会影响科大bbs验证码的去噪
        if str(self.ui.cb_dm.currentText()) == '去少像素1':
            key=dict[0][0]
        else:
            key=dict[len(dict)-1][0]
        b = int(key[:3])
        g = int(key[3:6])
        r = int(key[-3:])
        for i in range(width):
            for j in range(height):
                if self.img[j,i,0] != 255 or self.img[j,i,1] != 255 or self.img[j,i,2] != 255:
                    if str(self.ui.cb_dm.currentText()) == '去少像素2':
                        if math.sqrt(pow(self.img[j,i,0] - b,2) + pow(self.img[j,i,1] - g,2) + pow(self.img[j,i,2] - r,2)) > int(self.ui.le_dt.text()):
                            self.img[j,i,0] = 255
                            self.img[j,i,1] = 255
                            self.img[j,i,2] = 255
                    elif math.sqrt(pow(self.img[j,i,0] - b,2) + pow(self.img[j,i,1] - g,2) + pow(self.img[j,i,2] - r,2)) < int(self.ui.le_dt.text()):
                        self.img[j,i,0] = 255
                        self.img[j,i,1] = 255
                        self.img[j,i,2] = 255

        cv2.imwrite(self.output_path + 'denoise.png', self.img)
        self.ui.lb_cap.setPixmap(QtGui.QPixmap(self.output_path + "denoise.png"))

    def projection(self):
        height,width = self.img.shape
        s = 0#当前白点数,左右边界间的白点数要大于分割阈值
        left = -1#-1表示在找左边界,非负表示左边界值
        #找左右边界
        for i in range(width):
            t = 0#该列白点数
            for j in range(height):
                if(self.img[j,i] == 255):
                    t+=1
                    s+=1
            if left < 0:
                if(t > 0):
                    left = i
            elif(t == 0 and s > int(self.ui.le_divide_thresh.text()) or i == width - 1):#右边界有字符
                #先找上边界再找下边界，防i,j被分成两部分
                top = -1
                for j in range(height):
                    t = 0#该行白点数
                    for k in range(left,i):
                        if(self.img[j,k] == 255):
                            t+=1
                    if(t > 0):
                        top = j
                        break   
                for j in range(height - 1,top,-1):
                    t = 0#该行白点数
                    for k in range(left,i):
                        if(self.img[j,k] == 255):
                            t+=1
                    if(t > 0):
                        img = self.img[top:j + 1,left:i]
                        self.img = self.img[0:height - 1,i:width]
                        return img

    def connected(self):
        #注意cv2.findContours会修改原图
        self.bak = self.img.copy()
        #cv2.RETR_EXTERNAL表示只检测外轮廓
        contours, hierarchy = cv2.findContours(self.img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            x,y,w,h = cv2.boundingRect(contours[i])
            img=self.bak[y-1:y+h+1, x:x+w]
            self.img = self.bak.copy()
            self.img[y-1:y+h+1,x:x+w]=0
            return img
            #rect = cv2.minAreaRect(contours[i])
            #box = cv2.cv.BoxPoints(rect)
            #box = np.int0(box)
            #cv2.drawContours(self.bak,[box],0,(255,255,255),1)
            #cv2.drawContours(self.bak,contours,i,0,-1)
            #hull = cv2.convexHull(contours[i])
            #if hierarchy[0][i][3]>-1:
            #    cv2.drawContours(t,contours,i,0,-1)
            #else:
            #    cv2.drawContours(t,contours,i,255,-1)
            #cv2.rectangle(self.bak,(x,y),(x+w,y+h),(255,255,255),1)

    def divide(self):
        if self.img is None:
            if self.mode == 0:
                #高级模式
                QtGui.QMessageBox.information(self,u"错误",u"已分割完毕！")
            return False

        s=str(self.ui.cb_divide_method.currentText())
        if s=='固定长度':
            height,width = self.img.shape
            thresh=int(self.ui.le_divide_thresh.text())
            t = self.img[0:height,0:thresh]
            if width > thresh:
                self.img = self.img[0:height,thresh:width]
            else:
                self.img = None
        elif s =='投影':
            t = self.projection()
        elif s =='连通域':
            t=self.connected()

        if t is None:
            if self.mode == 0:
                #高级模式
                QtGui.QMessageBox.information(self,u"错误",u"已分割完毕！")
            return False

        t = resize(t)
        cv2.imwrite(self.output_path + 'current.png', t)
        if str(self.ui.cb_type.currentText())=='数字':
            self.tresult+=pytesseract.image_to_string(PIL.Image.open(self.output_path + 'current.png'),lang="eng",config="-psm 7 captcha_digits")
            if platform.system()=="Linux":
                prob=predict(self.output_path + 'current.png')
                lab = np.argsort(-prob)
                for i in range(62):
    	            for k,v in label_set.items():
                        if v==lab[i]:
                            lab[i]=int(k)
                            break
                    if lab[i]<10:
                        c=chr(ord('0')+lab[i])
                        self.vresult+=str(c)
            		print '标签：'+str(c)
            		print '概率：'+str(prob[lab[i]])
			break
        else:
            self.tresult+=pytesseract.image_to_string(PIL.Image.open(self.output_path + 'current.png'),lang="eng",config="-psm 7 captcha")
            if platform.system()=="Linux":
                prob=predict(self.output_path + 'current.png')
                lab = np.argsort(-prob)
                for k,v in label_set.items():
                    if v==lab[0]:
                        lab[0]=int(k)
                        break
                c=chr(ord('0')+lab[0])
                if lab[0]>9 and lab[0]<36:
                    c=chr(ord('A')+lab[0]-10)
                elif lab[0]>35:
                    c=chr(ord('a')+lab[0]-36)
                self.vresult+=str(c)
                print '标签：'+str(c)
                print '概率：'+str(prob[lab[0]])
        self.ui.lb_tess.setText(u'tesseract识别结果：' + self.tresult)
        self.ui.lb_vgg.setText(u'vgg识别结果：' + self.vresult)
        if str(self.ui.cb_engine.currentText())=='tesseract':
            self.ui.le_result.setText(self.tresult)
        else:
            self.ui.le_result.setText(self.vresult)
        self.ui.lb_current.setPixmap(QtGui.QPixmap(self.output_path + "current.png"))
        return True

    def selectPath(self):
        t = QtGui.QFileDialog.getExistingDirectory(self)
        if t != '':
            self.ui.le_lc.setText(t)
            self.files = os.listdir(t)
            self.current = 0

    def login(self):
        if platform.system()=="Windows":
            driver=webdriver.Firefox(executable_path = os.getcwd()+'/geckodriver.exe')
        else:
            driver=webdriver.Firefox(executable_path = os.getcwd()+'/geckodriver')
        while 1:
            try:
                driver.get('http://yjs.ustc.edu.cn/')
                #time.sleep(1)
                #driver.switch_to_window(driver.window_handles[0])
                elem_user = driver.find_element_by_name('userid')
                elem_psw = driver.find_element_by_name('userpwd')
                elem_code = driver.find_element_by_name('txt_check')
                click_login = driver.find_element_by_xpath("//input[@src='_images/pic_bot001.jpg']")
                elem_captcha = driver.find_element_by_xpath("//img[@src='checkcode.asp']")
            except:
                QtGui.QMessageBox.information(self,u"错误",u"未找到相应控件！")
                return
            left = elem_captcha.location['x']
            top = elem_captcha.location['y']
            right = elem_captcha.location['x'] + elem_captcha.size['width']
            bottom = elem_captcha.location['y'] + elem_captcha.size['height']
            driver.get_screenshot_as_file(self.output_path+'web.png')
            #截取验证码
            im =PIL.Image.open(self.output_path+'web.png').crop((int(left), int(top), int(right), int(bottom)))
            im.save(self.output_path+'captcha.png')
            self.img=cv2.imread(self.output_path + "captcha.png")
            self.ui.rb_kdyjs.setChecked(True)
            self.rec()
            if str(self.ui.cb_engine.currentText())=='tesseract':
                s=self.tresult
            else:
                s=self.vresult
            #s = str(pytesseract.image_to_string(PIL.Image.open(self.output_path+'captcha.png'),lang="eng",config="-psm 7 captcha_digits"))
            #s=s.replace(' ','')
            print s
            elem_user.send_keys(str(self.ui.le_name.text()))
            elem_psw.send_keys(str(self.ui.le_pass.text()))
            elem_code.send_keys(s)
            click_login.click()
            try:
                #每0.1秒判断一次条件是否成立，10秒还不成立抛出异常
                webdriver.support.wait.WebDriverWait(driver, 10, 0.1).until(EC.staleness_of(click_login))
            except:  
                QtGui.QMessageBox.information(self,u"错误",u"超时！")
                return
            try:
                if driver.switch_to_alert().text!=u'请输入正确的验证码':
                    QtGui.QMessageBox.information(self,u"错误",driver.switch_to_alert().text)
                    return
                driver.switch_to_alert().accept()
            except:
                return
        #cj = cookielib.CookieJar()   
        #opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))    
        #opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1; rv:20.0) Gecko/20100101 Firefox/20.0')]    
        #urllib2.install_opener(opener)
        ## 用openr访问验证码地址,获取cookie
        #picture = opener.open('http://yjs.ustc.edu.cn/checkcode.asp').read()
        #local = open(self.output_path+'captcha.bmp', 'wb')
        #local.write(picture)
        #local.close()
        #s = str(pytesseract.image_to_string(PIL.Image.open(self.output_path+'captcha.bmp'),lang="eng",config="-psm 7 captcha_digits"))
        #print s
        ##post的表单数据
        #post_data = urllib.urlencode({'userid': str(self.ui.le_name.text()),'userpwd': str(self.ui.le_pass.text()),'txt_check':s})
        #req = urllib2.Request('http://yjs.ustc.edu.cn/', post_data)
        #print urllib2.urlopen(req).read()

    def engine(self,s):
        if platform.system()=="Windows" and str(s)=='vgg':
            QtGui.QMessageBox.information(self,u"错误",u'vgg引擎暂不支持Windows系统')
            self.ui.cb_engine.setCurrentIndex(0)

app = QtGui.QApplication(sys.argv)
window = Ui()
window.show()
sys.exit(app.exec_())