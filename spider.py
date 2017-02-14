import urllib
import os 
#url = "http://bbs.ustc.edu.cn/cgi/bbscaptcha"
#url='http://passport.iqiyi.com/register/vcode.php'
#url='https://sso.le.com/verify'
#url='http://passport.csdn.net/ajax/verifyhandler.ashx'
url='http://yjs.ustc.edu.cn/checkcode.asp'
output_path = 'captcha/yjs/'
if not(os.path.exists(output_path)):
    os.makedirs(output_path)
urllib.URLopener.version = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:51.0) Gecko/20100101 Firefox/51.0'
for i in range(100):
    print i
    urllib.urlretrieve(url, output_path+str(i)+'.bmp')
print 'ok'
