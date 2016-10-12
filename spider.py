import urllib.request
url = "http://bbs.ustc.edu.cn/cgi/bbscaptcha"  
path = "bbscaptcha.png"  
urllib.request.urlretrieve(url, path)
