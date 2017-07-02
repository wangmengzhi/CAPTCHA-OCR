# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2, random, string,os
from io import BytesIO
from captchaimage import ImageCaptcha

class OCRBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def gen_rand():
    buf = ''.join(random.sample(string.digits, random.randint(4,8)))
    return buf

def get_label(buf):
    a=[]
    for x in buf:
        if x.isdigit():
            a.append(int(x))
        elif x.isupper():
            a.append(ord(x)-ord('A')+10)
        else:
            a.append(ord(x)-ord('a')+36)
    for i in range(8-len(buf)):
        a.append(10)
    return np.array(a)

def gen_sample(captcha, width, height, trainlabel):
    if trainlabel==0:
        num = gen_rand()
        img = captcha.generate(num)
        img = np.fromstring(img.getvalue(), dtype='uint8')
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    else:
        num=trainlabel
        img=cv2.imread('temp/current.png')
    img = cv2.resize(img, (width, height))
    img = np.multiply(img, 1/255.0)
    img = img.transpose(2, 0, 1)
    return (num, img)

class OCRIter(mx.io.DataIter):
    def __init__(self, count, batch_size, num_label, height, width, trainlabel=0):
        super(OCRIter, self).__init__()
        self.captcha = ImageCaptcha()
        self.trainlabel=trainlabel
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]
        
    def __iter__(self):
        for k in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = gen_sample(self.captcha, self.width, self.height, self.trainlabel)
                data.append(img)
                label.append(get_label(num))

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']
            
            data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_e2enet_variable(predict=0):
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    conv1 = mx.symbol.Convolution(name="convolution0",data=data, kernel=(5,5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(name="convolution1",data=relu1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(name="convolution2",data=relu2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    conv4 = mx.symbol.Convolution(name="convolution3",data=relu3, kernel=(3,3), num_filter=32)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=pool4, act_type="relu")

    flatten = mx.symbol.Flatten(data = relu4)
    fc1 = mx.symbol.FullyConnected(name="fullyconnected0",data = flatten, num_hidden = 1024)
    fc21 = mx.symbol.FullyConnected(name="fullyconnected1",data = fc1, num_hidden = 11)
    fc22 = mx.symbol.FullyConnected(name="fullyconnected2",data = fc1, num_hidden = 11)
    fc23 = mx.symbol.FullyConnected(name="fullyconnected3",data = fc1, num_hidden = 11)
    fc24 = mx.symbol.FullyConnected(name="fullyconnected4",data = fc1, num_hidden = 11)
    fc25 = mx.symbol.FullyConnected(name="fullyconnected5",data = fc1, num_hidden = 11)
    fc26 = mx.symbol.FullyConnected(name="fullyconnected6",data = fc1, num_hidden = 11)
    fc27 = mx.symbol.FullyConnected(name="fullyconnected7",data = fc1, num_hidden = 11)
    fc28 = mx.symbol.FullyConnected(name="fullyconnected8",data = fc1, num_hidden = 11)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24,fc25,fc26,fc27,fc28], dim = 0)
    if predict:
        return mx.symbol.SoftmaxOutput(data = fc2, name = "softmax")
    label = mx.symbol.transpose(data = label)
    label = mx.symbol.Reshape(data = label, target_shape = (0, ))
    return mx.symbol.SoftmaxOutput(data = fc2, label = label, name = "softmax")


def Accuracy(label, pred):
    label = label.T.reshape((-1, ))
    hit = 0
    total = 0
    for i in range(pred.shape[0] / 8):
        l1=[]
        l2=[]
        for j in range(8):
            k = i * 8 + j
            if np.argmax(pred[k])<10:
                l1.append(np.argmax(pred[k]))
            if int(label[k])<10:
                l2.append(int(label[k]))
        if l1==l2 and len(l1)>0:
            hit += 1
        total += 1
    return 1.0 * hit / total

def train(label):
    #_, arg , aux = mx.model.load_checkpoint("e2e-ocr-variable", 1)
    model = mx.model.FeedForward(ctx = mx.gpu(),
                                 symbol = get_e2enet_variable(),
                                 num_epoch = 1,
                                 learning_rate = 0.001,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.9)
    batch_size = 32 if label==0 else 1
    data_train = OCRIter(100000 if label==0 else 1, batch_size, 8, 30, 80, label)
    data_test = OCRIter(1000 if label==0 else 0, batch_size, 8, 30, 80)
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    model.fit(X = data_train, eval_data = data_test, eval_metric = Accuracy, batch_end_callback=mx.callback.Speedometer(batch_size, 50),)
    model.save("e2e-ocr-variable")

if __name__ == '__main__':
    train(0)
