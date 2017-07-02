# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2, random
from io import BytesIO
from captchaimage import ImageCaptcha
from e2e_ocr_variable import gen_rand, get_label, gen_sample, get_e2enet_variable

if __name__ == '__main__':
    captcha = ImageCaptcha()
    num, img = gen_sample(captcha, 80, 30, 0)
    print 'gen captcha:', num
    
    batch_size = 1
    _, arg_params, __ = mx.model.load_checkpoint("e2e-ocr-variable", 1)
    data_shape = [("data", (batch_size, 3, 30, 80))]
    input_shapes = dict(data_shape)
    sym = get_e2enet_variable(1)
    executor = sym.simple_bind(ctx = mx.cpu(), **input_shapes)
    for key in executor.arg_dict.keys():
        if key in arg_params:
            arg_params[key].copyto(executor.arg_dict[key])

    executor.forward(is_train = True, data = mx.nd.array([img]))
    probs = executor.outputs[0].asnumpy()
    line = ''
    for i in range(8):
        n=np.argmax(probs[i])
        if n<10:
            line+=str(n)
        #elif n<36:
        #    line+=chr(ord('A')+n-10)
        #elif n<62:
        #    line+=chr(ord('a')+n-36)
    print 'predicted: ' + line
