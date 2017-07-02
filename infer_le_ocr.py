# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import mxnet as mx
import numpy as np
import cv2
from le_ocr import get_lenet

if __name__ == '__main__':
    img=cv2.imread('data/train/10/img011-00002.png',0)
    batch_size = 1
    _, arg_params, __ = mx.model.load_checkpoint("le-ocr", 1)
    data_shape = [("data", (batch_size, 1, 28, 28))]
    input_shapes = dict(data_shape)
    sym = get_lenet()
    executor = sym.simple_bind(ctx = mx.cpu(), **input_shapes)
    for key in executor.arg_dict.keys():
        if key in arg_params:
            arg_params[key].copyto(executor.arg_dict[key])

    executor.forward(data = mx.nd.array([[img]]))
    probs = executor.outputs[0].asnumpy()
    line = ''
    for i in range(probs.shape[0]):
        n=np.argmax(probs[i])
        if n<10:
            line+=str(n)
        elif n<36:
            line+=chr(ord('A')+n-10)
        else:
            line+=chr(ord('a')+n-36)
    print 'predicted: ' + line
