import onnx_mxnet

import numpy as np
import mxnet as mx

from collections import namedtuple

sym, params = onnx_mxnet.import_model('/home/ubuntu/model.onnx')

print sym
print sym.attr_dict()

test_image = np.random.rand(1, 3, 256, 128)

mod = mx.mod.Module(symbol=sym, data_names=['input_0'], context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('input_0', test_image.shape)], label_shapes=None)
mod.set_params(arg_params=params, aux_params=None, allow_missing=True, allow_extra=True)
