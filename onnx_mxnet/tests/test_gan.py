# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import absolute_import as _abs
import mxnet as mx
import numpy as np
import os
import onnx_mxnet
from collections import namedtuple


print "Converting onnx format to mxnet's symbol and params..."
sym, params = onnx_mxnet.import_model('/home/ubuntu/outline2yellow_gan.onnx')
print sym

mod = mx.mod.Module(symbol=sym, data_names=['input_0'], context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('input_0', (1,1,128,128))])
