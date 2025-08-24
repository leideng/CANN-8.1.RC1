# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
log_sigmoid_grad
"""
import te.lang.cce as tbe
from tbe import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import para_check
from te.utils import shape_util


# 'pylint: disable=unused-argument,too-many-locals
@fusion_manager.register("log_sigmoid_grad")
def log_sigmoid_grad_compute(grads, features, backprops, kernel_name='log_sigmoid_grad'):
    """
    log-sigmoid backward compute
    :param grads: the shape and dtype of tensor from previous layer
    :param features: the shape and dtype of input
    :param backprops: the shape and dtype of output
    :param kernel_name: cce kernel name, default value is 'log_sigmoid_grad'
    :return: gradient of log_sigmoid
    """
    # input tensor broadcast
    shape_1 = shape_util.shape_to_list(grads.shape)
    shape_2 = shape_util.shape_to_list(features.shape)
    shape_1, shape_2, shape_max = shape_util.produce_shapes(shape_1, shape_2)
    dtype_grads = grads.dtype
    dtype_features = features.dtype
    grads = tbe.broadcast(grads, shape_max)
    features = tbe.broadcast(features, shape_max)
    if dtype_grads != "float32":
        grads = tbe.cast_to(grads, "float32")
        features = tbe.cast_to(features, "float32")
    zeros = tbe.broadcast(tvm.const(0, "float32"), shape_max)

    # log_sigmoid_backward compute
    # when feature is negative
    tempt1 = tbe.vexp(features)                               # e^x
    tempt2 = tbe.vadds(tempt1, tvm.const(1, "float32"))       # 1+e^x
    tempt3 = tbe.vrec(tempt2)                                 # 1/(1+e^x)
    # when x is positive
    tempt4 = tbe.vmuls(features, tvm.const(-1, "float32"))    # -x
    tempt5 = tbe.vexp(tempt4)                                 # e^(-x)
    tempt6 = tbe.vadds(tempt5, tvm.const(1, "float32"))       # 1+e^(-x)
    tempt7 = tbe.vdiv(tempt5, tempt6)                         # e^(-x)/(1+e^(-x))

    tempt8 = tbe.vcmpsel(features, zeros, "le", tempt3, tempt7)
    res = tbe.vmul(tempt8, grads)
    if dtype_features != "float32":
        res = tbe.cast_to(res, dtype_features)

    return res


# 'pylint: disable=unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def log_sigmoid_grad(grads, features, backprops, kernel_name='log_sigmoid_grad'):
    """
    log-sigmoid backward compute
    :param grads: the shape and dtype of tensor from previous layer
    :param features: the shape and dtype of input
    :param backprops: the shape and dtype of output
    :param kernel_name: cce kernel name, default value is 'log_sigmoid_grad'
    :return: gradient of log_sigmoid
    """
    # obtain operator information
    grads_shape = grads.get("shape")
    features_shape = features.get("shape")
    grads_dtype = grads.get("dtype").lower()
    features_dtype = features.get("dtype").lower()
    check_list = ("float16", "float32")

    # operator check
    para_check.check_shape_rule(grads_shape)
    para_check.check_shape_rule(features_shape)
    para_check.check_shape_size(grads_shape)
    para_check.check_shape_size(features_shape)
    para_check.check_dtype_rule(grads_dtype, check_list)
    para_check.check_dtype_rule(features_dtype, check_list)
    para_check.check_kernel_name(kernel_name)

    # tensor placeholder
    data_grads = tvm.placeholder(grads_shape, name='data_grads', dtype=grads_dtype)
    data_features = tvm.placeholder(features_shape, name='data_features', dtype=features_dtype)

    # log sigmoid backward compute function
    res = log_sigmoid_grad_compute(data_grads, data_features, backprops, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # compile configuration
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": (data_grads, data_features, res)}
    tbe.build(schedule, config)
