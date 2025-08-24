#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
common function for check ops parameter
"""
import warnings

from te.utils import para_check

STACKLEVEL_FOR_SHAPE_UTIL = 2


def squeeze_shape(shape):
    """
    squeeze shape
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import squeeze_shape
    return squeeze_shape(shape)


def wrap_axes_to_positive(axes, rank):
    """
    wrap axis to positive
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import wrap_axes_to_positive
    return wrap_axes_to_positive(axes, rank)


def refine_shape_axes(shape, axes):
    """
    refine shape and axes for reduce ops, fused reduced axes,
    and fused not reduced axes
    result is a tuple of (shape, axes)
    for example:
        input: shape is (2,3,4,5,6), axes is (1, -3)
        output: (2, 12, 30), (1,)

    Parameters
    ----------
    shape : shape which need refine

    axes : axes which need refine

    Returns
    -------
    shape : list
        refined shape

    axes : list
        refined axes

    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import refine_shape_axes
    return refine_shape_axes(shape, axes)


def broadcast_shapes(shape1, shape2, op_name=para_check.OP_NAME,
                     param_name_input1='', param_name_input2=''):
    """
    two input shapes produce three output shape
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import broadcast_shapes
    return broadcast_shapes(shape1, shape2, op_name, param_name_input1, param_name_input2)


def refine_shapes_for_broadcast(shape1, shape2):
    """
    Fusing the axes for the input shapes
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import refine_shapes_for_broadcast
    return refine_shapes_for_broadcast(shape1, shape2)


def simplify_axis_shape(shape, axis):
    """
    simplify the shape and aixs
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import simplify_axis_shape
    return simplify_axis_shape(shape, axis)


def shape_refine(shape, reduce_axis=None, keep_dims=True):
    """
    refine shape to drop 1 in shape according to reduce axis,
    if input is just shape, result is shape, and if inputs are shape and axis,
    result is a tuple of (shape, axis)

    Parameters
    ----------
    shape : shape of data

    reduce_axis : list, tuple or int
        axis want to reduce

    keepdims: if keepdims = True, we should not refine the shape

    Returns
    -------
    shape : list
        refined shape

    reduce_axis : list
        if input parameters send reduce axis, this will be the output.
        if all the reduce axis is illegal like the length of reduce axis is 1,
        a empty list([]) will be returned.

    """

    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import shape_refine
    return shape_refine(shape, reduce_axis, keep_dims)


def refine_axis(axis, shape):
    """
    refine axis

    Parameters
    ----------
    axis :
        axis want to reduce

    shape : shape of data

    Returns
    -------
    res_reduce_axis : list
        refined axis
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import refine_axis
    return refine_axis(axis, shape)


def _axis_value_type_check(shape_len, value):
    """
    Check the value of the axis
    """
    if type(value) != int:
        raise RuntimeError("type of axis value should be int")
    if value >= shape_len or value < -shape_len:
        raise RuntimeError(
            "input axis is out of range, axis value can be from %d to %d"
            % (-shape_len, shape_len - 1))
    if value < 0:
        value = shape_len + value
    return value


def axis_check(shape_len, axis):
    """
    Check the value of axis and return the sorted axis
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import axis_check
    return axis_check(shape_len, axis)


def produce_shapes(shape1, shape2):
    """
    two input shapes produce three output shape
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import produce_shapes
    return produce_shapes(shape1, shape2)


def check_reduce_need_refine(shape, reduce_axis, keep_dims):
    """
    # if the reduce axis correspond to shape[axis] is 1,
    we can not refine the shape,or the reduce axis will be wrong
    shape : shape of data

    reduce_axis : list, tuple or int  axis want to reduce

    :return: True or False
    """

    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import check_reduce_need_refine
    return check_reduce_need_refine(shape, reduce_axis, keep_dims)


def scalar2tensor_one(shape):
    """
    if the input_shape is [],convert the input_shape to [1]
    ----------
    shape: shape of input tensor

    Returns
    -------
    list:[1]
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import scalar2tensor_one
    return scalar2tensor_one(shape)


def axis_transform_5d(axis, data_format):
    """
    4d format axis to 5d mapping
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import axis_transform_5d
    return axis_transform_5d(axis, data_format)


def compare_tensor_dict_key(dict1, dict2, dict_key):
    """
    compare the key value between dict1 and dict2,
    the value is not equal, will raise error

    Parameters
    ----------
    dict1: dict
        input dict1
    dict2: dict
        input dict2
    dict_key: str
        the key that will be compare

    Returns
    -------
    None
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import compare_tensor_dict_key
    return compare_tensor_dict_key(dict1, dict2, dict_key)


def get_shape_size(shape):
    """
    get all dimension.
    ----------
    shape: shape of data

    Returns
    -------
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import get_shape_size
    return get_shape_size(shape)


def cast(x, dtype):
    """Cast input to specified data type.

    Parameters
    ----------
    x : tvm.Tensor or Expr
        Input argument.

    dtype : str
        Data type.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import cast
    return cast(x, dtype)


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    warnings.warn("te.utils.shape_util is expired, please replace it with tbe.common.utils.shape_util",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_SHAPE_UTIL)
    from tbe.common.utils import shape_to_list
    return shape_to_list(shape)
