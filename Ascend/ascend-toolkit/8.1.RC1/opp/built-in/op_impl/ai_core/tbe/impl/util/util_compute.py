#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
common function
"""
import collections
import tbe.common.utils as tbe_utils
from tbe import tvm
from te.lang.cce.te_compute.elewise_compute import vmuls
from te.lang.cce.te_compute.elewise_compute import vabs
from te.lang.cce.te_compute.elewise_compute import vadds
from te.lang.cce.te_compute.elewise_compute import vdiv
from impl.util.platform_adapter import tbe_context

BATCH_MATMUL_LENGTH = 5


def sign(input_data):
    """
    Algrithm:
        sign(x) = 2**(15)/(2**(-15) + 2**(15) *|x|)
    ----------
    Parameters
        input_data: the placeholder of data input
    ----------
    Returns
        A tensor of sign(x)
    -------
    """
    dtype = input_data.dtype

    if dtype == "float16":
        fp_max = tvm.const(2**15, dtype)
        fp_min = tvm.const(2**(-15), dtype)
    elif dtype == "float32":
        fp_max = tvm.const(2**62, dtype)
        fp_min = tvm.const(2**(-62), dtype)
    else:
        raise RuntimeError(
            "The type must be float16 or float32.")
    new_data = vmuls(input_data, fp_max)
    abs_data = vabs(new_data)
    denominator = vadds(abs_data, fp_min)
    res = vdiv(new_data, denominator)

    return res


def get_cof(axes, shape):
    """
    get cof
    ----------
    Parameters
        axes: all reduce axes
        shape: input shape
    ----------
    Returns
        the product of reduce axes
    """
    reduce_elts = 1.0
    if isinstance(axes, collections.abc.Iterable):
        for i in axes:
            reduce_elts *= shape[i]
    else:
        reduce_elts = shape[axes]
    return reduce_elts


def check_fc_fuse(input_tensor):
    """
    check if fused with fullyconnection

    Parameters:
    input_tensor: the tensor of elem input

    Returns result
    """
    queue = [input_tensor]
    visited = [input_tensor]
    fc_flag = False
    while queue:
        item = queue.pop(0)

        fc_flag = ("fc_flag" in item.op.attrs and item.op.attrs["fc_flag"].value == 1)
        if fc_flag:
            break

        for child in item.op.input_tensors:
            if child not in visited:
                queue.append(child)
                visited.append(child)

    # the min dim size of fc is 2
    fc_fuse_min_dim_flag = (len(input_tensor.shape) == 2 and
                            ("format" in input_tensor.op.attrs and input_tensor.op.attrs["format"] == "ND"))
    # the max dim size of fc is 4
    fc_fuse_max_dim_flag = (len(input_tensor.shape) == 4 and
                            ("format" in input_tensor.op.attrs and input_tensor.op.attrs["format"] == "FRACTAL_NZ"))
    if fc_flag and (fc_fuse_min_dim_flag or fc_fuse_max_dim_flag):
        return True

    return False


def only_static_support():
    """check fusion support, only static shape can support ub fusion"""
    if tbe_context.get_context().get_op_mode() == "static":
        return True
    return False


def _is_matmul_tensor(tensor):
    if tensor is None:
        return False
    # Matmul tag contains matmul, matmul_gemv, matmul_gevm
    if "matmul" in tensor.op.tag:
        return True
    # tag is gemm on milan
    if "gemm" in tensor.op.tag:
        return True
    # fixpipe fusion
    if "fixpipe_reform" in tensor.op.tag:
        # tensor_mmad -> fixpipe -> fixpipe_reform
        return _is_matmul_tensor(tensor.op.input_tensors[0].op.input_tensors[0])
    return False


def check_batchmatmul_fuse(input_tensor):
    """
    check if fused with batchmatmul

    Parameters:
    input_tensor: the tensor of elem input

    Returns result
    """
    queue = [input_tensor]
    visited = [input_tensor]
    while queue:
        item = queue.pop(0)
        if len(item.shape) == BATCH_MATMUL_LENGTH and _is_matmul_tensor(item) \
                and "format" in item.op.attrs and item.op.attrs["format"] == "FRACTAL_NZ":
            return True

        for child in item.op.input_tensors:
            if child not in visited:
                queue.append(child)
                visited.append(child)
    return False


def fetch_batchmatmul_fuse_tensor(input_tensor):
    """
    fetch fused tensor of batchmatmul

    Parameters:
    input_tensor: the tensor of elem input

    Returns result
    """
    queue = [input_tensor, ]
    visited = [input_tensor, ]
    while queue:
        item = queue.pop(0)
        if len(item.shape) == BATCH_MATMUL_LENGTH and _is_matmul_tensor(item) \
                and item.op.attrs["format"] == "FRACTAL_NZ":
            return item

        for child in item.op.input_tensors:
            if child not in visited:
                queue.append(child)
                visited.append(child)
    return None


def batchmatmul_elem_nd2nz(batch_matmul, elem_input, para_dict, para_name):
    """
    reshape batchmatmul+elem ubfusion inputs tensors

    Parameters:
    batch_matmul: the tensor of batchmatmul result

    elem_input: the tensor of elem

    para_dict: the dict with batch_shape and format_elem

    para_name: the elemwise name

    Returns result
    """
    batch_shape = para_dict.get("batch_shape", [])
    format_elem = para_dict.get("format_elem", "ND")
    shape_matmul = tbe_utils.shape_util.shape_to_list(batch_matmul.shape)
    shape_elem = tbe_utils.shape_util.shape_to_list(elem_input.shape)
    shape_max = batch_shape + shape_matmul[-4:]

    if format_elem != "FRACTAL_NZ" and shape_elem[-1] != 1:
        elem_ndim = shape_elem[-1]
        shape_elem_batch = [1] * len(batch_shape) if len(shape_elem) == 1 else shape_elem[0:-2]
        shape_elem_nz = shape_elem_batch + [elem_ndim // 16, 1, 1, 16]
        elem_input = tvm.compute(
            shape_elem_nz,
            lambda *indice: (elem_input(indice[-4]*16 + indice[-1]) if len(shape_elem) == 1
                             else elem_input(*indice[0:-4], 0, indice[-4]*16 + indice[-1])),
            name="broadcast_nz2nd_" + para_name,
            tag="broadcast_nz2nd")
    return elem_input, shape_max


def batchmatmul_elem_reshape(batch_matmul, elem_input, batch_shape, para_name):
    """
    reshape batchmatmul+elem ubfusion inputs tensors

    Parameters:
    batch_matmul: the tensor of batchmatmul result

    elem_input: the tensor of elem

    batch_shape: the shape of  batch

    para_name: the elemwise name

    Returns result
    """
    shape_matmul = tbe_utils.shape_util.shape_to_list(batch_matmul.shape)

    def _batch_reshape(indices, input_tensor):
        if len(batch_shape) == 1:
            return input_tensor(indices[0], *indices[-4:])
        elif len(batch_shape) == 2:
            return input_tensor(indices[0] // batch_shape[-1],
                                indices[0] % batch_shape[-1],
                                *indices[-4:])
        elif len(batch_shape) == 3:
            return input_tensor(indices[0] // batch_shape[-1] // batch_shape[-2],
                                indices[0] // batch_shape[-1] % batch_shape[-2],
                                indices[0] % batch_shape[-1],
                                *indices[-4:])
        return input_tensor(indices[0] // batch_shape[-1] // batch_shape[-2] // batch_shape[-3],
                            indices[0] // batch_shape[-1] // batch_shape[-2] % batch_shape[-3],
                            indices[0] // batch_shape[-1] % batch_shape[-2],
                            indices[0] % batch_shape[-1],
                            *indices[-4:])

    elem_input = tvm.compute(shape_matmul,
                             lambda *indices: _batch_reshape(indices, elem_input),
                             name="broadcast_reshape_" + para_name,
                             tag="broadcast_reshape")

    return elem_input
