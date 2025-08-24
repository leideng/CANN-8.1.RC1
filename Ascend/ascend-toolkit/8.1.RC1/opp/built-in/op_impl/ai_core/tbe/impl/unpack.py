#!/usr/bin/python
# -*- coding: utf-8 -*-
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
unpack
"""
import functools

from impl import copy_only
from impl import split_d
from impl.constant_util import BLOCK_SIZE
from impl.dynamic.unpack import op_select_format as bn_op_select_format
from impl.util import util_common
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
def get_op_support_info(x, y, num, axis, kernel_name="unpack"):
    """
    get unpack slice info
    """
    format_x = x.get("format")
    shape_x = x.get("shape")
    # infer the value of num
    real_axis = axis + len(shape_x) if axis < 0 else axis
    support_ori_format = util_common.get_fused_format_str(["N", "D", "H", "W", "C"]) + \
                         util_common.get_fused_format_str(["N", "H", "W", "C"])
    support_format = ["NC1HWC0", "ND"] + support_ori_format
    if num is None:
        num = shape_x[real_axis]
    if format_x in support_format:
        axis_split_matrix = []
        axis_reduce_list = None
        for idx, _ in enumerate(shape_x):
            output_split = []
            output_axis = idx - 1 if idx > real_axis else idx
            if idx == real_axis:
                continue
            for output_idx in range(num):
                output_split.append([output_idx, [output_axis]])
            split_info = [SplitInput([0, [idx], [-1], [-1]]), SplitOutput(*output_split)]
            axis_split_matrix.append(split_info)
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)

    return op_cal_info_in_json


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments,too-many-locals
def op_select_format(x, y, num, axis, kernel_name="unpack"):
    """
    unpacks the given dimension of a rank R tensor into rank (R-1) tensors. \n
    1.when unpack by C, but output size not C0 align so don't support NC1HWC0 \n
    2.when split_d by N,H,W, support NC1HWC0 \n
    3.when x's format is one of [NCHW,NHWC], the lengths of x's shape == 4,
     dim of axis in x's format != C: support 5HD format \n
        example:
        original:
        axis=1
        x's Tensor(shape=(2, 3, 4, 5), "NHWC")
        support conversion to 5HD fromat:
        x's Tensor(shape=(2, 1, 3, 4, 16), "5HD")
    """
    return bn_op_select_format(x, y, num, axis, kernel_name)


# 'pylint: disable=unused-argument
def check_supported(x, y, num=None, axis=0, kernel_name="unpack"):
    """
    The maximum number of outputs supported by the unpack must be less than 63*63
    """
    input_shape = x.get("ori_shape")
    real_num = _process_num_param(input_shape, num, axis, False, kernel_name)
    max_output_num = 63 * 63
    if real_num > max_output_num:
        reason = "real_num is bigger than max_output_num, real_num=%s, max_output_num=%s"\
                  % (real_num, max_output_num)
        return False, reason
    return True, ""


def _process_num_param(input_shape, num, axis, is_known_shape, kernel_name):
    """
    process num param

    Parameters
    ----------
    input_shape: tuple
        the shape of tensor.
    num: int
        the length of the dim axis.
    axis: int.
        the axis of unapck.
    is_known_shape: bool
        is known shape or not
    kernel_name: str
        cce kernel name.

    Returns
    -------
    real num
    """
    if num is None:
        num = input_shape[axis]
    if num is None:
        error_manager_vector.raise_err_check_params_rules(kernel_name, 'shape[axis] not be None', 'shape[axis]',
                                                          input_shape[axis])
    if num < 1:
        error_manager_vector.raise_err_check_params_rules(kernel_name, 'num can not smaller than 1', 'num', num)

    # [-1, -2] means we don't known the shape currently.
    is_axis_dim_known = True
    if len(input_shape) == 1 and input_shape[0] == -2:
        is_axis_dim_known = False
    if len(input_shape) > axis and input_shape[axis] == -1:
        is_axis_dim_known = False

    if (is_known_shape or is_axis_dim_known) and num != input_shape[axis]:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, 'num', input_shape[axis], num)
    return num


def _get_max_num_in_platform():
    # 1536B means stack holding the param provided to the platform,
    # 1 param takes 8 bytes, needs Multiple output param and 1 input param
    # mini has more parameters (offset, index) than cloud
    compile_plat = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if compile_plat in ("Ascend310", ):
        max_num = (1536 // 3) // 8 - 1
    else:
        max_num = 1536 // 8 - 1
    return max_num


def _check_params(shape, num, axis, dformat, dtype, kernel_name):
    """
    check the parameters including shape, num, axis, dformat, dtype, kernel_name

    Parameters
    ----------
    shape: tuple
        the shape of tensor.
    num: int
        the length of the dim axis.
    axis: int.
        the axis of unapck.
    dformat: str.
        the data format of input.
    dtype: str
        the data type.
    kernel_name: str
        cce kernel name.

    Returns
    -------
    None
    """
    para_check.check_shape(shape, param_name="x")

    # check format
    support_ori_format = util_common.get_fused_format_str(["N", "D", "H", "W", "C"]) + \
                         util_common.get_fused_format_str(["N", "H", "W", "C"])

    format_list = ["NC1HWC0", "ND"] + support_ori_format
    if dformat == "NC1HWC0":
        if len(shape) != 5:
            error_manager_vector.raise_err_input_param_range_invalid(kernel_name, 'x', 5, 5, len(shape))

        # index list of H W axis.
        suporrted_list = (-5, -3, -2, 0, 2, 3)
        if axis not in suporrted_list:
            error_manager_vector.raise_err_check_params_rules(
                kernel_name, "axis must be one of {-5, -3, -2, 0, 2, 3} when the format of 'x' is NC1HWC0", 'axis',
                axis)
    else:
        if dformat not in format_list:
            error_manager_vector.raise_err_input_format_invalid(kernel_name, 'x', format_list, dformat)

    # check axis value
    if axis < -len(shape) or axis >= len(shape):
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, 'axis', 1 - len(shape), len(shape), axis)

    num = _process_num_param(shape, num, axis, True, kernel_name)
    max_num = _get_max_num_in_platform()
    if num > max_num:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, 'num', 1, max_num, num)

    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="x")


def _get_public_param(dtype, coexisting_quantities=1):
    """
    get public parameters

    Parameters
    ----------
    dtype: str
        the data type.
    coexisting_quantities: int
        coexisting_quantities

    Returns
    -------
    total_ele: int
        the size of the  data for UB to move at a time.
    ele_each_block: int
        the numbers of data in one block.
    device_core_num: int
        the numbers of blockdim.
    """
    # reserve one BLOCK_SIZE buffer for reg buff (dma_copy)
    ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // 2 - BLOCK_SIZE
    # Convert bits to Bytes
    dtype_size = tbe_platform.get_bit_len(dtype) // 8
    # gm->ub maximum copy data at a time
    total_ele = ub_size_bytes // dtype_size // coexisting_quantities

    # 32 means one block size(32 Bytes), divide by 32 to get the numbers
    # of data that can be stored in one block.
    one_block_bytes_size = tbe_platform.VECTOR_INST_BLOCK_WIDTH // tbe_platform.VECTOR_INST_BLOCK_NUM
    ele_each_block = one_block_bytes_size // dtype_size

    # get core num according to the product
    device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

    return total_ele, ele_each_block, device_core_num


def _check_use_special_optimize(dtype, afterdim, flag=False):
    """
    Function: use to optimize special scene.
    """
    is_dtype_support = dtype in ("float16", "float32")
    is_shape_support = (afterdim in (8, 16, 32, 64) or afterdim < 8)

    _, ele_each_block, _ = _get_public_param(dtype)
    if afterdim < ele_each_block:
        if dtype in ("int8", "uint8") and not afterdim % 8:
            dtype = "uint64"
            afterdim = afterdim // 8
            flag = True
        if dtype in ("float16", "int16", "uint16") and not afterdim % 4:
            dtype = "uint64"
            afterdim = afterdim // 4
            flag = True
        if dtype in ("float32", "int32", "uint32") and not afterdim % 2:
            dtype = "uint64"
            afterdim = afterdim // 2
            flag = True

    return dtype, afterdim, is_dtype_support and is_shape_support and not flag


def _index_offset(shape, axis, offset, *index):
    """
    Compute the offset of index along one dimension.

    Parameters
    ----------
    shape: list
        the shape of tensor.
    axis: int
        the dimension along which to unpack.
    offset: int
        the offset of axis.
    index: list or tuple
        index value tuple.

    Returns
    -------
    output_index: tuple
        output index with one input index value add axis offset.
    """
    input_index = list(index)
    output_index = ()
    for i, _ in enumerate(shape):
        if i == axis:
            input_index[i] = input_index[i] + offset
        output_index += (input_index[i], )

    return output_index


def _tiling_axis(shape, dtype, coexisting_quantities=1):
    """
    Calculate the tile parameters.

    Parameters
    ----------
    shape: list or tuple
        the shape of tensor.
    dtype: str
        the dtype of tensor.
    coexisting_quantities: int
        coexisting_quantities

    Returns
    -------
    split_axis: int
        the target axis that is used for tile the tensor.
    split_factor: int
        the factor used when tile the target axis.
    """
    total_ele, ele_each_block, _ = _get_public_param(dtype, coexisting_quantities)

    tiling_shape = [dim for dim in shape]
    if shape[-1] % ele_each_block != 0:
        last_ele = ((shape[-1] + ele_each_block - 1) // ele_each_block) * ele_each_block
        tiling_shape[-1] = int(last_ele)

    split_axis = 0
    split_factor = 1
    for index, _ in enumerate(tiling_shape):
        ele_cnt = functools.reduce(lambda x, y: x * y, tiling_shape[index:])
        if ele_cnt <= total_ele:
            split_axis = index - 1
            split_factor = total_ele // ele_cnt
            break
        if index == len(tiling_shape) - 1:
            split_axis = index
            split_factor = total_ele
            break

    if split_axis < 0:
        split_axis = 0
        split_factor = tiling_shape[0]

    if split_axis == 0 and tiling_shape[0] % split_factor != 0:
        while (tiling_shape[0] % split_factor) * shape[-1] < ele_each_block:
            split_factor -= 1
            if split_factor == 1:
                break
    elif split_axis == 2 and shape[-1] % split_factor != 0:
        while shape[-1] % split_factor < ele_each_block:
            split_factor -= 1
            if split_factor == 1:
                break

    return split_axis, split_factor


# 'pylint: disable=unnecessary-lambda,too-many-locals,too-many-return-values
@register_operator_compute("unpack", op_mode="static", support_fusion=True)
def _unpack_compute_scalar(input_place, y, num, axis, kernel_name="unpack"):
    """
    unpack a tensor into `num` tensors along axis dimension.

    Parameters
    ----------
    input_place: TVM tensor
        the tensor of input.
    y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis.
    axis: int.
        the axis to unpack along.
    kernel_name : str.
        cce kernel name, default value is "unpack".

    Returns
    -------
    gm2ub_tensor: TVM tensor
        the tensors of gm2ub, tensor type is TVM tensor.
    ub2ub_tensor_list: list
        the list of ub2ub tensors, tensor type is TVM tensor.
    ub2gm_tensor_list: list
        the list of ub2gm tensors, tensor type is TVM tensor.
    virtual_node:
        the tensors of virtual output node, tensor type is TVM tensor.
    """
    input_shape = shape_util.shape_to_list(input_place.shape)

    gm2ub_tensor = tvm.compute(input_shape, lambda *index: input_place(*index), name="gm2ub_tensor")

    output_shape = input_shape
    for index, _ in enumerate(output_shape):
        output_shape[index] = output_shape[index] if index != axis else 1

    offset = 0
    ub2ub_tensor_list = []
    ub2gm_tensor_list = []
    for i in range(num):
        ub2ub_tensor = tvm.compute(output_shape,
                                   lambda *index: gm2ub_tensor(*_index_offset(output_shape, axis, offset, *index)),
                                   name=''.join(['tensor', str(i)]))
        ub2ub_tensor_list.append(ub2ub_tensor)

        ub2gm_tensor = tvm.compute(output_shape,
                                   lambda *index, tensor_in=ub2ub_tensor: tensor_in(*index),
                                   name=''.join(['res', str(i)]))
        ub2gm_tensor_list.append(ub2gm_tensor)

        offset = offset + output_shape[axis]

    # create a virtual node
    def _add_compute(*index):
        virtual_tensor = ub2gm_tensor_list[0](*index)
        for ub2gm_tensor in ub2gm_tensor_list[1:]:
            virtual_tensor += ub2gm_tensor(*index)
        return virtual_tensor

    virtual_node = tvm.compute(output_shape, lambda *index: _add_compute(*index), name="virtual_node")

    return gm2ub_tensor, ub2ub_tensor_list, ub2gm_tensor_list, virtual_node


# 'pylint: disable=too-many-locals
@register_operator_compute("Unpack", op_mode="static", support_fusion=True)
def _unpack_compute_copy(input_place, y, num, axis, kernel_name="unpack"):
    """
    unpack a tensor into `num` tensors along axis dimension.

    Parameters
    ----------
    input_place: TVM tensor
        the tensor of input.
    y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis.
    axis: int.
        the axis to unpack along.
    kernel_name : str.
        cce kernel name, default value is "unpack".

    Returns
    -------
    gm2ub_tensor_list: list
        the list of gm2ub tensors, tensor type is TVM tensor.
    ub2gm_tensor_list: list
        the list of ub2gm tensors, tensor type is TVM tensor.
    virtual_node:
        the tensors of virtual output node, tensor type is TVM tensor.
    """
    input_shape = shape_util.shape_to_list(input_place.shape)
    output_shape = input_shape
    for index, _ in enumerate(output_shape):
        output_shape[index] = output_shape[index] if index != axis else 1

    offset = 0
    gm2ub_tensor_list = []
    ub2gm_tensor_list = []
    for i in range(num):
        gm2ub_tensor = tvm.compute(output_shape,
                                   lambda *index: input_place(*_index_offset(output_shape, axis, offset, *index)),
                                   name=''.join(['tensor', str(i)]))
        gm2ub_tensor_list.append(gm2ub_tensor)

        ub2gm_tensor = tvm.compute(output_shape,
                                   lambda *index, tensor_in=gm2ub_tensor: tensor_in(*index),
                                   name=''.join(['res', str(i)]))
        ub2gm_tensor_list.append(ub2gm_tensor)

        offset = offset + output_shape[axis]

    # create a virtual node
    def _add_compute(*index):
        virtual_tensor = ub2gm_tensor_list[0](*index)
        for ub2gm_tensor in ub2gm_tensor_list[1:]:
            virtual_tensor += ub2gm_tensor(*index)
        return virtual_tensor

    virtual_node = tvm.compute(output_shape, lambda *index: _add_compute(*index), name="virtual_node")

    return gm2ub_tensor_list, ub2gm_tensor_list, virtual_node


# 'pylint: disable=too-many-branches,too-many-statements,too-many-locals
def _unpack_schedule(input_place, output_shape, y, num, axis, dtype):
    """
    Create unpack schedule.

    Parameters
    ----------
    input_place: TVM tensor
        the tensor of input.
    output_shape: tuple or list
        the shape of output tensor.
    y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis.
    axis: int.
        the axis to unpack along.
    dtype: str.
        the dtype of input.

    Returns
    -------
    sch: schedule
        the created schedule.
    build_list: list
        the list of input and output tensors, tensor type is TVM tensor.
    """
    _, ele_each_block, _ = _get_public_param(dtype)
    befordim, afterdim = output_shape[0], output_shape[-1]
    # can open multi-core scene
    if befordim >= ele_each_block > afterdim:
        return _unpack_schedule_one(input_place, output_shape, y, num, axis, dtype)
    else:
        return _unpack_schedule_two(input_place, output_shape, y, num, axis, dtype)


# 'pylint: disable=too-many-branches,too-many-statements,too-many-locals
def _unpack_schedule_one(input_place, output_shape, y, num, axis, dtype):
    """
    Create unpack schedule:befordim >= ele_each_block > afterdim case
    befordim: the dim axis accumulation before the axis to unpack along.
    ele_each_bloc: the num of each block size.
    afterdim: the dim axis accumulation after the axis to unpack along.

    Parameters
    ----------
    input_place: TVM tensor
        the tensor of input.
    output_shape: tuple or list
        the shape of output tensor.
    y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis.
    axis: int.
        the axis to unpack along.
    dtype: str.
        the dtype of input.

    Returns
    -------
    sch: schedule
        the created schedule.
    build_list: list
        the list of input and output tensors, tensor type is TVM tensor.
    """
    _, ele_each_block, device_core_num = _get_public_param(dtype)
    befordim, afterdim = output_shape[0], output_shape[-1]
    block_idx = tvm.thread_axis('blockIdx.x')

    befordim_in = ele_each_block // afterdim + 1
    befordim_out = (befordim + befordim_in - 1) // befordim_in
    while (befordim + befordim_out - 1) // befordim_out * afterdim < ele_each_block:
        befordim_out -= 1
    if befordim_out >= device_core_num:
        befordim_out = device_core_num
    afterdim_in = afterdim

    gm2ub_tensor, ub2ub_tensor_list, ub2gm_tensor_list, virtual_node = _unpack_compute_scalar(
        input_place, y, num, axis)

    res_op = []
    build_list = [input_place]
    for ub2gm_tensor in ub2gm_tensor_list:
        res_op.append(ub2gm_tensor.op)
        build_list.append(ub2gm_tensor)

    sch = tvm.create_schedule(virtual_node.op)
    sch[gm2ub_tensor].set_scope(tbe_platform.scope_ubuf)
    for tensor in ub2ub_tensor_list:
        sch[tensor].set_scope(tbe_platform.scope_ubuf)

    befordim_outer, befordim_inner = sch[virtual_node].split(virtual_node.op.axis[0], nparts=befordim_out)
    afterdim_outer, afterdim_inner = sch[virtual_node].split(virtual_node.op.axis[2], factor=afterdim_in)

    sch[virtual_node].reorder(befordim_outer, afterdim_outer, befordim_inner, afterdim_inner)
    fused_axis = sch[virtual_node].fuse(befordim_outer, afterdim_outer)
    sch[virtual_node].bind(fused_axis, block_idx)

    new_shape = ((befordim + befordim_out - 1) // befordim_out, num, afterdim_in)
    split_axis, split_factor = _tiling_axis(new_shape, dtype)
    if split_axis == 0:
        axis_outer, axis_inner = sch[virtual_node].split(befordim_inner, factor=split_factor)
    else:
        axis_outer, axis_inner = sch[virtual_node].split(afterdim_inner, factor=split_factor)

    sch[gm2ub_tensor].compute_at(sch[virtual_node], axis_outer)
    sch[gm2ub_tensor].emit_insn(gm2ub_tensor.op.axis[split_axis], tbe_platform.DMA_COPY)

    for i in range(num):
        sch[ub2gm_tensor_list[i]].compute_at(sch[virtual_node], axis_outer)
        sch[ub2ub_tensor_list[i]].compute_at(sch[virtual_node], axis_outer)

        sch[ub2ub_tensor_list[i]].emit_insn(ub2ub_tensor_list[i].op.axis[split_axis], tbe_platform.DATA_MOV)
        sch[ub2gm_tensor_list[i]].emit_insn(ub2gm_tensor_list[i].op.axis[split_axis], tbe_platform.DMA_COPY)

    sch[virtual_node].emit_insn(axis_inner, tbe_platform.PHONY_INSN)
    return sch, build_list


# 'pylint: disable=too-many-branches,too-many-statements,too-many-locals
def _unpack_schedule_two(input_place, output_shape, y, num, axis, dtype):
    """
    Create unpack schedule: not befordim >= ele_each_block > afterdim case
    befordim: the dim axis accumulation before the axis to unpack along.
    ele_each_bloc: the num of each block size.
    afterdim: the dim axis accumulation after the axis to unpack along.

    Parameters
    ----------
    input_place: TVM tensor
        the tensor of input.
    output_shape: tuple or list
        the shape of output tensor.
    y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis.
    axis: int.
        the axis to unpack along.
    dtype: str.
        the dtype of input.

    Returns
    -------
    sch: schedule
        the created schedule.
    build_list: list
        the list of input and output tensors, tensor type is TVM tensor.
    """
    total_ele, ele_each_block, device_core_num = _get_public_param(dtype)
    befordim, afterdim = output_shape[0], output_shape[-1]
    block_idx = tvm.thread_axis('blockIdx.x')

    gm2ub_tensor_list, ub2gm_tensor_list, virtual_node = _unpack_compute_copy(input_place, y, num, axis)
    res_op = []
    build_list = [input_place]
    for ub2gm_tensor in ub2gm_tensor_list:
        res_op.append(ub2gm_tensor.op)
        build_list.append(ub2gm_tensor)

    sch = tvm.create_schedule(virtual_node.op)
    for tensor in gm2ub_tensor_list:
        sch[tensor].set_scope(tbe_platform.scope_ubuf)

    # can open multi-core scene
    if afterdim >= ele_each_block:
        if befordim >= device_core_num:
            befordim_out = device_core_num
            afterdim_in = afterdim
        elif befordim == 1:
            befordim_out = befordim
            afterdim_in = (afterdim + device_core_num - 1) // device_core_num
        else:
            afterdim_outer = device_core_num // befordim
            afterdim_in = (afterdim + afterdim_outer - 1) // afterdim_outer
            while afterdim % afterdim_in < ele_each_block:
                afterdim_in += 1
            befordim_out = befordim

        befordim_outer, befordim_inner = sch[virtual_node].split(virtual_node.op.axis[0], nparts=befordim_out)
        afterdim_outer, afterdim_inner = sch[virtual_node].split(virtual_node.op.axis[2], factor=afterdim_in)

        sch[virtual_node].reorder(befordim_outer, afterdim_outer, befordim_inner, afterdim_inner)
        fused_axis = sch[virtual_node].fuse(befordim_outer, afterdim_outer)
        sch[virtual_node].bind(fused_axis, block_idx)

        new_shape = ((befordim + befordim_out - 1) // befordim_out, 1, afterdim_in)
        real_core_num = (befordim + new_shape[0] - 1) // new_shape[0]
        # coexisting_quantities 2
        split_axis, split_factor = _tiling_axis(new_shape, dtype, 2)
        ub_split_trail = new_shape[split_axis] - new_shape[split_axis] // split_factor * split_factor
        is_ub_split_trail_one = (ub_split_trail != 1)
        is_last_block_non_last_axis_split_trail_one = False
        if split_axis == 0:
            last_block_split_trail = (befordim - (real_core_num - 1) * new_shape[0]) % split_factor
            is_last_block_non_last_axis_split_trail_one = (last_block_split_trail != 1)
        # verify use dma_copy new attr
        if split_axis == 0 and split_factor > 1 and is_ub_split_trail_one \
           and is_last_block_non_last_axis_split_trail_one:
            is_use_dma_copy_new_attr = True
        else:
            is_use_dma_copy_new_attr = False
            # coexisting_quantities 1
            split_axis, split_factor = _tiling_axis(new_shape, dtype)
        if split_axis == 0:
            axis_outer, axis_inner = sch[virtual_node].split(befordim_inner, factor=split_factor)
        else:
            axis_outer, axis_inner = sch[virtual_node].split(afterdim_inner, factor=split_factor)
    else:
        # coexisting_quantities 2
        split_axis, split_factor = _tiling_axis(output_shape, dtype, 2)
        ub_split_trail = output_shape[split_axis] - output_shape[split_axis] // split_factor * split_factor
        is_ub_split_trail_one = (ub_split_trail != 1)
        if split_axis == 0 and split_factor > 1 and is_ub_split_trail_one:
            is_use_dma_copy_new_attr = True
        else:
            is_use_dma_copy_new_attr = False
            # coexisting_quantities 1
            split_axis, split_factor = _tiling_axis(output_shape, dtype)
        axis_outer, axis_inner = sch[virtual_node].split(virtual_node.op.axis[split_axis], factor=split_factor)

    for i in range(num):
        storage_axis = split_axis - 1 if split_axis != 0 else 0
        sch[gm2ub_tensor_list[i]].storage_align(gm2ub_tensor_list[i].op.axis[storage_axis], ele_each_block, 0)

        sch[gm2ub_tensor_list[i]].double_buffer()
        sch[gm2ub_tensor_list[i]].compute_at(sch[virtual_node], axis_outer)
        sch[ub2gm_tensor_list[i]].compute_at(sch[virtual_node], axis_outer)

        sch[gm2ub_tensor_list[i]].emit_insn(gm2ub_tensor_list[i].op.axis[split_axis], tbe_platform.DMA_COPY)
        if is_use_dma_copy_new_attr:
            # coexisting_quantities 2
            sch[gm2ub_tensor_list[i]].set_buffer_size(total_ele // 2)
            sch[ub2gm_tensor_list[i]].set_buffer_size(total_ele // 2)
            sch[ub2gm_tensor_list[i]].emit_insn(ub2gm_tensor_list[i].op.axis[split_axis], tbe_platform.DMA_COPY,
                                                {"no_overlap": "process_unaliged_stride_with_malloc_buf",
                                                 "no_overlap_malloc_buf_for_tail": 0})
        else:
            sch[ub2gm_tensor_list[i]].emit_insn(ub2gm_tensor_list[i].op.axis[split_axis], tbe_platform.DMA_COPY)

    sch[virtual_node].emit_insn(axis_inner, tbe_platform.PHONY_INSN)

    return sch, build_list


# 'pylint: disable=too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.DYNAMIC_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def unpack(x, y, num=None, axis=0, kernel_name="unpack"):
    """
    unpacks the given dimension of a rank R tensor into rank (R-1) tensors.

    Parameters
    ----------
    x : dict.
        shape, dtype and format of value to be unpacked.
    y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis, automatically inferred if None(default).
    axis: int.
        the axis to unpack along.
    kernel_name : str
        cce kernel name, default value is "unpack".

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    dformat = x.get("format")
    _check_params(shape, num, axis, dformat, dtype, kernel_name)

    # infer the value of num
    real_axis = axis + len(shape) if axis < 0 else axis
    num = shape[real_axis]

    # turn the input shape into three dimensions (a, b, c), so axis = 1
    beferdim = 1
    for befer_dim in shape[0:real_axis]:
        beferdim *= befer_dim
    afterdim = 1
    for after_dim in shape[real_axis + 1:]:
        afterdim *= after_dim
    reshape = (beferdim, shape[real_axis], afterdim)

    _, _, is_use_split = _check_use_special_optimize(dtype, afterdim, flag=False)
    reshape_input = x.copy()
    reshape_input["shape"] = reshape
    real_axis = 1
    # only 1 output tensor, so output equals to input
    if num == 1:
        copy_only.copy_only(reshape_input, reshape_input, kernel_name)
    # use split
    elif is_use_split:
        split_d.split_d(reshape_input, y, split_dim=real_axis, num_split=num, kernel_name=kernel_name)
    else:
        new_dtype, afterdim, _ = _check_use_special_optimize(dtype, afterdim, flag=False)
        new_shape = (beferdim, reshape[real_axis], afterdim)

        input_place = tvm.placeholder(new_shape, name="input_place", dtype=new_dtype)
        sch, build_list = _unpack_schedule(input_place, reshape, y, num, real_axis, dtype)

        with tbe_build.build_config():
            tvm.build(sch, build_list, "cce", name=kernel_name)
