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
confusion_matrix
"""
# 'pylint: disable=locally-disabled,too-many-lines,ungrouped-imports,unused-variable,
# 'pylint: disable=too-many-locals,too-many-statements,too-many-arguments,unused-import
import math

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_select_op_base import get_op_cal_info
from tbe.dsl.instrinsic.cce_intrin_md import reset_mask_insn
from tbe.dsl.instrinsic.cce_util import CCE_AXIS


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    Constant
    """
    # one vector instruction can compute numbers
    VEC_NUMS_HALF = 64
    VEC_NUMS = 128
    # uint8 Maximum
    UINT8_MAX = 255
    # one vector instruction calculation block numbers
    VEC_BLOCK_NUMS = 8
    BITS_NUMS = 8
    # 1 block has 32 Bytes
    BYTES_PER_BLOCK = 32
    # per block has 16 float16 numbers
    FLOAT16_NUMS = 16


# 'pylint: disable = unused-argument,invalid-name
def get_op_support_info(labels,
                        predictions,
                        weights,
                        y,
                        num_classes,
                        dtype,
                        kernel_name="confusion_matrix",
                        need_build=True,
                        need_print=False):
    """
    get_op_support_info
    """
    axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def cast_to(ibuilder, data_amounts, src_buf, dst_buf):
    """
    Complete data type conversion from src_buf to dst_buf
    """
    src_dtype = src_buf.dtype
    dst_dtype = dst_buf.dtype
    if src_dtype == "float16" and dst_dtype == "int32":
        vconv_instr = "vconv_f162s32f"
        vconv_compute_num = Constant.VEC_NUMS_HALF
    elif src_dtype == "float32" and dst_dtype == "float16":
        vconv_instr = "vconv_f322f16"
        vconv_compute_num = Constant.VEC_NUMS_HALF
    elif src_dtype == "float32" and dst_dtype == "int32":
        vconv_instr = "vconv_f322s32f"
        vconv_compute_num = Constant.VEC_NUMS_HALF
    # vconv_s322f32 only support cloud_v100
    elif src_dtype == "int32" and dst_dtype == "float32":
        vconv_instr = "vconv_s322f32"
        vconv_compute_num = Constant.VEC_NUMS_HALF
    elif src_dtype == "int8" and dst_dtype == "float16":
        vconv_instr = "vconv_s82f16"
        vconv_compute_num = Constant.VEC_NUMS
    elif src_dtype == "uint8" and dst_dtype == "float16":
        vconv_instr = "vconv_u82f16"
        vconv_compute_num = Constant.VEC_NUMS
    elif src_dtype == "float16" and dst_dtype == "float32":
        vconv_instr = "vconv_f162f32"
        vconv_compute_num = Constant.VEC_NUMS_HALF
    elif src_dtype == "float16" and dst_dtype == "int8":
        vconv_instr = "vconv_f162s8f"
        vconv_compute_num = Constant.VEC_NUMS
    elif src_dtype == "float16" and dst_dtype == "uint8":
        vconv_instr = "vconv_f162u8f"
        vconv_compute_num = Constant.VEC_NUMS

    def compute_stride(src_type, dst_type, vconv_num):
        """
        Calculated stride value
        """
        perblock_nums_a = compute_perblock_nums(src_type)
        perblock_nums_b = compute_perblock_nums(dst_type)
        src_stride = vconv_num // perblock_nums_a
        dst_stride = vconv_num // perblock_nums_b

        return src_stride, dst_stride

    src_strides, dst_strides = compute_stride(src_dtype, dst_dtype, vconv_compute_num)

    # recheck vconv_instr support
    if not tbe_platform.intrinsic_check_support("Intrinsic_vconv", \
           vconv_instr.split('_')[1]):
        error_detail = "This product don't support Intrinsic_vconv " + \
                           vconv_instr
        error_manager_vector.raise_err_specific_reson("confusion_matrix", error_detail)

    repeats = int(data_amounts // vconv_compute_num)
    remain = int(data_amounts % vconv_compute_num)
    init_times = int(repeats // Constant.UINT8_MAX)
    init_remain = int(repeats % Constant.UINT8_MAX)
    with ibuilder.if_scope(repeats != 0):
        if init_times != 0:
            with ibuilder.for_range(0, init_times) as rch:
                with ibuilder.new_scope():
                    reset_mask_insn(
                        ibuilder, dst_buf.dtype, bits=vconv_compute_num)
                    ibuilder.emit(tvm.call_extern(dst_buf.dtype, vconv_instr, \
                                                  dst_buf.access_ptr('w', offset=rch * Constant.UINT8_MAX
                                                                     * vconv_compute_num), \
                                                  src_buf.access_ptr('r', offset=rch * Constant.UINT8_MAX
                                                                     * vconv_compute_num), \
                                                  255, 1, 1, dst_strides, src_strides))
        if init_remain != 0:
            with ibuilder.new_scope():
                reset_mask_insn(ibuilder, dst_buf.dtype, bits=vconv_compute_num)
                ibuilder.emit(tvm.call_extern(dst_buf.dtype, vconv_instr, \
                                              dst_buf.access_ptr('w', offset=init_times * Constant.UINT8_MAX
                                                                 * vconv_compute_num), \
                                              src_buf.access_ptr('r', offset=init_times * Constant.UINT8_MAX
                                                                 * vconv_compute_num), \
                                              init_remain, 1, 1, dst_strides, src_strides))

    with ibuilder.if_scope(remain != 0):
        with ibuilder.new_scope():
            mask_len = remain
            reset_mask_insn(ibuilder, dst_buf.dtype, bits=mask_len)
            ibuilder.emit(tvm.call_extern(dst_buf.dtype, vconv_instr, \
                                          dst_buf.access_ptr('w', offset=repeats
                                                             * vconv_compute_num), \
                                          src_buf.access_ptr('r', offset=repeats
                                                             * vconv_compute_num), \
                                          1, 1, 1, 0, 0))


def vector_dump_set(ibuilder, scalar, block_num, buf):
    """
    do vector_dump only support float16,float32,int32
    """
    vec_dtype = buf.dtype
    if vec_dtype in ["float32", "int32"]:
        vec_compute_nums = Constant.VEC_NUMS_HALF
    else:
        vec_compute_nums = Constant.VEC_NUMS

    repeat_times = int(block_num // Constant.VEC_BLOCK_NUMS)
    remain_len = int(block_num % Constant.VEC_BLOCK_NUMS)
    init_times = int(repeat_times // Constant.UINT8_MAX)
    init_remain = int(repeat_times % Constant.UINT8_MAX)

    with ibuilder.if_scope(repeat_times != 0):
        if init_times != 0:
            with ibuilder.for_range(0, init_times) as rch:
                with ibuilder.new_scope():
                    reset_mask_insn(ibuilder, buf.dtype, bits=vec_compute_nums)
                    ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 2)
                    ibuilder.emit(tvm.call_extern(buf.dtype, "vector_dup", \
                                                  buf.access_ptr('w', offset=rch * Constant.UINT8_MAX
                                                                 * vec_compute_nums), \
                                                  tvm.const(scalar, dtype=vec_dtype), \
                                                  255, 1, 1, 8, 8))
        if init_remain != 0:
            with ibuilder.new_scope():
                reset_mask_insn(ibuilder, buf.dtype, bits=vec_compute_nums)
                ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 2)
                ibuilder.emit(tvm.call_extern(buf.dtype, "vector_dup", \
                                              buf.access_ptr('w', offset=init_times * Constant.UINT8_MAX
                                                             * vec_compute_nums), \
                                              tvm.const(scalar, dtype=vec_dtype), \
                                              init_remain, 1, 1, 8, 8))

    with ibuilder.if_scope(remain_len != 0):
        with ibuilder.new_scope():
            mask_len = remain_len * (vec_compute_nums // Constant.VEC_BLOCK_NUMS)
            reset_mask_insn(ibuilder, buf.dtype, bits=mask_len)
            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 2)
            ibuilder.emit(tvm.call_extern(buf.dtype, "vector_dup", \
                                          buf.access_ptr('w', offset=repeat_times
                                                         * vec_compute_nums), \
                                          tvm.const(scalar, dtype=vec_dtype), 1, 1, 1, 8, 8))


def apply_for_new_alloc(ibuilder,
                        dtype,
                        shape,
                        scope=tbe_platform.scope_ubuf,
                        name='tmp_buf'):
    """
    buffer allocation

    """
    buf_var = ibuilder.allocate(dtype, shape, name=name, scope=scope)
    tmp_buffer = tvm.decl_buffer(
        shape, buf_var.dtype, name=name, scope=scope, data=buf_var)
    return tmp_buffer


# 'pylint: disable=locally-disabled,too-many-branches
def params_check(shape_labels, shape_predictions, out_type, labels_dtype,
                 predictions_dtype, shape_weights, weights_dtype):
    """
    params_check for confusion_matrix op

    """

    para_check.check_shape(shape_labels, min_rank=1, max_rank=1, param_name="labels")
    para_check.check_shape(shape_predictions, min_rank=1, max_rank=1, param_name="predictions")
    if list(shape_labels) != list(shape_predictions):
        error_detail = "The shape of labels and predictions shoud be same"
        error_manager_vector.raise_err_two_input_shape_invalid("confusion_matrix", "labels", \
                                                               "predictions", error_detail)
    if shape_weights is not None:
        para_check.check_shape(shape_weights, min_rank=1, max_rank=1, param_name="weights")
        if list(shape_labels) != list(shape_weights):
            error_detail = "The shape of labels and weights shoud be same"
            error_manager_vector.raise_err_two_input_shape_invalid("confusion_matrix", "labels", \
                                                               "weights", error_detail)

    check_list = ["float32", "int32", "float16", "int8", "uint8"]
    if out_type not in check_list:
        error_manager_vector.raise_err_input_dtype_not_supported("confusion_matrix", "dtype", \
                                                                 check_list, out_type)
    if labels_dtype not in check_list:
        error_manager_vector.raise_err_input_dtype_not_supported("confusion_matrix", "labels", \
                                                                 check_list, labels_dtype)
    if predictions_dtype not in check_list:
        error_manager_vector.raise_err_input_dtype_not_supported("confusion_matrix", "predictions", \
                                                                 check_list, predictions_dtype)
    if shape_weights is not None:
        if weights_dtype not in check_list:
            error_manager_vector.raise_err_input_dtype_not_supported("confusion_matrix", "weights", \
                                                                     check_list, weights_dtype)

    if shape_weights is not None:
        if not tbe_platform.intrinsic_check_support(
                "Intrinsic_vconv", \
               "s322f32") and weights_dtype == "int32" and out_type != "int32":
            error_detail = "This product weights don't support int32(when out_type is not int32)"
            error_manager_vector.raise_err_specific_reson("confusion_matrix", error_detail)
        if not tbe_platform.intrinsic_check_support(\
                "Intrinsic_vconv", "f322s32f") and weights_dtype == "float32" \
               and out_type == "int32":
            error_detail = "This product weights don't support float32(when out_type is int32)"
            error_manager_vector.raise_err_specific_reson("confusion_matrix", error_detail)
    if not tbe_platform.intrinsic_check_support(\
            "Intrinsic_vconv", "f322s32f") and labels_dtype == "float32":
        error_manager_vector.raise_err_input_dtype_not_supported("confusion_matrix", "labels",
                                                                 "not float32", "float32")
    if not tbe_platform.intrinsic_check_support("Intrinsic_vconv", \
           "f322s32f") and predictions_dtype == "float32":
        error_manager_vector.raise_err_input_dtype_not_supported("confusion_matrix", "predictions",
                                                                 "not float32", "float32")


def compute_ub_length(number, dtype_a, dtype_b):
    """
    compute predictions ub length
    """
    ub_size = 15 * 1024  # define ub size
    type_size_a = tbe_platform.get_bit_len(dtype_a) // Constant.BITS_NUMS
    type_size_b = tbe_platform.get_bit_len(dtype_b) // Constant.BITS_NUMS

    if number * (type_size_a + type_size_b) > ub_size:  # label numbers
        length = ub_size // (type_size_a + type_size_b)
        label_factor = math.ceil(number / length)
    else:
        length = number
        label_factor = 1

    return length, label_factor


def compute_perblock_nums(dtype):
    """
    Calculate the numbers of this data type every block have
    """
    type_size = tbe_platform.get_bit_len(dtype) // Constant.BITS_NUMS
    perblock_nums = Constant.BYTES_PER_BLOCK // type_size

    return perblock_nums


def compute_outub_size(height, width, dtype, core_nums):
    """
    calculate the ubuf size for confusion_matrix, and 100*1024 is used for labels/pre/weight
    """
    ubuf_size = 100 * 1024  # ub whole size 100 * 1024 byte
    out_ele_perblock = compute_perblock_nums(dtype)
    out_blocks = math.ceil(height * width / out_ele_perblock)
    block_per_core = math.ceil(out_blocks / core_nums)
    use_cores = math.ceil(out_blocks / block_per_core)
    out_ele_size = tbe_platform.get_bit_len(dtype) // Constant.BITS_NUMS
    out_f16_size = tbe_platform.get_bit_len("float16") // Constant.BITS_NUMS
    out_int8_size = tbe_platform.get_bit_len("int8") // Constant.BITS_NUMS
    if dtype in ["int8", "uint8"]:
        need_size = block_per_core * out_ele_perblock * (out_f16_size + out_int8_size)
        if need_size > ubuf_size:
            block_num = ubuf_size // (out_ele_perblock * (out_f16_size + out_int8_size))
            out_factor = math.ceil(block_per_core / block_num)
            last_remian = block_per_core % block_num
        else:
            block_num = block_per_core
            out_factor = 1
            last_remian = 0
        total_len = block_num * out_ele_perblock
    else:
        need_size = block_per_core * out_ele_size * out_ele_perblock
        if need_size > ubuf_size:
            block_num = ubuf_size // Constant.BYTES_PER_BLOCK
            out_factor = math.ceil(block_per_core / block_num)
            last_remian = block_per_core % block_num
        else:
            block_num = block_per_core
            out_factor = 1
            last_remian = 0
        total_len = block_num * out_ele_perblock

    tiling_args = [block_num, block_per_core, out_factor, last_remian, total_len, use_cores]
    return tiling_args


def applyub_by_length(ibuilder, length, dtype, buf_name):
    """
    apply ub size by length
    """
    num_per_block = compute_perblock_nums(dtype)
    apply_buf = apply_for_new_alloc(
        ibuilder,
        dtype, (int(length // num_per_block) + 1, num_per_block),
        scope=tbe_platform.scope_ubuf,
        name=buf_name)

    return apply_buf


def copy_weight_to_ub(ibuilder, gm_in, dst_buf, out, length):
    """
    copy_weight_to_ub
    """
    num_per_block = compute_perblock_nums(dst_buf.dtype)
    with ibuilder.new_scope():
        ibuilder.emit(tvm.call_extern(dst_buf.dtype, "copy_gm_to_ubuf", \
                                      dst_buf.access_ptr('w', offset=0), \
                                      gm_in.access_ptr('r', offset=out * length), \
                                      0, 1, int(length // num_per_block) + 1, 0, 0))


# 'pylint: disable=locally-disabled,too-many-branches
def confusion_matrix_ir(labels, prediction, weight, output):
    """
    Generate the confusion_matrix op IR

    Parameters
    ----------
    labels     : Tensor
        Tensor of labels
    prediction : Tensor
        Tensor of prdiction
    weight     : Tensor
        Tensor of weight
    w_dtype    : int or fp
        the type of weight
    output     : Tensor
        tensor of confusion_matrix

    Returns
    -------
    IR
        ir of confusion_matrix op
    """
    labels_dtype = labels.dtype
    prediction_dtype = prediction.dtype
    weight_dtype = weight.dtype
    output_dtype = output.dtype
    labels_perblock_nums = compute_perblock_nums(labels_dtype)
    prediction_perblock_nums = compute_perblock_nums(prediction_dtype)
    weight_perblock_nums = compute_perblock_nums(weight_dtype)
    output_perblock_nums = compute_perblock_nums(output_dtype)

    height = list(output.shape)[0].value
    width = list(output.shape)[1].value
    number = list(labels.shape)[0].value
    length, label_factor = compute_ub_length(number, labels_dtype,
                                             prediction_dtype)

    # ================== IR builder Initial ==================
    # apply for register
    ibuilder = tvm.tir.ir_builder.create()
    device_core_num = tbe_platform.get_soc_spec(
        tbe_platform.CORE_NUM)
    core_num = tvm.thread_axis("blockIdx.x")
    ibuilder.scope_attr(core_num, "thread_extent", device_core_num)

    # apply for confusion_buf,weight_buf
    tiling_args = compute_outub_size(height, width, output_dtype, device_core_num)
    block_num, block_per_core, out_factor, last_remian, total_len, use_cores = tiling_args
    block_num_int8 = int(total_len // Constant.BYTES_PER_BLOCK) + 1
    block_num_fp16 = int(total_len // Constant.FLOAT16_NUMS) + 1
    if output_dtype in ["int8", "uint8"]:
        if weight_dtype in ["int8", "uint8", "float32"]:
            confusion_buf = apply_for_new_alloc(ibuilder, "float16", (block_num_fp16, Constant.FLOAT16_NUMS), \
                                                scope=tbe_platform.scope_ubuf, name="confusion_buf")
            confusion_buf_out = apply_for_new_alloc(ibuilder, output_dtype,
                                                    (block_num_int8, output_perblock_nums), \
                                                    scope=tbe_platform.scope_ubuf, name="confusion_buf_out")
            weight_buf_in = applyub_by_length(ibuilder, length, weight_dtype,
                                              "weight_buf_in")
            weight_buf = applyub_by_length(ibuilder, length, "float16",
                                           "weight_buf")
            weight_castto_dtype = "float16"
        else:
            if weight_dtype == "float16":
                confusion_buf = apply_for_new_alloc(ibuilder, weight_dtype,
                                                    (block_num_fp16, weight_perblock_nums), \
                                                    scope=tbe_platform.scope_ubuf, name="confusion_buf")
                weight_buf = applyub_by_length(ibuilder, length, weight_dtype,
                                               "weight_buf")
                weight_castto_dtype = weight_dtype
                confusion_buf_out = apply_for_new_alloc(ibuilder, output_dtype,
                                                        (block_num_int8, output_perblock_nums), \
                                                        scope=tbe_platform.scope_ubuf,
                                                        name="confusion_buf_out")
            else:
                # weight_dtype == "int32" only support cloud
                confusion_buf = apply_for_new_alloc(ibuilder, "float16", (block_num_fp16, Constant.FLOAT16_NUMS),
                                                    scope=tbe_platform.scope_ubuf, name="confusion_buf")
                confusion_buf_out = apply_for_new_alloc(ibuilder, output_dtype,
                                                        (block_num_int8, output_perblock_nums),
                                                        scope=tbe_platform.scope_ubuf,
                                                        name="confusion_buf_out")
                weight_buf_in = applyub_by_length(ibuilder, length,
                                                  weight_dtype, "weight_buf_in")
                weight_buf_fp32 = applyub_by_length(ibuilder, length, "float32",
                                                    "weight_buf_fp32")
                weight_buf = applyub_by_length(ibuilder, length, "float16",
                                               "weight_buf")
                weight_castto_dtype = "float16"
    else:
        confusion_buf = apply_for_new_alloc(ibuilder, output_dtype,
                                            (block_num, output_perblock_nums), \
                                            scope=tbe_platform.scope_ubuf, name="confusion_buf")
        weight_castto_dtype = output_dtype
        if output_dtype == "float16":
            if weight_dtype == "float16":
                weight_buf = applyub_by_length(ibuilder, length, weight_dtype,
                                               "weight_buf")
            elif weight_dtype in ["int8", "uint8", "float32"]:
                weight_buf_in = applyub_by_length(ibuilder, length,
                                                  weight_dtype, "weight_buf_in")
                weight_buf = applyub_by_length(ibuilder, length, "float16",
                                               "weight_buf")
            # int32 only support cloud (int32-->float32-->float16)
            else:
                weight_buf_in = applyub_by_length(ibuilder, length,
                                                  weight_dtype, "weight_buf_in")
                weight_buf_fp32 = applyub_by_length(ibuilder, length, "float32",
                                                    "weight_buf_fp32")
                weight_buf = applyub_by_length(ibuilder, length, "float16",
                                               "weight_buf")
        elif output_dtype == "float32":
            if weight_dtype == "float32":
                weight_buf = applyub_by_length(ibuilder, length, weight_dtype,
                                               "weight_buf")
            elif weight_dtype in ["int8", "uint8"]:
                weight_buf_in = applyub_by_length(ibuilder, length,
                                                  weight_dtype, "weight_buf_in")
                weight_buf_fp16 = applyub_by_length(ibuilder, length, "float16",
                                                    "weight_buf_fp16")
                weight_buf = applyub_by_length(ibuilder, length, "float32",
                                               "weight_buf")
            else:
                # int32 float16
                weight_buf_in = applyub_by_length(ibuilder, length,
                                                  weight_dtype, "weight_buf_in")
                weight_buf = applyub_by_length(ibuilder, length, "float32",
                                               "weight_buf")
        else:
            if weight_dtype == "int32":
                weight_buf = applyub_by_length(ibuilder, length, weight_dtype,
                                               "weight_buf")
            elif weight_dtype in ["int8", "uint8"]:
                weight_buf_in = applyub_by_length(ibuilder, length,
                                                  weight_dtype, "weight_buf_in")
                weight_buf_fp16 = applyub_by_length(ibuilder, length, "float16",
                                                    "weight_buf_fp16")
                weight_buf = applyub_by_length(ibuilder, length, "int32",
                                               "weight_buf")
            else:
                weight_buf_in = applyub_by_length(ibuilder, length,
                                                  weight_dtype, "weight_buf_in")
                weight_buf = applyub_by_length(ibuilder, length, "int32",
                                               "weight_buf")

    reg_buf = ibuilder.allocate("int32", (2,), scope=tbe_platform.scope_reg, name="reg_buf")
    reg_tmp = ibuilder.allocate(weight_castto_dtype, (4,), scope=tbe_platform.scope_reg, name="reg_tmp")
    # apply for weights add ub_buf
    value_buf = apply_for_new_alloc(ibuilder, weight_castto_dtype, (Constant.BITS_NUMS,), \
                                    scope=tbe_platform.scope_ubuf, name="value_buf")
    value_buf1 = apply_for_new_alloc(ibuilder, weight_castto_dtype, (Constant.BITS_NUMS,), \
                                     scope=tbe_platform.scope_ubuf, name="value_buf1")

    # apply for label_buf, predict_buf
    label_buf_int32 = apply_for_new_alloc(ibuilder, "int32",
                                          (int(length // Constant.BITS_NUMS) + 1, Constant.BITS_NUMS), \
                                          scope=tbe_platform.scope_ubuf, name="label_buf_int32")
    predict_buf_int32 = apply_for_new_alloc(ibuilder, "int32",
                                            (int(length // Constant.BITS_NUMS) + 1, Constant.BITS_NUMS), \
                                            scope=tbe_platform.scope_ubuf, name="predict_buf_int32")
    if labels_dtype != "int32":
        label_buf = apply_for_new_alloc(ibuilder, labels_dtype,
                                        (int(length // labels_perblock_nums) + 1,
                                         labels_perblock_nums), \
                                        scope=tbe_platform.scope_ubuf, name="label_buf")
    if labels_dtype in ("int8", "uint8"):
        cast_fp16_buf_a = apply_for_new_alloc(ibuilder, "float16",
                                              (int(length // Constant.FLOAT16_NUMS) + 1, Constant.FLOAT16_NUMS), \
                                              scope=tbe_platform.scope_ubuf, name="cast_fp16_buf_a")
    if prediction_dtype != "int32":
        predict_buf = apply_for_new_alloc(ibuilder, prediction_dtype,
                                          (int(length // prediction_perblock_nums) + 1,
                                           prediction_perblock_nums), \
                                          scope=tbe_platform.scope_ubuf, name="predict_buf")
    if prediction_dtype in ("int8", "uint8"):
        cast_fp16_buf_b = apply_for_new_alloc(ibuilder, "float16",
                                              (int(length // Constant.FLOAT16_NUMS) + 1, Constant.FLOAT16_NUMS), \
                                              scope=tbe_platform.scope_ubuf, name="cast_fp16_buf_b")

    with ibuilder.for_range(0, out_factor) as out:
        # initilation for confusion_buf
        if output_dtype in ["int8", "uint8"]:
            vector_dump_set(ibuilder, 0, block_num_fp16, confusion_buf)
        else:
            vector_dump_set(ibuilder, 0, block_num, confusion_buf)

        # ================== Traverse the value of labels/prediction ==================
        with ibuilder.for_range(0, label_factor) as b_out:
            # labels --> label_buf --> int32
            if labels_dtype != "int32":
                copy_weight_to_ub(ibuilder, labels, label_buf, b_out, length)
                if labels_dtype in ("int8", "uint8"):
                    cast_to(ibuilder, length, label_buf, cast_fp16_buf_a)
                    cast_to(ibuilder, length, cast_fp16_buf_a, label_buf_int32)
                else:
                    cast_to(ibuilder, length, label_buf, label_buf_int32)
            else:
                copy_weight_to_ub(ibuilder, labels, label_buf_int32, b_out,
                                  length)

            # predictions --> predict_buf --> int32
            if prediction_dtype != "int32":
                copy_weight_to_ub(ibuilder, prediction, predict_buf, b_out,
                                  length)
                if prediction_dtype in ("int8", "uint8"):
                    cast_to(ibuilder, length, predict_buf, cast_fp16_buf_b)
                    cast_to(ibuilder, length, cast_fp16_buf_b,
                            predict_buf_int32)
                else:
                    cast_to(ibuilder, length, predict_buf, predict_buf_int32)
            else:
                copy_weight_to_ub(ibuilder, prediction, predict_buf_int32,
                                  b_out, length)

            # weight --> weight_buf --> weight_castto_dtype
            if output_dtype in ["int8", "uint8"]:
                if weight_dtype in ["int8", "uint8", "float32"]:
                    copy_weight_to_ub(ibuilder, weight, weight_buf_in, b_out,
                                      length)
                    cast_to(ibuilder, length, weight_buf_in, weight_buf)
                else:
                    if weight_dtype == "float16":
                        copy_weight_to_ub(ibuilder, weight, weight_buf, b_out,
                                          length)
                    else:
                        copy_weight_to_ub(ibuilder, weight, weight_buf_in,
                                          b_out, length)
                        cast_to(ibuilder, length, weight_buf_in,
                                weight_buf_fp32)
                        cast_to(ibuilder, length, weight_buf_fp32, weight_buf)
            else:
                if output_dtype == "float16":
                    if weight_dtype == "float16":
                        copy_weight_to_ub(ibuilder, weight, weight_buf, b_out,
                                          length)
                    elif weight_dtype in ["int8", "uint8", "float32"]:
                        copy_weight_to_ub(ibuilder, weight, weight_buf_in,
                                          b_out, length)
                        cast_to(ibuilder, length, weight_buf_in, weight_buf)
                    else:
                        # int32 only support cloud (int32-->float32-->float16)
                        copy_weight_to_ub(ibuilder, weight, weight_buf_in,
                                          b_out, length)
                        cast_to(ibuilder, length, weight_buf_in,
                                weight_buf_fp32)
                        cast_to(ibuilder, length, weight_buf_fp32, weight_buf)
                elif output_dtype == "float32":
                    if weight_dtype == "float32":
                        copy_weight_to_ub(ibuilder, weight, weight_buf, b_out,
                                          length)
                    elif weight_dtype in ["int8", "uint8"]:
                        copy_weight_to_ub(ibuilder, weight, weight_buf_in,
                                          b_out, length)
                        cast_to(ibuilder, length, weight_buf_in,
                                weight_buf_fp16)
                        cast_to(ibuilder, length, weight_buf_fp16, weight_buf)
                    else:
                        # int32 float16
                        copy_weight_to_ub(ibuilder, weight, weight_buf_in,
                                          b_out, length)
                        cast_to(ibuilder, length, weight_buf_in, weight_buf)
                else:
                    if weight_dtype == "int32":
                        copy_weight_to_ub(ibuilder, weight, weight_buf, b_out,
                                          length)
                    elif weight_dtype in ["int8", "uint8"]:
                        copy_weight_to_ub(ibuilder, weight, weight_buf_in,
                                          b_out, length)
                        cast_to(ibuilder, length, weight_buf_in,
                                weight_buf_fp16)
                        cast_to(ibuilder, length, weight_buf_fp16, weight_buf)
                    else:
                        copy_weight_to_ub(ibuilder, weight, weight_buf_in,
                                          b_out, length)
                        cast_to(ibuilder, length, weight_buf_in, weight_buf)

            # single value loop
            with ibuilder.for_range(0, length) as i:
                with ibuilder.if_scope((b_out * length + i) < number):
                    # label <==> reg_buf[0]
                    with ibuilder.new_scope():
                        ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                        ibuilder.emit(tvm.call_extern("int32", "reg_mov", \
                                                      tvm.call_extern("int32", "reg", reg_buf[0]), \
                                                      label_buf_int32.access_ptr("r", offset=i), 0))
                    # prediction <==> reg_buf[1]
                    with ibuilder.new_scope():
                        ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                        ibuilder.emit(tvm.call_extern("int32", "reg_mov", \
                                                      tvm.call_extern("int32", "reg", reg_buf[1]), \
                                                      predict_buf_int32.access_ptr("r", \
                                                                                   offset=i), 0))

                    with ibuilder.if_scope(tvm.all(reg_buf[0]*width+reg_buf[1] >= out *
                                                   total_len+block_per_core*output_perblock_nums*core_num,
                                                   reg_buf[0]*width+reg_buf[1] < (out+1) *
                                                   total_len+block_per_core*output_perblock_nums*core_num)):

                        # weight <==> reg_tmp[0]
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg", reg_tmp[0]), \
                                                          weight_buf.access_ptr("r", offset=i), 0))

                        # ==========Values in the same position need to be superimposed==========
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg", reg_tmp[1]), \
                                                          confusion_buf.access_ptr("r", \
                                                          offset=reg_buf[0] * width + reg_buf[1] - out * \
                                              total_len-block_per_core*output_perblock_nums*core_num), 0))


                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          value_buf.access_ptr("w", offset=0), \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg", reg_tmp[0]), ))

                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          value_buf1.access_ptr("w", offset=0), \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg",
                                                                          reg_tmp[1]), ))

                        with ibuilder.new_scope():
                            mask_len = 8
                            reset_mask_insn(
                                ibuilder, value_buf.dtype, bits=mask_len)
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 2)
                            ibuilder.emit(tvm.call_extern(value_buf.dtype, "vadd", \
                                                          value_buf.access_ptr('w', offset=0), \
                                                          value_buf.access_ptr('r', offset=0), \
                                                          value_buf1.access_ptr('r', offset=0), \
                                                          1, 0, 0, 0, 0, 0, 0))

                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg", reg_tmp[0]), \
                                                          value_buf.access_ptr("r", offset=0), 0))

                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          confusion_buf.access_ptr("w", \
                                                          offset=reg_buf[0] * width + reg_buf[1] - out * \
                                                total_len-block_per_core*output_perblock_nums*core_num), \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg", reg_tmp[0]), ))

        # output
        with ibuilder.if_scope(core_num < use_cores):
            if output_dtype in ["int32", "float16", "float32"]:
                if last_remian != 0:
                    with ibuilder.if_scope(tvm.all(out == (out_factor - 1))):
                        last_block_num = last_remian
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                            ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                          output.access_ptr('w', offset=out * total_len + \
                                                          block_per_core*output_perblock_nums*core_num), \
                                                          confusion_buf.access_ptr('r', offset=0), \
                                                          0, 1, last_block_num, 0, 0))
                    with ibuilder.else_scope():
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                            ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                          output.access_ptr('w', offset=out * total_len + \
                                                          block_per_core*output_perblock_nums*core_num), \
                                                          confusion_buf.access_ptr('r', offset=0), \
                                                          0, 1, block_num, 0, 0))
                else:
                    with ibuilder.new_scope():
                        ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                        ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                      output.access_ptr('w', offset=out * total_len + \
                                                      block_per_core*output_perblock_nums*core_num), \
                                                      confusion_buf.access_ptr('r', offset=0), \
                                                      0, 1, block_num, 0, 0))

            else:
                cast_to(ibuilder, total_len, confusion_buf, confusion_buf_out)
                if last_remian != 0:
                    with ibuilder.if_scope(tvm.all(out == (out_factor - 1))):
                        last_block_num = last_remian
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                            ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                          output.access_ptr('w', offset=out * total_len + \
                                                          block_per_core*output_perblock_nums*core_num), \
                                                          confusion_buf_out.access_ptr('r', offset=0), \
                                                          0, 1, last_block_num, 0, 0))
                    with ibuilder.else_scope():
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                            ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                          output.access_ptr('w', offset=out * total_len + \
                                                          block_per_core*output_perblock_nums*core_num), \
                                                          confusion_buf_out.access_ptr('r', offset=0), \
                                                          0, 1, block_num, 0, 0))
                else:
                    with ibuilder.new_scope():
                        ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                        ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                      output.access_ptr('w', offset=out * total_len + \
                                                      block_per_core*output_perblock_nums*core_num), \
                                                      confusion_buf_out.access_ptr('r', offset=0), \
                                                      0, 1, block_num, 0, 0))

    return ibuilder.get()


def confusion_matrix_ir_weight_none(labels, prediction, output):
    """
    Generate the confusion_matrix op IR

    Parameters
    ----------
    labels     : Tensor
        Tensor of labels
    prediction : Tensor
        Tensor of prdiction
    weight     : Tensor
        Tensor of weight
    w_dtype    : int or fp
        the type of weight
    output     : Tensor
        tensor of confusion_matrix

    Returns
    -------
    IR
        ir of confusion_matrix op
    """
    labels_dtype = labels.dtype
    prediction_dtype = prediction.dtype
    output_dtype = output.dtype
    labels_perblock_nums = compute_perblock_nums(labels_dtype)
    prediction_perblock_nums = compute_perblock_nums(prediction_dtype)
    output_perblock_nums = compute_perblock_nums(output_dtype)

    height = list(output.shape)[0].value
    width = list(output.shape)[1].value
    number = list(labels.shape)[0].value
    length, label_factor = compute_ub_length(number, labels_dtype,
                                             prediction_dtype)

    # ================== IR builder Initial ==================
    # apply for register
    ibuilder = tvm.tir.ir_builder.create()
    device_core_num = tbe_platform.get_soc_spec(
        tbe_platform.CORE_NUM)
    core_num = tvm.thread_axis("blockIdx.x")
    ibuilder.scope_attr(core_num, "thread_extent", device_core_num)

    # apply for confusion_buf,weight_buf
    tiling_args = compute_outub_size(height, width, output_dtype, device_core_num)
    block_num, block_per_core, out_factor, last_remian, total_len, use_cores = tiling_args
    block_num_int8 = int(total_len // Constant.BYTES_PER_BLOCK) + 1
    block_num_fp16 = int(total_len // Constant.FLOAT16_NUMS) + 1

    if output_dtype in ["int8", "uint8"]:
        confusion_buf = apply_for_new_alloc(ibuilder, "float16", (block_num_fp16, Constant.FLOAT16_NUMS), \
                                            scope=tbe_platform.scope_ubuf, name="confusion_buf")
        confusion_buf_out = apply_for_new_alloc(ibuilder, output_dtype,
                                                (block_num_int8, output_perblock_nums), \
                                                scope=tbe_platform.scope_ubuf, name="confusion_buf_out")
        weight_buf = applyub_by_length(ibuilder, length, "float16",
                                       "weight_buf")
        weight_castto_dtype = "float16"

    else:
        confusion_buf = apply_for_new_alloc(ibuilder, output_dtype,
                                            (block_num, output_perblock_nums), \
                                            scope=tbe_platform.scope_ubuf, name="confusion_buf")
        weight_buf = applyub_by_length(ibuilder, length, output_dtype,
                                       "weight_buf")
        weight_castto_dtype = output_dtype

    reg_buf = ibuilder.allocate("int32", (2,), scope=tbe_platform.scope_reg, name="reg_buf")
    reg_tmp = ibuilder.allocate(weight_castto_dtype, (4,), scope=tbe_platform.scope_reg, name="reg_tmp")
    # apply for weights add ub_buf
    value_buf = apply_for_new_alloc(ibuilder, weight_castto_dtype, (Constant.BITS_NUMS,), \
                                    scope=tbe_platform.scope_ubuf, name="value_buf")
    value_buf1 = apply_for_new_alloc(ibuilder, weight_castto_dtype, (Constant.BITS_NUMS,), \
                                     scope=tbe_platform.scope_ubuf, name="value_buf1")

    # apply for label_buf, predict_buf
    label_buf_int32 = apply_for_new_alloc(ibuilder, "int32",
                                          (int(length // Constant.BITS_NUMS) + 1, Constant.BITS_NUMS), \
                                          scope=tbe_platform.scope_ubuf, name="label_buf_int32")
    predict_buf_int32 = apply_for_new_alloc(ibuilder, "int32",
                                            (int(length // Constant.BITS_NUMS) + 1, Constant.BITS_NUMS), \
                                            scope=tbe_platform.scope_ubuf, name="predict_buf_int32")
    if labels_dtype != "int32":
        label_buf = apply_for_new_alloc(ibuilder, labels_dtype,
                                        (int(length // labels_perblock_nums) + 1,
                                         labels_perblock_nums), \
                                        scope=tbe_platform.scope_ubuf, name="label_buf")
    if labels_dtype in ("int8", "uint8"):
        cast_fp16_buf_a = apply_for_new_alloc(ibuilder, "float16",
                                              (int(length // Constant.FLOAT16_NUMS) + 1, Constant.FLOAT16_NUMS), \
                                              scope=tbe_platform.scope_ubuf, name="cast_fp16_buf_a")
    if prediction_dtype != "int32":
        predict_buf = apply_for_new_alloc(ibuilder, prediction_dtype,
                                          (int(length // prediction_perblock_nums) + 1,
                                           prediction_perblock_nums), \
                                          scope=tbe_platform.scope_ubuf, name="predict_buf")
    if prediction_dtype in ("int8", "uint8"):
        cast_fp16_buf_b = apply_for_new_alloc(ibuilder, "float16",
                                              (int(length // Constant.FLOAT16_NUMS) + 1, Constant.FLOAT16_NUMS), \
                                              scope=tbe_platform.scope_ubuf, name="cast_fp16_buf_b")

    weight_blocks = int(length // compute_perblock_nums(weight_castto_dtype)) + 1
    with ibuilder.for_range(0, out_factor) as out:
        # initilation for confusion_buf and weight_buf
        if output_dtype in ["int8", "uint8"]:
            vector_dump_set(ibuilder, 0, block_num_fp16, confusion_buf)
        else:
            vector_dump_set(ibuilder, 0, block_num, confusion_buf)
        vector_dump_set(ibuilder, 1, weight_blocks, weight_buf)

        # ================== Traverse the value of labels/prediction ==================
        with ibuilder.for_range(0, label_factor) as b_out:
            # labels --> label_buf --> int32
            if labels_dtype != "int32":
                copy_weight_to_ub(ibuilder, labels, label_buf, b_out, length)
                if labels_dtype in ("int8", "uint8"):
                    cast_to(ibuilder, length, label_buf, cast_fp16_buf_a)
                    cast_to(ibuilder, length, cast_fp16_buf_a, label_buf_int32)
                else:
                    cast_to(ibuilder, length, label_buf, label_buf_int32)
            else:
                copy_weight_to_ub(ibuilder, labels, label_buf_int32, b_out,
                                  length)

            # predictions --> predict_buf --> int32
            if prediction_dtype != "int32":
                copy_weight_to_ub(ibuilder, prediction, predict_buf, b_out,
                                  length)
                if prediction_dtype in ("int8", "uint8"):
                    cast_to(ibuilder, length, predict_buf, cast_fp16_buf_b)
                    cast_to(ibuilder, length, cast_fp16_buf_b,
                            predict_buf_int32)
                else:
                    cast_to(ibuilder, length, predict_buf, predict_buf_int32)
            else:
                copy_weight_to_ub(ibuilder, prediction, predict_buf_int32,
                                  b_out, length)

            # single value loop
            with ibuilder.for_range(0, length) as i:
                with ibuilder.if_scope((b_out * length + i) < number):
                    # label <==> reg_buf[0]
                    with ibuilder.new_scope():
                        ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                        ibuilder.emit(tvm.call_extern("int32", "reg_mov", \
                                                      tvm.call_extern("int32", "reg", reg_buf[0]), \
                                                      label_buf_int32.access_ptr("r", offset=i), 0))
                    # prediction <==> reg_buf[1]
                    with ibuilder.new_scope():
                        ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                        ibuilder.emit(tvm.call_extern("int32", "reg_mov", \
                                                      tvm.call_extern("int32", "reg", reg_buf[1]), \
                                                      predict_buf_int32.access_ptr("r", \
                                                                                   offset=i), 0))

                    with ibuilder.if_scope(tvm.all(reg_buf[0]*width+reg_buf[1] >= out *
                                                   total_len+block_per_core*output_perblock_nums*core_num,
                                                   reg_buf[0]*width+reg_buf[1] < (out+1) *
                                                   total_len+block_per_core*output_perblock_nums*core_num)):

                        # weight <==> reg_tmp[0]
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg", reg_tmp[0]), \
                                                          weight_buf.access_ptr("r", offset=i), 0))

                        # ==========Values in the same position need to be superimposed==========
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg", reg_tmp[1]), \
                                                          confusion_buf.access_ptr("r", \
                                                          offset=reg_buf[0] * width + reg_buf[1] - out * \
                                              total_len-block_per_core*output_perblock_nums*core_num), 0))


                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          value_buf.access_ptr("w", offset=0), \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg", reg_tmp[0]), ))

                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          value_buf1.access_ptr("w", offset=0), \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg",
                                                                          reg_tmp[1]), ))

                        with ibuilder.new_scope():
                            mask_len = 8
                            reset_mask_insn(
                                ibuilder, value_buf.dtype, bits=mask_len)
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 2)
                            ibuilder.emit(tvm.call_extern(value_buf.dtype, "vadd", \
                                                          value_buf.access_ptr('w', offset=0), \
                                                          value_buf.access_ptr('r', offset=0), \
                                                          value_buf1.access_ptr('r', offset=0), \
                                                          1, 0, 0, 0, 0, 0, 0))

                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg", reg_tmp[0]), \
                                                          value_buf.access_ptr("r", offset=0), 0))

                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 1)
                            ibuilder.emit(tvm.call_extern(weight_castto_dtype, "reg_mov", \
                                                          confusion_buf.access_ptr("w", \
                                                          offset=reg_buf[0] * width + reg_buf[1] - out * \
                                                total_len-block_per_core*output_perblock_nums*core_num), \
                                                          tvm.call_extern(weight_castto_dtype,
                                                                          "reg", reg_tmp[0]), ))

        # output
        with ibuilder.if_scope(core_num < use_cores):
            if output_dtype in ["int32", "float16", "float32"]:
                if last_remian != 0:
                    with ibuilder.if_scope(tvm.all(out == (out_factor - 1))):
                        last_block_num = last_remian
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                            ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                          output.access_ptr('w', offset=out * total_len + \
                                                          block_per_core*output_perblock_nums*core_num), \
                                                          confusion_buf.access_ptr('r', offset=0), \
                                                          0, 1, last_block_num, 0, 0))
                    with ibuilder.else_scope():
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                            ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                          output.access_ptr('w', offset=out * total_len + \
                                                          block_per_core*output_perblock_nums*core_num), \
                                                          confusion_buf.access_ptr('r', offset=0), \
                                                          0, 1, block_num, 0, 0))
                else:
                    with ibuilder.new_scope():
                        ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                        ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                      output.access_ptr('w', offset=out * total_len + \
                                                      block_per_core*output_perblock_nums*core_num), \
                                                      confusion_buf.access_ptr('r', offset=0), \
                                                      0, 1, block_num, 0, 0))

            else:
                cast_to(ibuilder, total_len, confusion_buf, confusion_buf_out)
                if last_remian != 0:
                    with ibuilder.if_scope(tvm.all(out == (out_factor - 1))):
                        last_block_num = last_remian
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                            ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                          output.access_ptr('w', offset=out * total_len + \
                                                          block_per_core*output_perblock_nums*core_num), \
                                                          confusion_buf_out.access_ptr('r', offset=0), \
                                                          0, 1, last_block_num, 0, 0))
                    with ibuilder.else_scope():
                        with ibuilder.new_scope():
                            ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                            ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                          output.access_ptr('w', offset=out * total_len + \
                                                          block_per_core*output_perblock_nums*core_num), \
                                                          confusion_buf_out.access_ptr('r', offset=0), \
                                                          0, 1, block_num, 0, 0))
                else:
                    with ibuilder.new_scope():
                        ibuilder.scope_attr(CCE_AXIS, "coproc_scope", 6)
                        ibuilder.emit(tvm.call_extern(output_dtype, "copy_ubuf_to_gm", \
                                                      output.access_ptr('w', offset=out * total_len + \
                                                      block_per_core*output_perblock_nums*core_num), \
                                                      confusion_buf_out.access_ptr('r', offset=0), \
                                                      0, 1, block_num, 0, 0))

    return ibuilder.get()


# @util.check_input_type((list, tuple), int, str, str, bool, bool)
# 'pylint: disable=locally-disabled,invalid-name,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_STR,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL)
def confusion_matrix(labels,
                     predictions,
                     weights,
                     y,
                     num_classes,
                     dtype,
                     kernel_name="confusion_matrix",
                     need_build=True,
                     need_print=False):
    """Generate the confusion_matrix op IR

    Parameters
    ----------
    shape : tuple
        shape of input tensor
    num_classes : var
        the length of confusion_matrix
    w_dtype : int or fp
        the dtype of weight

    kernel_name : cce kernel name, default value is confusion_matrix

    need_buid : if need to build CCEC kernel, default value is True

    need_print : if need to print the ir, default value is False

    Returns
    -------        None
    """

    shape = labels.get("shape")
    para_check.check_shape(shape, param_name="labels")
    shape_predictions = predictions.get("shape")
    para_check.check_shape(shape_predictions, param_name="predictions")
    if weights is not None:
        shape_weights = weights.get("shape")
        para_check.check_shape(shape_weights, param_name="weights")
        weights_dtype = weights.get("dtype").lower()
    else:
        shape_weights = None
        weights_dtype = None
    labels_dtype = labels.get("dtype").lower()
    predictions_dtype = predictions.get("dtype").lower()

    params_check(shape, shape_predictions, dtype, labels_dtype,
                 predictions_dtype, shape_weights, weights_dtype)

    labels = tvm.placeholder(shape, dtype=labels_dtype, name="labels")
    prediction = tvm.placeholder(
        shape, dtype=predictions_dtype, name="prediction")
    out_shape = (num_classes, num_classes)
    if weights is not None:
        weight = tvm.placeholder(shape, dtype=weights_dtype, name="weight")
        res = tvm.extern([out_shape], [labels, prediction, weight],
                         lambda ins, outs: confusion_matrix_ir(ins[0], ins[1], ins[2], output=outs[0]),
                         dtype=dtype, name=kernel_name)
        sch = tvm.create_schedule([res.op])
        if need_build:
            with tbe_build.build_config():
                mod = tvm.build(
                    sch, [labels, prediction, weight, res],
                    "cce",
                    name=kernel_name)
    else:
        res = tvm.extern([out_shape], [labels, prediction], \
                         lambda ins, outs: confusion_matrix_ir_weight_none(ins[0], ins[1],
                                                                           output=outs[0]),
                         dtype=dtype, name=kernel_name)
        sch = tvm.create_schedule([res.op])
        if need_build:
            with tbe_build.build_config():
                mod = tvm.build(
                    sch, [labels, prediction, res], "cce", name=kernel_name)
