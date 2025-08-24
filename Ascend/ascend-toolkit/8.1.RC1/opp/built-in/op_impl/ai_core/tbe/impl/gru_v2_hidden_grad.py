#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

lstm_grad
"""

# 'pylint: disable=locally-disabled,import-error,unused-import,ungrouped-imports
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.platform import insn_cmd
from te import tik
from tbe import tvm


# 'pylint: disable=too-many-locals,too-many-arguments
def copy_input_to_ub_compute(tensor_h1, tensor_n2,
                             tensor_i2, tensor_r2, tensor_n2_mid,
                             tensor_d_h2, tensor_dy,
                             tensor_list, emit_list, scope_list):
    """
    copy_input_to_ub_compute
    :param tensor_h1:
    :param tensor_n2:
    :param tensor_i2:
    :param tensor_r2:
    :param tensor_n2_mid:
    :param tensor_d_h2:
    :param tensor_dy:
    :param tensor_list:
    :param emit_list:
    :param scope_list:
    :return:
    """
    tensor_ele_shape = [1, tensor_h1.shape[1].value, tensor_h1.shape[2].value,
                        tensor_h1.shape[3].value, tensor_h1.shape[4].value]
    tensor_h1_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, m: tensor_h1[0, i, j, k, m],
                               name="tensor_h1_ub", tag="tensor_h1_ub")
    tensor_list["tensor_h1_ub_ele"] = tensor_h1_ub
    emit_list["tensor_h1_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_h1_ub_ele"] = tbe_platform.scope_ubuf

    tensor_n2_i2_ub = tvm.compute(tensor_ele_shape,
                                  lambda t, i, j, k, m: tensor_n2[0, i, j, k, m],
                                  name="tensor_n2_i2_ub", tag="tensor_n2_i2_ub")
    tensor_list["tensor_n2_i2_ub_ele"] = tensor_n2_i2_ub
    emit_list["tensor_n2_i2_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_n2_i2_ub_ele"] = tbe_platform.scope_ubuf

    tensor_n2_ub = tvm.compute(tensor_ele_shape,
                               lambda t, i, j, k, m: tensor_n2[0, i, j, k, m],
                               name="tensor_n2_ub", tag="tensor_n2_ub")
    tensor_list["tensor_n2_ub_ele"] = tensor_n2_ub
    emit_list["tensor_n2_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_n2_ub_ele"] = tbe_platform.scope_ubuf

    tensor_i2_ub = tvm.compute(tensor_ele_shape,
                               lambda t, i, j, k, m: tensor_i2[0, i, j, k, m],
                               name="tensor_i2_ub", tag="tensor_i2_ub")
    tensor_list["tensor_i2_ub_ele"] = tensor_i2_ub
    emit_list["tensor_i2_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_i2_ub_ele"] = tbe_platform.scope_ubuf

    tensor_i2_n2i_ub = tvm.compute(tensor_ele_shape,
                                   lambda t, i, j, k, m: tensor_i2[0, i, j, k, m],
                                   name="tensor_i2_n2i_ub", tag="tensor_i2_n2i_ub")
    tensor_list["tensor_i2_n2i_ub_ele"] = tensor_i2_n2i_ub
    emit_list["tensor_i2_n2i_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_i2_n2i_ub_ele"] = tbe_platform.scope_ubuf

    tensor_r2_dn2h_ub = tvm.compute(tensor_ele_shape,
                                    lambda t, i, j, k, m: tensor_r2[0, i, j, k, m],
                                    name="tensor_r2_dn2h_ub", tag="tensor_r2_dn2h_ub")
    tensor_list["tensor_r2_dn2h_ub_ele"] = tensor_r2_dn2h_ub
    emit_list["tensor_r2_dn2h_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_r2_dn2h_ub_ele"] = tbe_platform.scope_ubuf

    tensor_r2_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, m: tensor_r2[0, i, j, k, m],
                               name="tensor_r2_ub", tag="tensor_r2_ub")
    tensor_list["tensor_r2_ub_ele"] = tensor_r2_ub
    emit_list["tensor_r2_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_r2_ub_ele"] = tbe_platform.scope_ubuf

    tensor_n2_mid_ub = tvm.compute(tensor_ele_shape,
                                   lambda t, i, j, k, m: tensor_n2_mid[0, i, j, k, m],
                                   name="tensor_n2_mid_ub", tag="tensor_n2_mid_ub")
    tensor_list["tensor_n2_mid_ub_ele"] = tensor_n2_mid_ub
    emit_list["tensor_n2_mid_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_n2_mid_ub_ele"] = tbe_platform.scope_ubuf

    tensor_d_h2_i2_ub = tvm.compute(tensor_ele_shape,
                                    lambda t, i, j, k, m: tensor_d_h2[0, i, j, k, m],
                                    name="tensor_d_h2_i2_ub", tag="tensor_d_h2_i2_ub")
    tensor_list["tensor_d_h2_i2_ub_ele"] = tensor_d_h2_i2_ub
    emit_list["tensor_d_h2_i2_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_h2_i2_ub_ele"] = tbe_platform.scope_ubuf

    tensor_d_h2_n2i_ub = tvm.compute(tensor_ele_shape,
                                     lambda t, i, j, k, m: tensor_d_h2[0, i, j, k, m],
                                     name="tensor_d_h2_n2i_ub", tag="tensor_d_h2_n2i_ub")
    tensor_list["tensor_d_h2_n2i_ub_ele"] = tensor_d_h2_n2i_ub
    emit_list["tensor_d_h2_n2i_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_h2_n2i_ub_ele"] = tbe_platform.scope_ubuf

    tensor_dy_i2_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, m: tensor_dy[0, i, j, k, m],
                                  name="tensor_dy_i2_ub", tag="tensor_dy_i2_ub")
    tensor_list["tensor_dy_i2_ub_ele"] = tensor_dy_i2_ub
    emit_list["tensor_dy_i2_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_dy_i2_ub_ele"] = tbe_platform.scope_ubuf

    tensor_dy_n2i_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, m: tensor_dy[0, i, j, k, m],
                                   name="tensor_dy_n2i_ub", tag="tensor_dy_n2i_ub")
    tensor_list["tensor_dy_n2i_ub_ele"] = tensor_dy_n2i_ub
    emit_list["tensor_dy_n2i_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_dy_n2i_ub_ele"] = tbe_platform.scope_ubuf

    return tensor_list, emit_list, scope_list


# 'pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda
# 'pylint: disable=too-many-locals,invalid-name,too-many-arguments
def elewise_compute_for_gate(tensor_dh2_di2, tensor_dy_di2, tensor_h1, tensor_n2_di2,
                             tensor_i2_di2, tensor_dh2_dn2i, tensor_dy_dn2i, tensor_n2_dn2i,
                             tensor_i2_dn2i, tensor_r2_dn2h, tensor_n2_mid, tensor_r2_dr2,
                             tensor_list, emit_list, scope_list, dtype, gate_order):
    """
    elewise_compute_for_gate
    :param tensor_dh2_di2:
    :param tensor_dy_di2:
    :param tensor_h1:
    :param tensor_n2_di2:
    :param tensor_i2_di2:
    :param tensor_dh2_dn2i:
    :param tensor_dy_dn2i:
    :param tensor_n2_dn2i:
    :param tensor_i2_dn2i:
    :param tensor_r2_dn2h:
    :param tensor_n2_mid:
    :param tensor_r2_dr2:
    :param tensor_list:
    :param emit_list:
    :param scope_list:
    :param dtype:
    :param gate_order:
    :return:
    """
    vector_shape_list = [1, tensor_dh2_di2.shape[1].value,
                         tensor_dh2_di2.shape[2].value,
                         tensor_dh2_di2.shape[3].value,
                         tensor_dh2_di2.shape[4].value]

    # compute for di2
    tensor_dh2_di2_add = tbe.vadd(tensor_dh2_di2, tensor_dy_di2)
    tensor_list["tensor_dh2_di2_add_ele"] = tensor_dh2_di2_add
    emit_list["tensor_dh2_di2_add_ele"] = insn_cmd.ADD
    scope_list["tensor_dh2_di2_add_ele"] = tbe_platform.scope_ubuf

    tensor_dh1_sub_n2_di2 = tbe.vsub(tensor_h1, tensor_n2_di2)
    tensor_list["tensor_dh1_sub_n2_di2_ele"] = tensor_dh1_sub_n2_di2
    emit_list["tensor_dh1_sub_n2_di2_ele"] = insn_cmd.SUB
    scope_list["tensor_dh1_sub_n2_di2_ele"] = tbe_platform.scope_ubuf

    data_one_di2 = tbe.broadcast(tvm.const(1, dtype), vector_shape_list, dtype)
    tensor_list["data_one_di2_ele"] = data_one_di2
    emit_list["data_one_di2_ele"] = insn_cmd.DUP
    scope_list["data_one_di2_ele"] = tbe_platform.scope_ubuf

    tensor_one_sub_i2_di2 = tbe.vsub(data_one_di2, tensor_i2_di2)
    tensor_list["tensor_one_sub_i2_di2_ele"] = tensor_one_sub_i2_di2
    emit_list["tensor_one_sub_i2_di2_ele"] = insn_cmd.SUB
    scope_list["tensor_one_sub_i2_di2_ele"] = tbe_platform.scope_ubuf

    tensor_res_mul_dh2_di2 = tbe.vmul(tensor_dh1_sub_n2_di2, tensor_dh2_di2_add)
    tensor_list["tensor_res_mul_dh2_di2_ele"] = tensor_res_mul_dh2_di2
    emit_list["tensor_res_mul_dh2_di2_ele"] = insn_cmd.MUL
    scope_list["tensor_res_mul_dh2_di2_ele"] = tbe_platform.scope_ubuf

    tensor_res_mul_one_di2 = tbe.vmul(tensor_res_mul_dh2_di2, tensor_one_sub_i2_di2)
    tensor_list["tensor_res_mul_one_di2_ele"] = tensor_res_mul_one_di2
    emit_list["tensor_res_mul_one_di2_ele"] = insn_cmd.MUL
    scope_list["tensor_res_mul_one_di2_ele"] = tbe_platform.scope_ubuf

    tensor_res_mul_i2_di2 = tbe.vmul(tensor_res_mul_one_di2, tensor_i2_di2)
    tensor_list["tensor_res_mul_i2_di2_ele"] = tensor_res_mul_i2_di2
    emit_list["tensor_res_mul_i2_di2_ele"] = insn_cmd.MUL
    scope_list["tensor_res_mul_i2_di2_ele"] = tbe_platform.scope_ubuf

    # compute for dn2_i
    tensor_power_n2_dn2i = tbe.vmul(tensor_n2_dn2i, tensor_n2_dn2i)
    tensor_list["tensor_power_n2_dn2i_ele"] = tensor_power_n2_dn2i
    emit_list["tensor_power_n2_dn2i_ele"] = insn_cmd.MUL
    scope_list["tensor_power_n2_dn2i_ele"] = tbe_platform.scope_ubuf

    data_one_dn2i = tbe.broadcast(tvm.const(1, dtype), vector_shape_list, dtype)
    tensor_list["data_one_dn2i_ele"] = data_one_dn2i
    emit_list["data_one_dn2i_ele"] = insn_cmd.DUP
    scope_list["data_one_dn2i_ele"] = tbe_platform.scope_ubuf

    tensor_one_sub_n2_dn2i = tbe.vsub(data_one_dn2i, tensor_power_n2_dn2i)
    tensor_list["tensor_one_sub_n2_dn2i_ele"] = tensor_one_sub_n2_dn2i
    emit_list["tensor_one_sub_n2_dn2i_ele"] = insn_cmd.SUB
    scope_list["tensor_one_sub_n2_dn2i_ele"] = tbe_platform.scope_ubuf

    tensor_one_sub_i2_dn2i = tbe.vsub(data_one_dn2i, tensor_i2_dn2i)
    tensor_list["tensor_one_sub_i2_dn2i_ele"] = tensor_one_sub_i2_dn2i
    emit_list["tensor_one_sub_i2_dn2i_ele"] = insn_cmd.SUB
    scope_list["tensor_one_sub_i2_dn2i_ele"] = tbe_platform.scope_ubuf

    tensor_n2_mul_i2_dn2i = tbe.vmul(tensor_one_sub_n2_dn2i, tensor_one_sub_i2_dn2i)
    tensor_list["tensor_n2_mul_i2_dn2i_ele"] = tensor_n2_mul_i2_dn2i
    emit_list["tensor_n2_mul_i2_dn2i_ele"] = insn_cmd.MUL
    scope_list["tensor_n2_mul_i2_dn2i_ele"] = tbe_platform.scope_ubuf
    # dn2i_ele dn2i_add_ele
    tensor_dh2_dn2i_add = tbe.vadd(tensor_dh2_dn2i, tensor_dy_dn2i)
    tensor_list["tensor_dh2_dn2i_add_ele"] = tensor_dh2_dn2i_add
    emit_list["tensor_dh2_dn2i_add_ele"] = insn_cmd.ADD
    scope_list["tensor_dh2_dn2i_add_ele"] = tbe_platform.scope_ubuf

    tensor_dh2_mul_i2_dn2i = tbe.vmul(tensor_n2_mul_i2_dn2i, tensor_dh2_dn2i_add)
    tensor_list["tensor_dh2_mul_i2_dn2i_ele"] = tensor_dh2_mul_i2_dn2i
    emit_list["tensor_dh2_mul_i2_dn2i_ele"] = insn_cmd.MUL
    scope_list["tensor_dh2_mul_i2_dn2i_ele"] = tbe_platform.scope_ubuf

    tensor_dh2_mul_i2_dn2i_gm = tvm.compute(
        vector_shape_list, lambda t, i, j, k, m: tensor_dh2_mul_i2_dn2i[0, i, j, k, m],
        name="tensor_dh2_mul_i2_dn2i_gm", tag="tensor_dh2_mul_i2_dn2i_gm")
    tensor_list["tensor_dh2_mul_i2_dn2i_gm_ele"] = tensor_dh2_mul_i2_dn2i_gm
    emit_list["tensor_dh2_mul_i2_dn2i_gm_ele"] = insn_cmd.DMA_COPY

    tensor_fake_dh2_mul_i2_dn2i_ub = tvm.compute(
        vector_shape_list, lambda t, i, j, k, m: tensor_dh2_mul_i2_dn2i_gm[0, i, j, k, m],
        name="tensor_fake_dh2_mul_i2_dn2i_ub", tag="tensor_fake_dh2_mul_i2_dn2i_ub")
    tensor_list["tensor_fake_dh2_mul_i2_dn2i_ub_ele"] = tensor_fake_dh2_mul_i2_dn2i_ub
    scope_list["tensor_fake_dh2_mul_i2_dn2i_ub_ele"] = tbe_platform.scope_ubuf
    emit_list["tensor_fake_dh2_mul_i2_dn2i_ub_ele"] = insn_cmd.DMA_COPY

    # compute for dn2h
    tensor_dn2i_mul_r2_dn2h = tbe.vmul(tensor_fake_dh2_mul_i2_dn2i_ub, tensor_r2_dn2h)
    tensor_list["tensor_dn2i_mul_r2_dn2h_ele"] = tensor_dn2i_mul_r2_dn2h
    emit_list["tensor_dn2i_mul_r2_dn2h_ele"] = insn_cmd.MUL
    scope_list["tensor_dn2i_mul_r2_dn2h_ele"] = tbe_platform.scope_ubuf

    # compute for dr2
    data_one_dr2 = tbe.broadcast(tvm.const(1, dtype), vector_shape_list, dtype)
    tensor_list["data_one_dr2_ele"] = data_one_dr2
    emit_list["data_one_dr2_ele"] = insn_cmd.DUP
    scope_list["data_one_dr2_ele"] = tbe_platform.scope_ubuf

    data_one_sub_r2_dr2 = tbe.vsub(data_one_dr2, tensor_r2_dr2)
    tensor_list["data_one_sub_r2_dr2_ele"] = data_one_sub_r2_dr2
    emit_list["data_one_sub_r2_dr2_ele"] = insn_cmd.SUB
    scope_list["data_one_sub_r2_dr2_ele"] = tbe_platform.scope_ubuf

    data_n2_mid_mul_r2_dr2 = tbe.vmul(data_one_sub_r2_dr2, tensor_n2_mid)
    tensor_list["data_n2_mid_mul_r2_dr2_ele"] = data_n2_mid_mul_r2_dr2
    emit_list["data_n2_mid_mul_r2_dr2_ele"] = insn_cmd.MUL
    scope_list["data_n2_mid_mul_r2_dr2_ele"] = tbe_platform.scope_ubuf

    data_dn2h_mul_r2_dr2 = tbe.vmul(data_n2_mid_mul_r2_dr2, tensor_dn2i_mul_r2_dn2h)
    tensor_list["data_dn2h_mul_r2_dr2_ele"] = data_dn2h_mul_r2_dr2
    emit_list["data_dn2h_mul_r2_dr2_ele"] = insn_cmd.MUL
    scope_list["data_dn2h_mul_r2_dr2_ele"] = tbe_platform.scope_ubuf
    if gate_order == "zrh":
        tensors = [tensor_list["tensor_res_mul_i2_di2_ele"],
                   tensor_list["data_dn2h_mul_r2_dr2_ele"],
                   tensor_list["tensor_dn2i_mul_r2_dn2h_ele"]]
    else:
        tensors = [tensor_list["data_dn2h_mul_r2_dr2_ele"],
                   tensor_list["tensor_res_mul_i2_di2_ele"],
                   tensor_list["tensor_dn2i_mul_r2_dn2h_ele"]]
    tensor_d_gate = tbe.concat(tensors, 1)
    tensor_list["tensor_d_gate"] = tensor_d_gate

    return tensor_list, emit_list, scope_list


# 'pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda
# 'pylint: disable=too-many-locals,invalid-name,too-many-arguments
def matmul_compute(tensor_weight, tensor_d_h2, tensor_dy, tensor_i2, tensor_list, dst_type):
    """
    matmul_compute
    :param tensor_weight:
    :param tensor_d_h2:
    :param tensor_dy:
    :param tensor_i2:
    :param tensor_list:
    :param dst_type:
    :return:
    """
    tensor_d_gate = tensor_list["tensor_d_gate"]
    if dst_type == "float32":
        tensor_d_gate_ub = tvm.compute(tensor_d_gate.shape,
                                       lambda *i: tensor_d_gate(*i), name="tensor_d_gate_ub")
        tensor_list["tensor_d_gate_ub"] = tensor_d_gate_ub
        tensor_d_gate_cast = tbe.cast_to(tensor_d_gate_ub, "float16")
        tensor_list["tensor_d_gate_cast"] = tensor_d_gate_cast
        tensor_d_gate = tensor_d_gate_cast
    para_dict = {
        "trans_a": True,
        "trans_b": False,
        "format_a": "FRACTAL_NZ",
        "format_b": "FRACTAL_NZ",
        "dst_dtype": dst_type
        }
    matmul_res_gm = tbe.gemm(tensor_d_gate, tensor_weight, para_dict)

    matmul_res = matmul_res_gm.op.input_tensors[0]
    tensor_list["matmul_res"] = matmul_res
    # compute for dh2 * i2
    tensor_ele_shape = [1, tensor_d_h2.shape[1].value,
                        tensor_d_h2.shape[2].value,
                        tensor_d_h2.shape[3].value,
                        tensor_d_h2.shape[4].value]

    tensor_d_h2_res = tvm.compute(tensor_ele_shape,
                                  lambda t, i, j, k, m: tensor_d_h2[0, i, j, k, m],
                                  name="tensor_d_h2_res", tag="tensor_d_h2_res")
    tensor_list["tensor_d_h2_res"] = tensor_d_h2_res

    tensor_dy_res = tvm.compute(tensor_ele_shape, lambda t, i, j, k, m: tensor_dy[0, i, j, k, m],
                                name="tensor_dy_res", tag="tensor_dy_res")
    tensor_list["tensor_dy_res"] = tensor_dy_res
    tensor_i2_res = tvm.compute(tensor_ele_shape, lambda t, i, j, k, m: tensor_i2[0, i, j, k, m],
                                name="tensor_i2_res", tag="tensor_i2_res")
    tensor_list["tensor_i2_res"] = tensor_i2_res

    tensor_res_dh2_add = tbe.vadd(tensor_d_h2_res, tensor_dy_res)
    tensor_list["tensor_res_dh2_add"] = tensor_res_dh2_add

    tensor_res_dh2_mul_i2 = tbe.vmul(tensor_res_dh2_add, tensor_i2_res)
    tensor_list["tensor_res_dh2_mul_i2"] = tensor_res_dh2_mul_i2

    tensor_dh1_res = tbe.vadd(matmul_res, tensor_res_dh2_mul_i2)
    tensor_list["tensor_dh1_res"] = tensor_dh1_res

    dh1_shape = tensor_ele_shape
    tensor_dh1_gm = tvm.compute(dh1_shape, lambda t, i, j, k, m: tensor_dh1_res[0, i, j, k, m],
                                name="tensor_dh1_gm", tag="tensor_dh1_gm")
    tensor_list["tensor_dh1_gm"] = tensor_dh1_gm

    return tensor_list


# 'pylint: disable=too-many-arguments
def basic_gru_cell_compute(tensor_weight, tensor_h1, tensor_n2, tensor_i2,
                           tensor_r2, tensor_n2_mid,
                           tensor_d_h2, tensor_dy, cast_type, gate_order):
    """
    basic_gru_cell_compute
    :param tensor_weight:
    :param tensor_h1:
    :param tensor_n2:
    :param tensor_i2:
    :param tensor_r2:
    :param tensor_n2_mid:
    :param tensor_d_h2:
    :param tensor_dy:
    :param cast_type:
    :param gate_order:
    :return:
    """
    tensor_list = {}
    emit_list = {}
    scope_list = {}
    tensor_list, emit_list, scope_list = \
        copy_input_to_ub_compute(tensor_h1, tensor_n2,
                                 tensor_i2, tensor_r2, tensor_n2_mid,
                                 tensor_d_h2, tensor_dy,
                                 tensor_list, emit_list, scope_list)
    dtype = "float16"
    if cast_type:
        dtype = "float32"
    tensor_list, emit_list, scope_list = \
        elewise_compute_for_gate(tensor_list["tensor_d_h2_i2_ub_ele"],
                                 tensor_list["tensor_dy_i2_ub_ele"],
                                 tensor_list["tensor_h1_ub_ele"],
                                 tensor_list["tensor_n2_i2_ub_ele"],
                                 tensor_list["tensor_i2_ub_ele"],
                                 tensor_list["tensor_d_h2_n2i_ub_ele"],
                                 tensor_list["tensor_dy_n2i_ub_ele"],
                                 tensor_list["tensor_n2_ub_ele"],
                                 tensor_list["tensor_i2_n2i_ub_ele"],
                                 tensor_list["tensor_r2_dn2h_ub_ele"],
                                 tensor_list["tensor_n2_mid_ub_ele"],
                                 tensor_list["tensor_r2_ub_ele"],
                                 tensor_list, emit_list, scope_list, dtype, gate_order)

    dst_type = "float16"
    if cast_type:
        dst_type = "float32"
    tensor_list = matmul_compute(tensor_weight, tensor_d_h2, tensor_dy, tensor_i2,
                                 tensor_list, dst_type)

    return tensor_list, emit_list, scope_list


def get_tiling(output_size):
    """
    get tiling
    :param output_size:
    :return:
    """
    # limit by L0A/L0B size
    max_output_factor = 6
    tiling_info = [1, min(max_output_factor, output_size)]

    return tiling_info


# 'pylint: disable=too-many-locals,too-many-statements
def schedule_for_cell(tensor_list, emit_list, scope_list, cast_type):
    """
    schedule_for_cell
    :param tensor_list:
    :param emit_list:
    :param scope_list:
    :param cast_type:
    :return:
    """
    tensor_dh1_gm = tensor_list["tensor_dh1_gm"]

    sch = tvm.create_schedule(tensor_dh1_gm.op)
    # for fake compute

    tensor_d_gate = tensor_list["tensor_d_gate"]
    output_size = tensor_d_gate.shape[1].value // 3
    tiling_info = get_tiling(output_size)
    batch_factor = tiling_info[0]
    output_factor = tiling_info[1]
    o_outer, o_inner = sch[tensor_d_gate].split(tensor_d_gate.op.axis[1], factor=output_factor*3)
    n_outer, n_inner = sch[tensor_d_gate].split(tensor_d_gate.op.axis[2], factor=batch_factor)
    sch[tensor_d_gate].reorder(tensor_d_gate.op.axis[0],
                               o_outer, n_outer, o_inner, n_inner,
                               tensor_d_gate.op.axis[3],
                               tensor_d_gate.op.axis[4])
    concat_compute_axis = n_outer
    for key in tensor_list.keys():
        if scope_list.__contains__(key):
            sch[tensor_list[key]].set_scope(scope_list[key])
        if emit_list.__contains__(key):
            sch[tensor_list[key]].emit_insn(sch[tensor_list[key]].op.axis[1], emit_list[key])
            sch[tensor_list[key]].compute_at(sch[tensor_d_gate], concat_compute_axis)

    sch[tensor_d_gate].emit_insn(o_inner, insn_cmd.DMA_COPY)

    if cast_type:
        tensor_d_gate_ub = tensor_list["tensor_d_gate_ub"]
        n_ub_outer, _ = sch[tensor_d_gate_ub].split(tensor_d_gate_ub.op.axis[2],
                                                    factor=batch_factor)
        sch[tensor_d_gate_ub].set_scope(tbe_platform.scope_ubuf)
        sch[tensor_d_gate_ub].emit_insn(n_ub_outer, insn_cmd.DMA_COPY)
        sch[tensor_list["tensor_d_gate_cast"]].set_scope(tbe_platform.scope_ubuf)
        sch[tensor_list["tensor_d_gate_cast"]].emit_insn(
            tensor_list["tensor_d_gate_cast"].op.axis[1], insn_cmd.CAST)

    # matmul_res
    matmul_res = tensor_list["matmul_res"]
    tensor_d_h2_res = tensor_list["tensor_d_h2_res"]
    tensor_dy_res = tensor_list["tensor_dy_res"]
    tensor_i2_res = tensor_list["tensor_i2_res"]
    tensor_res_dh2_add = tensor_list["tensor_res_dh2_add"]
    tensor_res_dh2_mul_i2 = tensor_list["tensor_res_dh2_mul_i2"]
    tensor_dh1_res = tensor_list["tensor_dh1_res"]

    tensor_c = matmul_res.op.input_tensors[0]
    tensor_a_l0 = tensor_c.op.input_tensors[0]
    tensor_b_l0 = tensor_c.op.input_tensors[1]
    tensor_a_l1 = tensor_a_l0.op.input_tensors[0]
    tensor_b_l1 = tensor_b_l0.op.input_tensors[0]

    sch[tensor_a_l1].set_scope(tbe_platform.scope_cbuf)
    sch[tensor_b_l1].set_scope(tbe_platform.scope_cbuf)
    sch[tensor_a_l0].set_scope(tbe_platform.scope_ca)
    sch[tensor_b_l0].set_scope(tbe_platform.scope_cb)
    sch[tensor_c].set_scope(tbe_platform.scope_cc)

    sch[tensor_d_h2_res].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_dy_res].set_scope(tbe_platform.scope_ubuf)

    n_outer, n_inner = sch[tensor_c].split(tensor_c.op.axis[1],
                                           factor=output_factor)

    m_outer, m_inner = sch[tensor_c].split(tensor_c.op.axis[2],
                                           factor=batch_factor)

    kb_outer, kb_inner = sch[tensor_c].split(tensor_c.op.reduce_axis[0],
                                             factor=output_factor*3)

    n_inner_outer, n_inner_inner = sch[tensor_c].split(n_inner,
                                                       factor=output_factor)
    m_inner_outer, m_inner_inner = sch[tensor_c].split(m_inner,
                                                       factor=batch_factor)

    kb_inner_outer, kb_inner_inner = sch[tensor_c].split(kb_inner,
                                                         factor=output_factor*3)

    sch[tensor_c].reorder(tensor_c.op.axis[0], n_outer, m_outer, kb_outer,
                          n_inner_outer, m_inner_outer, kb_inner_outer,
                          n_inner_inner, m_inner_inner, tensor_c.op.axis[3],
                          tensor_c.op.axis[4], kb_inner_inner,
                          tensor_c.op.reduce_axis[1])
    # wanglinmu
    compute_axis = kb_inner_outer
    sch[tensor_d_gate].compute_at(sch[tensor_c], compute_axis)
    if cast_type:
        sch[tensor_list["tensor_d_gate_ub"]].compute_at(sch[tensor_c], compute_axis)
        sch[tensor_list["tensor_d_gate_cast"]].compute_at(sch[tensor_c], compute_axis)

    sch[tensor_a_l1].compute_at(sch[tensor_c], compute_axis)
    sch[tensor_b_l1].compute_at(sch[tensor_c], compute_axis)
    sch[tensor_a_l0].compute_at(sch[tensor_c], compute_axis)
    sch[tensor_b_l0].compute_at(sch[tensor_c], compute_axis)

    res_o_outer, res_o_inner = sch[tensor_dh1_gm].split(tensor_dh1_gm.op.axis[1],
                                                        factor=output_factor)
    res_n_outer, res_n_inner = sch[tensor_dh1_gm].split(tensor_dh1_gm.op.axis[2],
                                                        factor=batch_factor)
    sch[tensor_dh1_gm].reorder(tensor_dh1_gm.op.axis[0],
                               res_o_outer, res_n_outer, res_o_inner, res_n_inner,
                               tensor_dh1_gm.op.axis[3],
                               tensor_dh1_gm.op.axis[4])

    sch[tensor_dh1_gm].emit_insn(res_o_inner, insn_cmd.DMA_COPY)
    res_compute_axis = res_n_outer

    sch[tensor_list["tensor_d_h2_res"]].compute_at(sch[tensor_dh1_gm], res_compute_axis)
    sch[tensor_list["tensor_dy_res"]].compute_at(sch[tensor_dh1_gm], res_compute_axis)
    sch[tensor_list["tensor_i2_res"]].compute_at(sch[tensor_dh1_gm], res_compute_axis)
    sch[tensor_list["tensor_res_dh2_add"]].compute_at(sch[tensor_dh1_gm], res_compute_axis)
    sch[tensor_list["tensor_res_dh2_mul_i2"]].compute_at(sch[tensor_dh1_gm], res_compute_axis)
    sch[tensor_list["tensor_dh1_res"]].compute_at(sch[tensor_dh1_gm], res_compute_axis)
    sch[matmul_res].compute_at(sch[tensor_dh1_gm], res_compute_axis)
    sch[tensor_c].compute_at(sch[tensor_dh1_gm], res_compute_axis)

    mad_pattern = tbe_platform.cce_params.GEMM_MODE
    mad_dict = {"mad_pattern": mad_pattern,
                "k_outer": [kb_outer, kb_inner_outer],
                }

    sch[tensor_c].emit_insn(n_inner_inner, 'mad', mad_dict)
    sch[tensor_d_h2_res].emit_insn(tensor_d_h2_res.op.axis[1], insn_cmd.DMA_COPY)
    sch[tensor_dy_res].emit_insn(tensor_dy_res.op.axis[1], insn_cmd.DMA_COPY)

    sch[tensor_i2_res].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_i2_res].emit_insn(tensor_i2_res.op.axis[1], insn_cmd.DMA_COPY)

    sch[tensor_res_dh2_add].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_res_dh2_add].emit_insn(tensor_res_dh2_add.op.axis[1], insn_cmd.ADD)
    sch[tensor_res_dh2_mul_i2].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_res_dh2_mul_i2].emit_insn(tensor_res_dh2_mul_i2.op.axis[1],
                                         insn_cmd.MUL)
    sch[tensor_dh1_res].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_dh1_res].emit_insn(tensor_dh1_res.op.axis[1], insn_cmd.ADD)

    sch[tensor_a_l1].emit_insn(tensor_a_l1.op.axis[1], insn_cmd.DMA_COPY)
    sch[tensor_b_l1].emit_insn(tensor_b_l1.op.axis[1], insn_cmd.DMA_COPY)
    sch[tensor_a_l0].emit_insn(tensor_a_l0.op.axis[1], insn_cmd.DMA_COPY)
    sch[tensor_b_l0].emit_insn(tensor_b_l0.op.axis[1], insn_cmd.DMA_COPY)
    sch[matmul_res].set_scope(tbe_platform.scope_ubuf)
    sch[matmul_res].emit_insn(matmul_res.op.axis[1], insn_cmd.DMA_COPY)
    return sch


# 'pylint: disable=too-many-locals
def do_cell_process(input_list, is_extent_buffer):
    """
    do_cell_process
    :param input_list:
    :param is_extent_buffer:
    :return:
    """
    # there compute for cell. And then, use tik to do the compute
    tensor_weight = input_list[0]
    tensor_h1 = input_list[1]
    tensor_n2 = input_list[2]
    tensor_i2 = input_list[3]
    tensor_r2 = input_list[4]
    tensor_n2_mid = input_list[5]
    tensor_d_h2 = input_list[6]
    tensor_dy = input_list[7]
    cast_type = is_extent_buffer[0]
    gate_order = is_extent_buffer[1]

    tensor_list, emit_list, scope_list = \
        basic_gru_cell_compute(tensor_weight, tensor_h1, tensor_n2, tensor_i2,
                               tensor_r2, tensor_n2_mid, tensor_d_h2, tensor_dy,
                               cast_type, gate_order)

    tensor_output_list = [tensor_list["tensor_dh1_gm"], tensor_list["tensor_d_gate"],
                          tensor_list["tensor_dh2_mul_i2_dn2i_gm_ele"]]

    sch = schedule_for_cell(tensor_list, emit_list, scope_list, cast_type)

    return tensor_output_list, sch


# 'pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop
# 'pylint: disable=unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def gru_v2_hidden_grad(weight_hidden, init_h, h, dy, dh, update, reset, new,
                       hidden_new, dh_prev, dgate_h, dnt_x, gate_order="zrh",
                       kernel_name="gru_hidden_grad"):
    """
    gru_v2_hidden_grad
    :param weight_hidden:
    :param init_h:
    :param h:
    :param dy:
    :param dh:
    :param update:
    :param reset:
    :param new:
    :param hidden_new:
    :param dh_prev:
    :param dgate_h:
    :param dnt_x:
    :param gate_order:
    :param kernel_name:
    :return:
    """
    dh1 = dh_prev
    dn_i = dnt_x
    tik_instance = tik.Tik(tik.Dprofile())
    weight_src_dtype = weight_hidden["dtype"].lower()
    inp_src_dtype = init_h["dtype"].lower()

    w_shape = [1, weight_hidden["shape"][0], weight_hidden["shape"][1],
               weight_hidden["shape"][2], weight_hidden["shape"][3]]
    tensor_w = tik_instance.Tensor(weight_src_dtype, w_shape, tbe_platform.scope_gm, "tensor_w")

    init_h_shape = [1, init_h["shape"][0], init_h["shape"][1],
                    init_h["shape"][2], init_h["shape"][3]]
    tensor_init_h = tik_instance.Tensor(inp_src_dtype, init_h_shape, tbe_platform.scope_gm, "tensor_init_h")

    h_shape = [h["shape"][0], h["shape"][1], h["shape"][2], h["shape"][3], h["shape"][4]]
    tensor_h = tik_instance.Tensor(inp_src_dtype, h_shape, tbe_platform.scope_gm, "tensor_h")

    dy_shape = [dy["shape"][0], dy["shape"][1], dy["shape"][2], dy["shape"][3], dy["shape"][4]]
    tensor_dy = tik_instance.Tensor(inp_src_dtype, dy_shape, tbe_platform.scope_gm, "tensor_dy")

    dh_shape = [1, dh["shape"][0], dh["shape"][1], dh["shape"][2], dh["shape"][3]]
    tensor_dh = tik_instance.Tensor(inp_src_dtype, dh_shape, tbe_platform.scope_gm, "tensor_dh")

    update_shape = [update["shape"][0], update["shape"][1], update["shape"][2],
                    update["shape"][3], update["shape"][4]]
    tensor_update = tik_instance.Tensor(inp_src_dtype, update_shape, tbe_platform.scope_gm, "tensor_update")

    reset_shape = [reset["shape"][0], reset["shape"][1], reset["shape"][2],
                   reset["shape"][3], reset["shape"][4]]
    tensor_reset = tik_instance.Tensor(inp_src_dtype, reset_shape, tbe_platform.scope_gm, "tensor_reset")

    new_shape = [new["shape"][0], new["shape"][1], new["shape"][2],
                 new["shape"][3], new["shape"][4]]
    tensor_new = tik_instance.Tensor(inp_src_dtype, new_shape, tbe_platform.scope_gm, "tensor_new")

    hidden_new_shape = [hidden_new["shape"][0], hidden_new["shape"][1],
                        hidden_new["shape"][2], hidden_new["shape"][3], hidden_new["shape"][4]]
    tensor_hidden_new = tik_instance.Tensor(inp_src_dtype, hidden_new_shape,
                                            tbe_platform.scope_gm, "tensor_hidden_new")

    # output
    dh1_shape = [1, dh1["shape"][0], dh1["shape"][1], dh1["shape"][2], dh1["shape"][3]]
    tensor_dh1 = tik_instance.Tensor(inp_src_dtype, dh1_shape, tbe_platform.scope_gm, "tensor_dh1")

    dgate_h_shape = [dgate_h["shape"][0], dgate_h["shape"][1], dgate_h["shape"][2],
                     dgate_h["shape"][3], dgate_h["shape"][4]]
    tensor_dgate_h = tik_instance.Tensor(inp_src_dtype, dgate_h_shape, tbe_platform.scope_gm, "tensor_dgate_h")

    dn_i_shape = [dn_i["shape"][0], dn_i["shape"][1], dn_i["shape"][2], dn_i["shape"][3],
                  dn_i["shape"][4]]
    tensor_dn_i = tik_instance.Tensor(inp_src_dtype, dn_i_shape, tbe_platform.scope_gm, "tensor_dn_i")

    temp_output_dh1_shape = [dgate_h["shape"][0], dh1["shape"][0], dh1["shape"][1],
                             dh1["shape"][2], dh1["shape"][3]]
    temp_output_dh1 = tik_instance.Tensor(inp_src_dtype, temp_output_dh1_shape, tbe_platform.scope_gm,
                                          "temp_output_dh1", is_workspace=True)

    t_num = dn_i["shape"][0]

    n_num = h["shape"][2]

    start_num = 0
    stop_num = t_num

    cast_type = False
    if init_h["dtype"] == "float32":
        cast_type = True

    with tik_instance.for_range(start_num, stop_num) as cur_t:
        with tik_instance.for_range(0, n_num, block_num=n_num) as batch:
            i = t_num - cur_t - 1
            tensor_w_cur = tensor_w
            tensor_n2_cur = tensor_new[i: i + 1, :, batch:batch + 1, :, :]
            tensor_i2_cur = tensor_update[i: i + 1, :, batch:batch + 1, :, :]
            tensor_r2_cur = tensor_reset[i: i + 1, :, batch:batch + 1, :, :]
            tensor_n2mid_cur = tensor_hidden_new[i: i + 1, :, batch:batch + 1, :, :]

            tensor_dy_cur = tensor_dy[i: i + 1, :, batch:batch + 1, :, :]

            # cur output
            tensor_dh1_cur = tensor_dh1[0, :, batch:batch + 1, :, :]
            tensor_dgate_h_cur = tensor_dgate_h[i: i + 1, :, batch:batch + 1, :, :]
            tensor_dni_cur = tensor_dn_i[i: i + 1, :, batch:batch + 1, :, :]

            temp_output_dh1_cur = temp_output_dh1[cur_t + 1: cur_t + 2, :, batch:batch + 1, :, :]

            temp_input_dh1_cur = temp_output_dh1[cur_t: cur_t + 1, :, batch:batch + 1, :, :]

            # tensor_h1 tensor_n2 tensor_i2 tensor_r2 tensor_n2_mid tensor_d_h2 tensor_dy
            with tik_instance.if_scope(cur_t == 0):
                with tik_instance.if_scope(cur_t == t_num - 1):
                    tik_instance.call_module(
                        do_cell_process,
                        [tensor_w_cur,
                         tensor_init_h[0, :, batch:batch + 1, :, :],
                         tensor_n2_cur, tensor_i2_cur, tensor_r2_cur, tensor_n2mid_cur,
                         tensor_dh[0, :, batch:batch + 1, :, :],
                         tensor_dy_cur],
                        [tensor_dh1_cur,
                         tensor_dgate_h_cur,
                         tensor_dni_cur],
                        [cast_type, gate_order])
                with tik_instance.else_scope():
                    tik_instance.call_module(
                        do_cell_process,
                        [tensor_w_cur,
                         tensor_h[i - 1: i, :, batch:batch + 1, :, :],
                         tensor_n2_cur, tensor_i2_cur, tensor_r2_cur, tensor_n2mid_cur,
                         tensor_dh[0, :, batch:batch + 1, :, :],
                         tensor_dy_cur],
                        [temp_output_dh1_cur,
                         tensor_dgate_h_cur,
                         tensor_dni_cur],
                        [cast_type, gate_order])
            with tik_instance.else_scope():
                with tik_instance.if_scope(cur_t == t_num - 1):
                    tik_instance.call_module(
                        do_cell_process,
                        [tensor_w_cur,
                         tensor_init_h[0, :, batch:batch + 1, :, :],
                         tensor_n2_cur, tensor_i2_cur, tensor_r2_cur, tensor_n2mid_cur,
                         temp_input_dh1_cur,
                         tensor_dy_cur],
                        [tensor_dh1_cur,
                         tensor_dgate_h_cur,
                         tensor_dni_cur],
                        [cast_type, gate_order])
                with tik_instance.else_scope():
                    tik_instance.call_module(
                        do_cell_process,
                        [tensor_w_cur,
                         tensor_h[i - 1: i, :, batch:batch + 1, :, :],
                         tensor_n2_cur, tensor_i2_cur, tensor_r2_cur, tensor_n2mid_cur,
                         temp_input_dh1_cur,
                         tensor_dy_cur],
                        [temp_output_dh1_cur,
                         tensor_dgate_h_cur,
                         tensor_dni_cur],
                        [cast_type, gate_order])

    config_map = {
        "dump_cce_code": False,
    }

    input_list = [tensor_w, tensor_init_h, tensor_h, tensor_dy, tensor_dh, tensor_update,
                  tensor_reset, tensor_new, tensor_hidden_new]
    output_list = [tensor_dh1, tensor_dgate_h, tensor_dn_i]
    tik_instance.BuildCCE(kernel_name,
                          input_list,
                          output_list,
                          config=config_map)
