#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
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
batch_matmul_fixpipe
"""
from itertools import product

from tbe import tvm
from tbe.common.platform import platform_info
from tbe.common.utils.const import ComputeFlow
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_te_var

from impl.dynamic.batch_matmul_v2 import get_ori_batch_shape, get_input_shape
from impl.dynamic.quant_batch_matmul import QuantBmmCompute
from impl.util.platform_adapter import error_manager_vector as err_mgr
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_gemm import _get_shape_info


QUANT_PRE_HIGH_BITS_OFFSET = 0x100000000
QUANT_PRE_LOW_BITS_SIGN_OFFSET = 0x80000000
BLOCK_SIZE = 16


def _binary_constant_process(bmm_fixp_inputs):
    x1, x2, quant_pre, bias, y, adj_x1, adj_x2 = bmm_fixp_inputs

    add_compile_info("binary_constant_flag", True)
    context = tbe_context.op_context.get_context()
    context.set_op_mode("dynamic")
    context.add_addition("binary_constant_type", "batch_matmul_fixpipe")
    context.add_addition("is_binary_constant", 1)

    inputs = [
        {
            "shape": x1.get("shape"),
            "ori_shape": x1.get("ori_shape"),
            "dtype": x1.get("dtype"),
            "format": x1.get("format")
        },
        {
            "shape": x2.get("shape"),
            "ori_shape": x2.get("ori_shape"),
            "dtype": x2.get("dtype"),
            "format": x2.get("format")
        },
        {
            "shape": quant_pre.get("shape"),
            "ori_shape": quant_pre.get("ori_shape"),
            "dtype": quant_pre.get("dtype"),
            "format": quant_pre.get("format")
        },
    ]
    if bias is not None:
        inputs.append({
            "shape": bias.get("shape"),
            "ori_shape": bias.get("ori_shape"),
            "dtype": bias.get("dtype"),
            "format": bias.get("format")
        })
        bias["range"], bias["shape"], bias["ori_shape"] = _get_shape_info(bias)

    outputs = [{"shape": y.get("shape"), "ori_shape": y.get("ori_shape"),
                "dtype": y.get("dtype"), "format": y.get("format")}]

    attrs = ({"name": "adj_x1", "dtype": "bool", "value": adj_x1},
             {"name": "adj_x2", "dtype": "bool", "value": adj_x2})

    context.add_addition("op_tiling_params", (inputs, outputs, attrs))
    x1["range"], x1["shape"], x1["ori_shape"] = _get_shape_info(x1)
    x2["range"], x2["shape"], x2["ori_shape"] = _get_shape_info(x2)
    y["range"], y["shape"], y["ori_shape"] = _get_shape_info(y)

    return (x1, x2, quant_pre, bias, y)


def _create_bias_placeholder(bias, unaligned_flag):
    if bias is None:
        return None

    n_ori = get_te_var("n_ori").get_tvm_var()
    n = get_te_var("n").get_tvm_var()
    block_n0 = int(platform_info.get_soc_spec("cube_n_size"))
    if unaligned_flag:
        bias_shape = [n_ori]
    else:
        bias_shape = [n * block_n0]
    tensor_bias = tvm.placeholder(bias_shape,
                                  name="bias",
                                  dtype=bias.get("dtype"),
                                  attrs={
                                      'ori_shape': bias_shape,
                                      'format': bias.get("format")
                                  })

    return tensor_bias


def _create_quant_pre_tensor(quant_pre, pre_conv_mode, binary_constant_flag):
    n = get_te_var("n").get_tvm_var()
    block_n0 = int(platform_info.get_soc_spec("cube_n_size"))

    if pre_conv_mode == "F322B8":
        quant_pre_ori_shape = (1, 1, 1, 1)
        quant_pre_shape = (1, 1, 1, 1, block_n0)
        quant_pre_value = 0

        tensor_deq_scale = tvm.placeholder(quant_pre_shape,
                                           name="quant_pre",
                                           dtype=quant_pre.get("dtype"),
                                           attrs={
                                               "ori_shape": quant_pre_ori_shape,
                                               "const_value": [quant_pre_value, *(0, ) * (block_n0 - 1)]
                                           })
    else:
        if binary_constant_flag:
            add_compile_info("vector_pre_conv_mode", True)

        n_ori = get_te_var("n_ori").get_tvm_var()
        n = get_te_var("n").get_tvm_var()
        quant_pre_ori_shape = (1, 1, 1, n_ori)
        quant_pre_shape = (1, n, 1, 1, block_n0)
        tensor_deq_scale = tvm.placeholder(quant_pre_shape,
                                           name="quant_pre",
                                           dtype=quant_pre.get("dtype"),
                                           attrs={'ori_shape': quant_pre_ori_shape})
    return tensor_deq_scale


def _is_dynamic(tensor):
    return any(dim < 0 for dim in tensor.get("shape", []))


def _get_input_shape(bmm_fixp_inputs, kernel_name, unaligned_flag, binary_constant_flag):
    x1, x2, _, bias, y, adj_x1, adj_x2 = bmm_fixp_inputs
    if binary_constant_flag:
        x1, x2, _, bias, y = _binary_constant_process(bmm_fixp_inputs)

    extra_params = {"op_type": "BatchMatMulV2"}
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if support_l0c2out and x1.get("format") == "ND" and x2.get("format") == "ND":
        extra_params["nd2nz_type"] = ComputeFlow.on_the_fly.value
    res = get_input_shape(x1, x2, bias, y, adj_x1, adj_x2, kernel_name, extra_params, unaligned_flag)

    return res


def _get_build_option(binary_constant_flag, bmm_fixp_inputs):
    unaligned_flag_choice = [1]
    pre_conv_mode_choice = ["VF322B8", "F322B8"]

    if binary_constant_flag:
        _, x2, quant_pre, _, _, _, adj_x2 = bmm_fixp_inputs
        n_dim = x2.get("ori_shape")[-2] if adj_x2 else x2.get("ori_shape")[-1]
        pre_conv_mode_choice = ["VF322B8"] if quant_pre.get("ori_shape")[-1] == n_dim else ["F322B8"]

    return (unaligned_flag_choice, pre_conv_mode_choice)


def batch_matmul_fixpipe_impl(x1, x2, quant_pre, bias=None, y=None, adj_x1=False, adj_x2=False,
                              kernel_name="batch_matmul_fixpipe"):
    bmm_fixp_inputs = [x1, x2, quant_pre, bias, y, adj_x1, adj_x2]
    binary_constant_flag = not _is_dynamic(x1) and not _is_dynamic(x2)
    unaligned_flag_choice, pre_conv_mode_choice = _get_build_option(binary_constant_flag, bmm_fixp_inputs)

    tensor_lists = []
    schs = []

    for unaligned_flag, pre_conv_mode in product(unaligned_flag_choice, pre_conv_mode_choice):
        with tbe.compute():
            res = _get_input_shape(bmm_fixp_inputs, kernel_name, 1, binary_constant_flag)
            shape_x1, shape_x2, is_cache_tiling, nd2nz_type, input_range = res

            tensor_x1 = tvm.placeholder(shape_x1, name="tensor_a", dtype=x1.get("dtype").lower(),
                                        attrs={"ori_shape": x1.get("ori_shape"), "format": x1.get("format")})
            tensor_x2 = tvm.placeholder(shape_x2, name="tensor_b", dtype=x1.get("dtype").lower(),
                                        attrs={"ori_shape": x1.get("ori_shape"), "format": x1.get("format")})

            ori_batch_x1, ori_batch_x2 = get_ori_batch_shape(x1, x2)
            tensor_x1.op.attrs["ori_batch_shape"] = ori_batch_x1
            tensor_x2.op.attrs["ori_batch_shape"] = ori_batch_x2
            tensor_bias = _create_bias_placeholder(bias, unaligned_flag)
            tensor_quant_pre = _create_quant_pre_tensor(quant_pre, pre_conv_mode, binary_constant_flag)
            tensor_list = [tensor_x1, tensor_x2, tensor_quant_pre]
            if tensor_bias is not None:
                tensor_list.append(tensor_bias)

            para_dict = {
                "format_a": x1.get("format"), "format_b": x2.get("format"), "format_out": y.get("format"),
                "trans_a": adj_x1, "trans_b": adj_x2, "dst_dtype": y.get("dtype").lower(),
                "tensor_c": tensor_bias,
                "cache_tiling_flag": is_cache_tiling,
                "unaligned_flag": unaligned_flag,
                "pre_conv_mode": pre_conv_mode,
                "kernel_name": kernel_name,
                "input_range": input_range,
                "nd2nz_type": nd2nz_type
            }

            out = QuantBmmCompute(tensor_x1, tensor_x2, tensor_quant_pre, para_dict).compute()

        with tvm.target.cce():
            sch = tbe.auto_schedule(out)

        tensor_list.append(out)
        tensor_lists.append(tensor_list)
        schs.append(sch)

    if binary_constant_flag:
        context = tbe_context.op_context.get_context()
        context.set_op_mode("static")

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensor_lists, "double_buffer_non_reuse": True,
              "build_args": {"constant_realize_extent_in_infer_bound": False,
                             "enable_db_fold":True,
                             "InjectSync":{"sync_opt_for_notail_db": 1,
                                           "sync_opt_for_preload_loop_zero": True}}}
    tbe.build(schs, config)


@tbe_register.register_param_generalization("BatchMatmulFixpipe")
def batch_matmul_fixpipe_generalization(x1,
                                        x2,
                                        quant_pre,
                                        bias=None,
                                        y=None,
                                        adj_x1=False,
                                        adj_x2=False,
                                        kernel_name="batch_matmul_fixpipe",
                                        generalize_config=None):
    result = []
    if generalize_config.get("mode") == "all_shape":
        x1.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        x2.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        quant_pre.update({"ori_shape": [-2], "ori_format": "NHWC", "shape": [-2], "format": "NC1HWC0"})
        y.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        if bias is not None:
            bias.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        result.append([x1, x2, quant_pre, bias, y, adj_x1, adj_x2])

    return result


@register_operator("BatchMatmulFixpipe")
def batch_matmul_fixpipe(x1,
                         x2,
                         quant_pre,
                         bias=None,
                         y=None,
                         adj_x1=False,
                         adj_x2=False,
                         kernel_name="batch_matmul_fixpipe"):
    """
    batch_matmul_fixpipe op.

    Parameters:
    input_x1: dict, required
        A dict object, dict with keys(shape, dtype, and range)

    input_x2: dict, required
        A dict object, dict with keys(shape, dtype and range)

    quant_pre: dict, required
        A dict object of const node, const_value is required.

    bias: dict, optional
        A dict object, dict with keys(shape and dtype) or None

    y: dict
        A dict object, dict with keys(shape, dtype, format and range)

    adj_x1: bool
        If true, shape_a == transposed before multiplication

    adj_x2: bool
        If true, shape_b == transposed before multiplication

    kernel_name: str
        cce kernel_name
    """

    return batch_matmul_fixpipe_impl(x1, x2, quant_pre, bias, y, adj_x1, adj_x2, kernel_name)
