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
batch_matmul
"""
from functools import reduce

from tbe import tvm
from tbe.tvm import Tensor
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.dynamic.fix_pipe import fixpipe_compute
from impl.dynamic.batch_matmul_v2 import get_ori_batch_shape, get_input_shape
from tbe.common.context import get_context
from tbe.common.buildcfg import get_current_build_config
from tbe.common.platform import platform_info
from tbe.common.utils.const import ComputeFlow
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.compute.gemm_integrated_compute import GEMMCompute
from impl.util.platform_adapter import tbe_register


ND_LENGTH = 2
NZ_LENGTH = 4
PRE_CONV_MODE = {
    "float16":["S322F16", "VS322F16"]
}

NC1HWC0_C1_IDX = -4
NC1HWC0_C0_IDX = -1
MARK_VALUE_INT32 = 0X80000000


def quant_batch_matmul_compute(tensor_x1, tensor_x2, tensor_deq_scale, para_dict):
    quant_gemm_compute = QuantBmmCompute(tensor_x1, tensor_x2, tensor_deq_scale, para_dict)
    result = quant_gemm_compute.compute()
    return result


class QuantBmmCompute(GEMMCompute):
    def __init__(self, tensor_a, tensor_b, tensor_deq_scale, para_dict):
        super().__init__(tensor_a, tensor_b, para_dict)
        self.tensor_deq_scale = tensor_deq_scale
        self.pre_conv_mode = para_dict.get("pre_conv_mode")

    def compute(self):
        """
        the main func of gemm
        """
        # infer and update params from the origin inputs
        self._preprocess()
        tensor_a = self._compute_tensor_a()
        tensor_b = self._compute_tensor_b()
        tensor_mmad = self._compute_tensor_mmad(tensor_a, tensor_b)
        output_dict = self._get_output_dict()
        tensor_res = fixpipe_compute(tensor_mmad, None, self.tensor_deq_scale, None, None, None,
                                     None, None, None, None, output_dict, [], [], "")
        return tensor_res

    def _get_output_dict(self):
        m_ori = get_te_var("m_ori").get_tvm_var()
        n_ori = get_te_var("n_ori").get_tvm_var()
        ori_shape_out = [m_ori, n_ori]
        if len(self.shape_a) in [3, 5] or len(self.shape_b) in [3, 5]:
            batch = get_te_var("batch").get_tvm_var()
            ori_shape_out.insert(0, batch)
        shape_out = self._get_dynamic_out_shape()
        return {"shape": shape_out, "ori_shape": ori_shape_out, "format": self.format_out, "dtype":self.dst_dtype}

    def _get_mmad_kwargs(self):
        kwargs = super()._get_mmad_kwargs()
        attr_dict = kwargs.get("attrs")
        attr_dict["unaligned_flag"] = self.unaligned_flag
        attr_dict["pre_conv_mode"] = self.pre_conv_mode
        kwargs["attrs"] = attr_dict
        return kwargs


def _get_input_shape(quant_bmm_inputs, kernel_name, unaligned_flag):
    x1, x2, bias, deq_scale, y, adj_x1, adj_x2 = quant_bmm_inputs
    extra_params = {"op_type": "BatchMatMulV2"}
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if support_l0c2out and x1.get("format") == "ND" and x2.get("format") == "ND":
        extra_params["nd2nz_type"] = ComputeFlow.on_the_fly.value
    res = get_input_shape(x1, x2, bias, y, adj_x1, adj_x2, kernel_name, extra_params, unaligned_flag)
    return res


def _creat_bias_placeholder(bias, unaligned_flag):
    if bias is None:
        return None
    n_ori = get_te_var("n_ori").get_tvm_var()
    n = get_te_var("n").get_tvm_var()
    block_n0 = int(platform_info.get_soc_spec("cube_n_size"))
    if unaligned_flag:
        bias_shape = [n_ori]
    else:
        bias_shape = [n * block_n0]
    tensor_bias = tvm.placeholder(
        bias_shape, name="bias", dtype=bias.get("dtype"), attrs={'ori_shape': bias_shape})
    return tensor_bias


def _creat_deq_scale_tensor(deq_scale, pre_conv_mode):
    n = get_te_var("n").get_tvm_var()
    block_n0 = int(platform_info.get_soc_spec("cube_n_size"))
    if pre_conv_mode == "S322F16":
        deq_s_ori_shape = (1, 1, 1, 1)
        deq_s_shape = (1, 1, 1, 1, block_n0)
        tensor_deq_scale = tvm.placeholder(deq_s_shape, name="deq_scale", dtype=deq_scale.get("dtype"),
                                           attrs={"ori_shape": deq_s_ori_shape})
    else:
        n_ori = get_te_var("n_ori").get_tvm_var()
        n = get_te_var("n").get_tvm_var()
        deq_s_ori_shape = (1, 1, 1, n_ori)
        deq_s_shape = (1, n, 1, 1, block_n0)
        tensor_deq_scale = tvm.placeholder(deq_s_shape, name="deq_scale", dtype=deq_scale.get("dtype"),
                                           attrs={'ori_shape': deq_s_ori_shape})
    return tensor_deq_scale


@register_operator("QuantBatchMatmul")
def quant_batch_matmul(x1, x2, deq_scale, bias, y=None, adj_x1=False, adj_x2=False, kernel_name="quant_batch_matmul"):
    quant_bmm_inputs = [x1, x2, bias, deq_scale, y, adj_x1, adj_x2]
    tensor_lists = []
    schs = []
    res = _get_input_shape(quant_bmm_inputs, kernel_name, 1)
    shape_x1, shape_x2, is_cache_tiling, nd2nz_type, input_range = res
    format_a = x1.get("format")
    format_b = x2.get("format")
    dtype_in = x1.get("dtype").lower()
    dtype_out = y.get("dtype").lower()
    for unaligned_flag in [0, 1]:
        tensor_x1 = tvm.placeholder(shape_x1, name="tensor_a", dtype=dtype_in)
        tensor_x2 = tvm.placeholder(shape_x2, name="tensor_b", dtype=dtype_in)
        ori_batch_x1, ori_batch_x2 = get_ori_batch_shape(x1, x2)
        tensor_x1.op.attrs["ori_batch_shape"] = ori_batch_x1
        tensor_x2.op.attrs["ori_batch_shape"] = ori_batch_x2
        tensor_bias = _creat_bias_placeholder(bias, unaligned_flag)
        for pre_conv_mode in PRE_CONV_MODE.get("float16"):
            tensor_deq_scale = _creat_deq_scale_tensor(deq_scale, pre_conv_mode)
            tensor_list = [tensor_x1, tensor_x2, tensor_deq_scale]
            if tensor_bias is not None:
                tensor_list.append(tensor_bias)
            para_dict = {
                "format_a": format_a,
                "format_b": format_b,
                "format_out": y.get("format"),
                "trans_a": adj_x1,
                "trans_b": adj_x2,
                "dst_dtype": dtype_out,
                "tensor_c": tensor_bias,
                "cache_tiling_flag": is_cache_tiling,
                "unaligned_flag": unaligned_flag,
                "pre_conv_mode": pre_conv_mode,
                "kernel_name": kernel_name,
                "input_range": input_range,
                "nd2nz_type": nd2nz_type
            }
            with tbe.compute():
                out = quant_batch_matmul_compute(tensor_x1, tensor_x2, tensor_deq_scale, para_dict)
            with tvm.target.cce():
                sch = tbe.auto_schedule(out)
            tensor_list.append(out)
            tensor_lists.append(tensor_list)
            schs.append(sch)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_lists,
        "double_buffer_non_reuse": True,
        "build_args": {"enable_db_fold": True}
    }
    tbe.build(schs, config)



@tbe_register.register_param_generalization("QuantBatchMatmul")
def quant_batch_matmul_generalization(x1,
                                      x2,
                                      deq_scale,
                                      bias,
                                      y=None,
                                      adj_x1=False,
                                      adj_x2=False,
                                      kernel_name="quant_batch_matmul",
                                      generalize_config=None):
    result = []
    if isinstance(generalize_config, dict) and generalize_config.get("mode") == "all_shape":
        x1.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        x2.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        deq_scale.update({"ori_shape": [-2], "ori_format": "NHWC", "shape": [-2], "format": "NC1HWC0"})
        if bias is not None:
            bias.update({"ori_sha pe": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        y.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        result.append([x1, x2, deq_scale, bias, y, adj_x1, adj_x2])
    return result