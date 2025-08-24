# Copyright 2020 Huawei Technologies Co., Ltd
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
avg_pool_1d
"""
import copy

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import buildcfg
from tbe.tvm.driver.cce_build_module import build_fatbin
from tbe.common.buildcfg import build_config
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    C0 = 16


# 'pylint: disable=too-many-arguments,too-many-locals,unused-argument,invalid-name
@register_operator_compute("avg_pool_1d", op_mode="dynamic", support_fusion=True)
def avg_pool_1d_compute(x,
                        div,
                        out_dict,
                        kernel,
                        pad,
                        stride,
                        ceil_mode=True,
                        count_include_pad=False,
                        kernel_name="avg_pool_1d"):
    """
    avg_pool_1d compute

    Parameters
    ----------
    x: input tensor dict
    div: matrix tensor dict
    out_dict: output dict
    kernel: the size of the window
    pad: implicit zero padding to be added on both sides
    stride: the stride of the window
    ceil_mode: when True, will use ceil instead of floor to compute the output shape
    count_include_pad: when True, will include the zero-padding in the averaging calculation
    kernel_name: kernel name

    Returns
    -------
    output tensor, reduce_tensor_list, tensor_list
    """
    x_wi = x.shape[1]
    x_wo = div.shape[1]
    pad_l, pad_r = pad

    # set padding
    x_fused_axis = x.shape[0]
    x_c0 = x.shape[-1].value
    mid_shape = (x_fused_axis, x_wi + pad_l + pad_r, x_c0)

    tensor_zero = tvm.compute(mid_shape, lambda *i: tvm.const(0, div.dtype), name="tensor_zero")

    tensor_pad = tvm.compute(
        mid_shape,
        lambda x_fused_axis, w, c0: tvm.select(tvm.all(w >= pad_l, w < x_wi + pad_l), x[x_fused_axis, w - pad_l, c0]),
        name="tensor_pad")

    tensor_mid_shape_in_ub = tvm.compute(mid_shape, lambda *i: tensor_zero[i] + tensor_pad[i], name="tensor_with_pad")

    # reduce w
    reduce_tensor_list = []
    re_shape = (x_fused_axis, x_wo, x_c0)
    if kernel > 1:
        # Add the first and second points of the sliding window
        tensor_w = tvm.compute(
            re_shape,
            lambda fused_axis, w, c0: tvm.sum(tensor_mid_shape_in_ub[fused_axis, w * stride + 0, c0],
                                              tensor_mid_shape_in_ub[fused_axis, w * stride + 1, c0]),
            name="tensor_w")
        reduce_tensor_list.append(tensor_w)
        # then accumulate the Nth point in sequence.
        for j in range(2, kernel):
            tensor_w_tmp = tvm.compute(
                re_shape,
                lambda fused_axis, w, c0, it=j: tvm.sum(tensor_mid_shape_in_ub[fused_axis, w * stride + it, c0],
                                                        tensor_w[fused_axis, w, c0]),
                name="tensor_w" + str(j))
            tensor_w = tensor_w_tmp
            reduce_tensor_list.append(tensor_w)
    elif kernel == 1:
        tensor_w = tvm.compute(
            re_shape,
            lambda fused_axis, w, c0: tensor_mid_shape_in_ub(fused_axis, w * stride, c0) + 0,
            name="tensor_w")
        reduce_tensor_list.append(tensor_w)

    tensor_list = [x, div, tensor_mid_shape_in_ub, tensor_zero, tensor_pad]
    res = tvm.compute(re_shape,
                      lambda i, j, k: tensor_w(i, j, k) * div(0, j, k),
                      attrs={
                          "stride": stride,
                          "kernel": kernel
                      },
                      name="res")
    return res, reduce_tensor_list, tensor_list


# 'pylint: disable=too-many-statements,invalid-name,unused-variable
def _avg_pool_1d_schedule(res, fmap_wo_var, reduce_tensor_list, tensor_list, cut_nc1h_for_block, cut_wo_for_block,
                          cut_wo_block_in, nc1h_in_factor, cut_wo, wo_ub_factor_max, ksize, strides):
    """
    avg_pool_1d schedule

    Parameters
    ----------
    res: result of compute
    reduce_tensor_list: list of reduce tensor
    tensor_list: list of tensors
    cut_nc1_for_block:
    cut_wo_for_block:
    cut_wo_block_in:
    nc1h_in_factor:
    cut_wo:
    wo_ub_factor_max:
    ksize:
    strides:

    Returns
    -------
    output sch
    """
    tensor_x = tensor_list[0]
    tensor_div = tensor_list[1]
    tensor_mid_shape_in_ub = tensor_list[2]
    tensor_zero = tensor_list[3]
    tensor_pad = tensor_list[4]

    sch = tvm.create_schedule(res.op)

    tensor_div_in_ub = sch.cache_read(tensor_div, tbe_platform.scope_ubuf, [res])
    sch[tensor_mid_shape_in_ub].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_zero].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_pad].set_scope(tbe_platform.scope_ubuf)

    for tensor in reduce_tensor_list:
        sch[tensor].set_scope(tbe_platform.scope_ubuf)
    tensor_ub_mul = sch.cache_write(res, tbe_platform.scope_ubuf)
    sch[tensor_mid_shape_in_ub].reused_by(tensor_zero, tensor_pad)

    block = tvm.thread_axis("blockIdx.x")
    factor_vars = []
    if cut_nc1h_for_block:
        fuse_factor = tvm.var("fuse_factor")
        factor_vars.append(fuse_factor)
        nc1h_out, nc1h_in = sch[res].split(res.op.axis[0], fuse_factor)
        sch[res].bind(nc1h_out, block)
        if cut_wo:
            wo_factor = tvm.var("wo_factor")
            sch.set_var_range(wo_factor, 1, wo_ub_factor_max)
            factor_vars.append(wo_factor)
            wo_out, wo_in = sch[res].split(res.op.axis[1], wo_factor)
            compute_at_axis = wo_out
            emit_insn_axis = wo_in
        else:
            fuse_in_factor = tvm.var("fuse_in_factor")
            factor_vars.append(fuse_in_factor)
            fmap_wo_var_min = 1 if nc1h_in_factor[0] is None else max(wo_ub_factor_max // nc1h_in_factor[0], 1)
            fmap_wo_var_max = max(wo_ub_factor_max // nc1h_in_factor[1], max(fmap_wo_var_min, 2))
            sch.set_var_range(fmap_wo_var, fmap_wo_var_min, fmap_wo_var_max)
            fuse_in_out, fuse_in_in = sch[res].split(nc1h_in, wo_ub_factor_max // fmap_wo_var_max)
            compute_at_axis = fuse_in_out
            emit_insn_axis = fuse_in_in

            sch[tensor_div_in_ub].set_buffer_size(wo_ub_factor_max * Constant.C0)
            sch[tensor_ub_mul].set_buffer_size(wo_ub_factor_max * Constant.C0)
            for i in range(ksize - 1):
                sch[reduce_tensor_list[i]].set_buffer_size(wo_ub_factor_max * Constant.C0)
            sch[tensor_zero].set_buffer_size(wo_ub_factor_max * Constant.C0 * (strides + ksize))

    else:
        if cut_wo_for_block:
            wo_block_factor = tvm.var("wo_block_factor")
            factor_vars.append(wo_block_factor)
            wo_block_out, wo_block_in = sch[res].split(res.op.axis[1], wo_block_factor)
            fuse_block = sch[res].fuse(res.op.axis[0], wo_block_out)
            sch[res].bind(fuse_block, block)
            if cut_wo_block_in:
                sch.set_var_range(wo_block_factor, wo_ub_factor_max + 1, None)
                wo_block_in_factor = tvm.var("wo_block_in_factor")
                sch.set_var_range(wo_block_in_factor, 1, wo_ub_factor_max)
                factor_vars.append(wo_block_in_factor)
                wo_block_in_out, wo_block_in_in = sch[res].split(wo_block_in, wo_block_in_factor)
                compute_at_axis = wo_block_in_out
                emit_insn_axis = wo_block_in_in
            else:
                sch.set_var_range(wo_block_factor, 1, wo_ub_factor_max)
                compute_at_axis = fuse_block
                emit_insn_axis = wo_block_in
        else:
            fuse_block = sch[res].fuse(res.op.axis[0], res.op.axis[1])
            sch[res].bind(fuse_block, block)
            compute_at_axis = fuse_block
            emit_insn_axis = res.op.axis[2]

    sch[tensor_div_in_ub].compute_at(sch[res], compute_at_axis)
    sch[tensor_mid_shape_in_ub].compute_at(sch[res], compute_at_axis)
    sch[tensor_pad].compute_at(sch[res], compute_at_axis)
    sch[tensor_zero].compute_at(sch[res], compute_at_axis)
    for tensor in reduce_tensor_list:
        sch[tensor].compute_at(sch[res], compute_at_axis)
    sch[tensor_ub_mul].compute_at(sch[res], compute_at_axis)
    sch[tensor_div_in_ub].emit_insn(tensor_div_in_ub.op.axis[0], tbe_platform.DMA_COPY)
    sch[tensor_zero].emit_insn(tensor_zero.op.axis[0], tbe_platform.DMA_COPY)
    sch[tensor_pad].emit_insn(tensor_pad.op.axis[0], tbe_platform.DMA_COPY)
    sch[tensor_mid_shape_in_ub].emit_insn(tensor_mid_shape_in_ub.op.axis[0], tbe_platform.PHONY_INSN)
    for tensor in reduce_tensor_list:
        sch[tensor].emit_insn(tensor.op.axis[0], tbe_platform.ADD)
    sch[tensor_ub_mul].emit_insn(tensor_ub_mul.op.axis[0], tbe_platform.MUL)
    sch[res].emit_insn(emit_insn_axis, tbe_platform.DMA_COPY)
    return sch, factor_vars


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,unused-variable
@register_operator("AvgPool1DD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def avg_pool_1d(x_dict,
                div_dict,
                out_dict,
                ksize,
                strides,
                pads,
                ceil_mode=True,
                count_include_pad=False,
                kernel_name="avg_pool_1d"):
    """
    Parameters
    ----------
    x_dict : dict, shape and dtype of input_data

    div_dict : dict, shape and dtype of matrix_data

    out_dict : dict, shape and dtype of output_data

    ksize : the size of the window

    strides : the strides of the window.

    pads : implicit zero padding to be added on both sides

    ceil_mode: when True, will use ceil instead of floor to compute the output shape

    count_include_pad: when True, will include the zero-padding in the averaging calculation

    kernel_name : cce kernel name, default value is "avg_pool_1d"

    Returns
    -------
    None
    """

    shape = x_dict.get("shape")
    div_shape = div_dict.get("shape")
    out_shape = out_dict.get("shape")
    dtype = x_dict.get("dtype")
    dtype_div = div_dict.get("dtype")

    pad_l, pad_r = pads

    x_n, x_c1, x_h, x_w, x_c0 = shape
    _, _, _, div_x_w, div_x_c0 = div_shape

    core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

    case1 = {"cut_nc1h_for_block": True, "nc1h_in_factor": [8, 2], "cut_wo": False, "cut_wo_for_block": False,
             "cut_wo_block_in": False, "nc1h_range": (core_num, None)}
    case2 = {"cut_nc1h_for_block": True, "nc1h_in_factor": [], "cut_wo": True, "cut_wo_for_block": False,
             "cut_wo_block_in": False, "nc1h_range": (core_num, None)}
    case3 = {"cut_nc1h_for_block": True, "nc1h_in_factor": [2, 1], "cut_wo": False, "cut_wo_for_block": False,
             "cut_wo_block_in": False, "nc1h_range": (core_num, None)}
    case4 = {"cut_nc1h_for_block": False, "nc1h_in_factor": [], "cut_wo": False, "cut_wo_for_block": True,
             "cut_wo_block_in": True, "nc1h_range": (1, core_num)}
    case5 = {"cut_nc1h_for_block": False, "nc1h_in_factor": [], "cut_wo": False, "cut_wo_for_block": True,
             "cut_wo_block_in": False, "nc1h_range": (1, core_num)}
    case6 = {"cut_nc1h_for_block": False, "nc1h_in_factor": [], "cut_wo": False, "cut_wo_for_block": False,
             "cut_wo_block_in": False, "nc1h_range": (1, core_num)}
    case7 = {"cut_nc1h_for_block": True, "nc1h_in_factor": [32, 8], "cut_wo": False, "cut_wo_for_block": False,
             "cut_wo_block_in": False, "nc1h_range": (core_num, None)}
    case8 = {"cut_nc1h_for_block": True, "nc1h_in_factor": [64, 32], "cut_wo": False, "cut_wo_for_block": False,
             "cut_wo_block_in": False, "nc1h_range": (core_num, None)}
    case9 = {"cut_nc1h_for_block": True, "nc1h_in_factor": [None, 64], "cut_wo": False, "cut_wo_for_block": False,
             "cut_wo_block_in": False, "nc1h_range": (core_num, None)}
    tiling_case = [case1, case2, case3, case4, case5, case6, case7, case8, case9]
    sch_list = []
    var_list = []
    rules = []
    ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    dtype_bytes_size = get_bit_len(dtype) // 8
    total_ele = ub_size_bytes // dtype_bytes_size
    wo_ub_factor_max = total_ele // Constant.C0 // (strides + 2 * ksize + 1)
    for idx, case in enumerate(tiling_case):
        fmap_nc1h_var = tvm.var("fmap_nc1h_var")
        fmap_wi_var = tvm.var("fmap_wi_var")
        fmap_wo_var = tvm.var("fmap_wo_var")

        if ceil_mode:
            pad_r = (fmap_wo_var - 1) * strides + ksize - fmap_wi_var - pad_l

        shape = [fmap_nc1h_var, fmap_wi_var, x_c0]
        div_shape = [1, fmap_wo_var, div_x_c0]

        tensor_a = tvm.placeholder(shape, name="tensor_a", dtype=dtype)
        tensor_div = tvm.placeholder(div_shape, name="tensor_div", dtype=dtype_div)

        res, reduce_tensor_list, tensor_list = \
            avg_pool_1d_compute(tensor_a, tensor_div, out_dict, ksize, [pad_l, pad_r],
                                strides, ceil_mode, count_include_pad, kernel_name)

        shape_vars = [fmap_nc1h_var, fmap_wi_var, fmap_wo_var]

        key = idx
        sch, factor_vars = _avg_pool_1d_schedule(res, fmap_wo_var, reduce_tensor_list, tensor_list,
                                                 cut_nc1h_for_block=case["cut_nc1h_for_block"],
                                                 nc1h_in_factor=case["nc1h_in_factor"],
                                                 cut_wo=case["cut_wo"],
                                                 cut_wo_for_block=case["cut_wo_for_block"],
                                                 cut_wo_block_in=case["cut_wo_block_in"],
                                                 wo_ub_factor_max=wo_ub_factor_max,
                                                 ksize=ksize,
                                                 strides=strides)
        rules.append(key)
        nc1h_min, nc1h_max = case['nc1h_range']
        sch.set_var_range(fmap_nc1h_var, nc1h_min, nc1h_max)
        sch_list.append(sch)
        var_list.append(shape_vars + factor_vars + [tensor_a, tensor_div, res])

    build_config_item = {"parse_ddr_args": True, "build_fatbin": True}
    dynamic_config = buildcfg.default_buildcfg.dynamic_build_config_dict
    with buildcfg.build_config(**dynamic_config):
        upper_config = buildcfg.get_current_build_config("all")
    upper_config.update(build_config_item)

    build_configs = []
    for sch in sch_list:
        dynamic_single_sch_build_config = copy.deepcopy(upper_config)
        build_configs.append(build_config(**dynamic_single_sch_build_config))
    build_fatbin(build_configs, sch_list, var_list, rules, kernel_name)

    tbe_context.get_context().add_compile_info("core_num", core_num)
    tbe_context.get_context().add_compile_info("max_w_in_ub", wo_ub_factor_max)
    tbe_context.get_context().add_compile_info("ksize", ksize)
    tbe_context.get_context().add_compile_info("strides", strides)
    tbe_context.get_context().add_compile_info("pad_l", pads[0])
    tbe_context.get_context().add_compile_info("pad_r", pads[1])
    tbe_context.get_context().add_compile_info("ceil_mode", ceil_mode)
