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

import te.platform as tbe_platform
from tbe import tvm
from te.utils import para_check
from te.utils import error_manager
from te.utils.error_manager import error_manager_vector
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.platform_adapter import build_config
from tbe.common.platform import get_bit_len


# 'pylint: disable = unused-argument,redefined-builtin,too-many-arguments
def get_op_support_info(input_x,
                        assist_matrix,
                        out_dict,
                        ksize,
                        strides,
                        pads,
                        ceil_mode=True,
                        count_include_pad=False,
                        kernel_name="avg_pool_1d"):
    """
    get avg_pool_1d slice info
    """
    format_x = input_x.get("format")
    if format_x == "NC1HWC0":
        axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])],
                             [SplitInput([0, [2], [-1], [-1]]), SplitOutput([0, [2]])]]
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)

    return op_cal_info_in_json


# 'pylint: disable=too-many-arguments
def _parameter_check(shape, div_shape, shape_out, dtype, ksize, pads):
    para_check.check_shape(shape, param_name="x")
    para_check.check_shape(div_shape, param_name="div")
    para_check.check_shape(shape_out, param_name="out")
    para_check.check_dtype(dtype, ("float16", "float32"), param_name="x")
    half_kernel_size = ksize // 2
    if pads[0] > half_kernel_size:
        error_manager_vector.raise_err_check_params_rules(
            "avg_pool_1d", 'pad should be smaller than half of kernel size, kernel_size is {}'.format(ksize), 'pad',
            pads)


# 'pylint: disable=too-many-arguments,too-many-locals,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("avg_pool_1d")
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
    shape = [i.value for i in x.shape]
    x_wi = shape[-2]
    pad_l, pad_r = pad

    if ceil_mode:
        x_wo = (x_wi + pad_l + pad_r - kernel + stride - 1) // stride + 1
    else:
        x_wo = ((x_wi + pad_l + pad_r) - kernel) // stride + 1

    if x_wo <= 0:
        dict_args = {
            'errCode': 'E80009',
            'op_name': 'avg_pool_1d',
            'rule_desc': 'x W(after pad) must >= filter W',
            'param_name': '[x W, filter W]',
            'param_value': '[{},{}]'.format(x_wi + pad_l + pad_r, kernel)
        }
        raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

    if pad_l:
        # ensure that the last pooling starts inside the image needed to avoid problems in ceil mode
        # existing bug in pytorch code
        # pad_l = 0 and stride is big, but kernel is small, return nan
        if ((x_wo - 1) * stride) >= (x_wi + pad_l):
            x_wo -= 1
    pad_r = (x_wo - 1) * stride + kernel - x_wi - pad_l

    # set padding
    x_fused_axis, x_w, x_c0 = shape
    mid_shape = (x_fused_axis, x_w + pad_l + pad_r, x_c0)
    tensor_mid_shape_in_ub = tvm.compute(
        mid_shape,
        lambda x_fused_axis, w, c0: tvm.select(tvm.any(w < pad_l, w >= x_wi + pad_l), tvm.const(0, dtype=x.dtype), x[
            x_fused_axis, w - pad_l, c0]),
        name="tensor_mid_shape_in_ub")

    # reduce w
    reduce_tensor_list = []
    re_shape = (x_fused_axis, x_wo, x_c0)
    if kernel > 1:
        # 100 is an experience num to avoid stack overflow
        if kernel <= 100:
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
        else:
            w_axis = tvm.reduce_axis((0, kernel), "w_sum")
            tensor_w = tvm.compute(re_shape,
                                   lambda fused_axis, w, c0: tvm.sum(
                                       tensor_mid_shape_in_ub[fused_axis, w * stride + w_axis, c0], axis=w_axis),
                                   name="tensor_w")
            reduce_tensor_list.append(tensor_w)
    elif kernel == 1:
        tensor_w = tvm.compute(re_shape,
                               lambda fused_axis, w, c0: tensor_mid_shape_in_ub(fused_axis, w * stride, c0) + 0,
                               name="tensor_w")
        reduce_tensor_list.append(tensor_w)

    tensor_list = [x, div, tensor_mid_shape_in_ub]
    res = tvm.compute(re_shape,
                      lambda i, j, k: tensor_w(i, j, k) * div(0, j, k),
                      attrs={
                          "stride": stride,
                          "kernel": kernel
                      },
                      name="res")
    return res, reduce_tensor_list, tensor_list


# 'pylint: disable=too-many-statements,invalid-name
def _avg_pool_1d_schedule(res, reduce_tensor_list, tensor_list):
    """
    avg_pool_1d schedule

    Parameters
    ----------
    res: result of compute
    reduce_tensor_list: list of reduce tensor
    tensor_list: list of tensors

    Returns
    -------
    output sch
    """
    tensor_x = tensor_list[0]
    tensor_div = tensor_list[1]
    tensor_mid_shape_in_ub = tensor_list[2]

    def _ceil(m, n):
        return (m + n - 1) // n

    def _tiling(shape, dtype, wo_out, stride, kernel):
        ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        dtype_bytes_size = get_bit_len(dtype) // 8
        total_ele = ub_size_bytes // dtype_bytes_size // 2

        nc1h, _, c0 = shape
        core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        fused_axis_block_factor, w_block_factor = nc1h, wo_out

        if fused_axis_block_factor >= core_num:
            fused_axis_block_factor = _ceil(fused_axis_block_factor, core_num)
        else:
            w_block_factor = _ceil(wo_out, _ceil(core_num, fused_axis_block_factor))
            fused_axis_block_factor = 1

        # for wi = (wo - 1) * stride + kernel
        # wo_buffer_num * N * C1 * H * Wo * C0 + wi_buffer_num * N * C1 * H * Wi * C0 <= total_ele
        wo_buffer_num = 4
        wi_buffer_num = 2
        nc1wo_limit = (total_ele // c0 + (stride - kernel) * wi_buffer_num) // (wo_buffer_num + stride * wi_buffer_num)
        nc1_limit = (total_ele // c0) // (wo_buffer_num * wo_out + wi_buffer_num * wo_out * stride - wi_buffer_num *
                                          (stride - kernel))

        if nc1_limit > 1:
            fused_factor, wo_factor = min(fused_axis_block_factor, nc1_limit), w_block_factor
        elif nc1wo_limit >= 8:
            # To align 8 blocks, round down to a multiple of 8.
            fused_factor, wo_factor = 1, nc1wo_limit // 8 * 8
        else:
            fused_factor, wo_factor = 1, 1
        return [fused_axis_block_factor, w_block_factor], [fused_factor, wo_factor]

    x_shape = [i.value for i in tensor_x.shape]
    stride = int(res.op.attrs['stride'].value)
    kernel = int(res.op.attrs['kernel'].value)
    [fused_b_factor, w_b_factor], [fused_factor, wo_factor] = _tiling(x_shape, tensor_x.dtype,
                                                                      tensor_div.shape[-2].value, stride, kernel)

    sch = tvm.create_schedule(res.op)
    # set output ub
    tensor_div_in_ub = sch.cache_read(tensor_div, tbe_platform.scope_ubuf, [res])
    sch[tensor_mid_shape_in_ub].set_scope(tbe_platform.scope_ubuf)
    for tensor in reduce_tensor_list:
        sch[tensor].set_scope(tbe_platform.scope_ubuf)
    tensor_ub_mul = sch.cache_write(res, tbe_platform.scope_ubuf)

    fused_b_out, fused_b_in = sch[res].split(res.op.axis[0], fused_b_factor)
    fused_out, fused_in = sch[res].split(fused_b_in, fused_factor)
    wo_b_out, wo_b_in = sch[res].split(res.op.axis[1], w_b_factor)
    wo_out, wo_in = sch[res].split(wo_b_in, wo_factor)
    sch[res].reorder(fused_b_out, wo_b_out, wo_out, fused_out, fused_in, wo_in, res.op.axis[-1])

    # split tensor_w for reduce sum
    sch[tensor_div_in_ub].compute_at(sch[res], wo_out)
    sch[tensor_mid_shape_in_ub].compute_at(sch[res], fused_out)
    for tensor in reduce_tensor_list:
        sch[tensor].compute_at(sch[res], fused_out)
    sch[tensor_ub_mul].compute_at(sch[res], fused_out)

    sch[tensor_div_in_ub].double_buffer()
    sch[tensor_mid_shape_in_ub].preload()
    sch[tensor_mid_shape_in_ub].double_buffer()
    for tensor in reduce_tensor_list:
        sch[tensor].double_buffer()
    sch[tensor_ub_mul].double_buffer()

    # for multi cores
    block = tvm.thread_axis("blockIdx.x")
    block_axis = sch[res].fuse(fused_b_out, wo_b_out)
    sch[res].bind(block_axis, block)

    # set emit_insn
    sch[tensor_div_in_ub].emit_insn(tensor_div_in_ub.op.axis[0], tbe_platform.DMA_COPY)
    sch[tensor_mid_shape_in_ub].emit_insn(tensor_mid_shape_in_ub.op.axis[0], tbe_platform.DMA_PADDING)
    for tensor in reduce_tensor_list:
        sch[tensor].emit_insn(tensor.op.axis[0], tbe_platform.ADD)
    sch[tensor_ub_mul].emit_insn(tensor_ub_mul.op.axis[0], tbe_platform.MUL)
    sch[res].emit_insn(fused_in, tbe_platform.DMA_COPY)
    return sch


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
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
    _parameter_check(shape, div_shape, out_shape, dtype, ksize, pads)

    x_n, x_c1, x_h, x_w, x_c0 = shape
    _, _, _, div_x_w, div_x_c0 = div_shape
    shape = [x_n * x_c1 * x_h, x_w, x_c0]
    div_shape = [1, div_x_w, div_x_c0]

    tensor_a = tvm.placeholder(shape, name="tensor_a", dtype=dtype)
    tensor_div = tvm.placeholder(div_shape, name="tensor_div", dtype=dtype_div)

    res, reduce_tensor_list, tensor_list = avg_pool_1d_compute(tensor_a, tensor_div, out_dict, ksize, pads, strides,
                                                               ceil_mode, count_include_pad, kernel_name)
    sch = _avg_pool_1d_schedule(res, reduce_tensor_list, tensor_list)

    with build_config():
        tvm.build(sch, [tensor_a, tensor_div, res], "cce", name=kernel_name)
