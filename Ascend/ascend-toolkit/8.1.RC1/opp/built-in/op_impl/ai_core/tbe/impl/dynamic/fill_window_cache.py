# Copyright 2024 Huawei Technologies Co., Ltd
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
fill_window_cache
"""

import numpy as np
from tbe import tvm
from tbe.common.utils import para_check
from tbe.common.utils.errormgr import error_manager_vector
from impl import common_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform


BLOCK_SIZE = tbe_platform.get_block_size()
M0 = 1
ND_LEN = 2
NZ_N1 = -4
NCHW_N = 0
NCHW_C = 1
NCHW_H = 2
NCHW_W = 3
FORMAT_MAPS = {
    "NCHW":"NC1HWC0",
    "NHWC":"NC1HWC0",
    "ND":"FRACTAL_NZ"
}


def check_params(x, y, axis, cache_depth):
    """
    Verify the operator parameters.
    """
    op_name = "FillWindowCache"
    x_shape = x.get("shape")
    x_ori_shape = x.get("ori_shape")
    x_format = x.get("format")
    x_ori_format = x.get("ori_format")
    if axis < 0:
        axis = axis + len(x_ori_shape)
    if axis < 0 or axis >= len(x_ori_shape):
        error_message = "The axis is out of the dimensional range of the input shape."
        error_manager_vector.raise_err_specific_reson(op_name, error_message)
    y_shape = y.get("shape")
    y_ori_shape = y.get("ori_shape")
    y_format = y.get("format")
    y_ori_format = y.get("ori_format")
    if y_format != x_format:
        error_message = "The input and output formats are inconsistent."
        error_manager_vector.raise_err_specific_reson(op_name, error_message)
    elif x_format != "NC1HWC0" and x_format != "FRACTAL_NZ":
        error_message = f"Currently, the {x_format} format is not supported."
        error_manager_vector.raise_err_specific_reson(op_name, error_message)
    elif y_ori_format != x_ori_format:
        error_message = "The input and output original formats are inconsistent."
        error_manager_vector.raise_err_specific_reson(op_name, error_message)
    elif y_format != FORMAT_MAPS.get(y_ori_format) or x_format != FORMAT_MAPS.get(x_ori_format):
        error_message = "The format does not match the original format."
        error_manager_vector.raise_err_specific_reson(op_name, error_message)
    elif len(x_shape) != len(y_shape):
        error_message = "The lengths of the input and output shapes do not match."
        error_manager_vector.raise_err_specific_reson(op_name, error_message)
    elif len(x_ori_shape) != len(y_ori_shape):
        error_message = "The lengths of the original input and original output shapes do not match."
        error_manager_vector.raise_err_specific_reson(op_name, error_message)
    real_cache_depth = list(y_ori_shape)[axis] / list(x_ori_shape)[axis]
    if real_cache_depth != cache_depth:
        error_message = "The cache_depth does not match the input/output shape."
        error_manager_vector.raise_err_specific_reson(op_name, error_message)


def combine_nz_dims(nz_shape, axis):
    axis = axis - (len(nz_shape) - ND_LEN) if axis >= 0 else axis
    *batch, n1, m1, m0, n0 = nz_shape
    if axis >= -ND_LEN:
        batch1, batch2 = 1, int(np.prod(batch))
    else:
        axis -= ND_LEN
        batch1, batch2 = int(np.prod(nz_shape[:axis])), int(np.prod(nz_shape[axis:NZ_N1]))
    new_shape = (batch1, batch2, n1, m1, m0, n0)
    return new_shape


def reshape_fifo_tensor(tensor, y_infos, axis):
    def lambda_reshape_fifo_tensor(*indices):
        if output_shape_lens < tensor_lens:
            indices = [0] * (tensor_lens - output_shape_lens) + list(indices)
            return tensor(*indices)
        else:
            address = 0
            for i, index in enumerate(indices[:NZ_N1]):
                address += index * int(np.prod(res_shape[(i + 1):NZ_N1]))
            res_a2 = address % a2_axis
            res_a1 = address // a2_axis
            return tensor(res_a1, res_a2, *indices[NZ_N1:])
    res_shape = y_infos.get("shape")
    _, a2_axis, *_ = tensor.shape
    tensor_lens = len(tensor.shape)
    output_shape_lens = len(res_shape)
    res_tensor = tvm.compute(
        res_shape, 
        lambda *indices: lambda_reshape_fifo_tensor(*indices),
        name="fwc_res",
        attrs={
            "ori_format": y_infos.get("ori_format"),
            "ori_shape": y_infos.get("ori_shape")
        }
    )
    return res_tensor


def nz_last_dim_not_align_compute(attrs_dict):
    def lambda_last_dim_not_align_compute(a1, a2, n1, n0, m):
        a1_1, a2_1, n1_1, n0_1, m_1 = a1_2, a2_2, n1_2, n0_2, m_2 = a1, a2, n1, n0, m
        rotation_axis = n1 * n0_axis + n0
        a1_1 = (a2_axis * x_n1_axis + a1 * a2_axis * n1_axis + a2 * n1_axis +
                (n1 * n0_axis + n0 + axis_dim) // n0_axis) // x_n1_axis // x_a2_axis
        a2_1 = (a2_axis * x_n1_axis + a1 * a2_axis * n1_axis + a2 * n1_axis +
                (n1 * n0_axis + n0 + axis_dim) // n0_axis) // x_n1_axis % x_a2_axis
        n1_1 = ((n1 * n0_axis + n0 + axis_dim) // n0_axis) % x_n1_axis
        n0_1 = (n1 * n0_axis + n0 + axis_dim) % n0_axis
        n1_2 = ((n1 * n0_axis + n0) - axis_dim * (cache_depth - 1)) // n0_axis
        n0_2 = ((n1 * n0_axis + n0) - axis_dim * (cache_depth - 1)) % n0_axis
        return tvm.select(
            clean_cache_tensor[0] < tvm.const(1, clean_cache_tensor.dtype),
            tvm.select(
                rotation_axis < axis_dim * (cache_depth - 1),
                x_transpose_tensor(a1_1, a2_1, n1_1, n0_1, m_1),
                tvm.select(
                    rotation_axis < axis_dim * cache_depth,
                    x_transpose_tensor(a1_2, a2_2, n1_2, n0_2, m_2),
                    tvm.const(0, x_transpose_tensor.dtype)
                )                
            ),
            tvm.select(
                rotation_axis < axis_dim * (cache_depth - 1),
                tvm.const(0, x_transpose_tensor.dtype),
                tvm.select(
                    rotation_axis < axis_dim * cache_depth,
                    x_transpose_tensor(a1_2, a2_2, n1_2, n0_2, m_2),
                    tvm.const(0, x_transpose_tensor.dtype)
                )
            )
        )
    input_tensor = attrs_dict.get("input_tensor")
    x_ori_shape = attrs_dict.get("x_ori_shape")
    clean_cache_tensor = attrs_dict.get("clean_cache_tensor")
    y_combine_shape = attrs_dict.get("y_combine_shape")
    axis = attrs_dict.get("axis")
    cache_depth = attrs_dict.get("cache_depth")

    axis_dim = x_ori_shape[axis]
    input_shape = input_tensor.shape
    _, x_a2_axis, x_n1_axis, *_ = input_shape
    x_reshape_shape = [*input_shape[:-3], input_shape[-3] * input_shape[-2], input_shape[-1]]
    x_transpose_shape = [*x_reshape_shape[:-2], x_reshape_shape[-1], x_reshape_shape[-2]]
    fifo_shape = [*y_combine_shape[:-3], y_combine_shape[-1], y_combine_shape[-3] * y_combine_shape[-2]]
    fifo_transpose_shape = [*fifo_shape[:-2], fifo_shape[-1], fifo_shape[-2]]
    fifo_transpose_reshape = [
        *fifo_transpose_shape[:-2], (fifo_transpose_shape[-2] + input_shape[-2] - 1) // input_shape[-2],
        input_shape[-2], fifo_transpose_shape[-1]
    ]
    _, a2_axis, n1_axis, n0_axis, _ = fifo_shape
    x_reshape_tensor = tvm.compute(
        x_reshape_shape, lambda a1, a2, n1, m, n0: input_tensor(a1, a2, n1, m // M0, m % M0, n0),
        name="x_reshape_tensor"
    )
    x_transpose_tensor = tvm.compute(
        x_transpose_shape, lambda a1, a2, n1, n0, m: x_reshape_tensor(a1, a2, n1, m, n0), name="x_tranpose_tensor"
    )
    fifo_tensor = tvm.compute(
        fifo_shape, lambda *indices: lambda_last_dim_not_align_compute(*indices), name="fifo_tensor"
    )
    fifo_transpose_tensor = tvm.compute(
        fifo_transpose_shape, lambda a1, a2, n1, m, n0: fifo_tensor(a1, a2, n1, n0, m), name="fifo_transpose_tensor"
    )
    fifo_trans_reshape_tensor = tvm.compute(
        fifo_transpose_reshape, lambda a1, a2, n1, m1, m0, n0: fifo_transpose_tensor(a1, a2, n1, m1 * M0 + m0, n0),
        name="fifo_trans_reshape_tensor"
    )
    return fifo_trans_reshape_tensor


def nz_last_dim_align_compute(attrs):
    def lambda_last_dim_align_compute(*indices):
        a1, a2, n1, m1, m0, n0 = indices
        a1_1, a2_1, n1_1, m1_1, m0_1, n0_1 = a1_2, a2_2, n1_2, m1_2, m0_2, n0_2 = a1, a2, n1, m1, m0, n0
        rotation_axis = n1 * n0_axis + n0
        a1_1 = (a1_axis * a2_axis * n1_axis + a1 * a2_axis * ((n_axis * cache_depth + n0_axis - 1) //
                n0_axis) + a2 * ((n_axis * cache_depth + n0_axis - 1) // n0_axis) + (((n1 * n0_axis + n0) +
                axis_dim) // n0_axis)) // n1_axis // a2_axis
        a2_1 = (a1_axis * a2_axis * n1_axis + a1 * a2_axis * ((n_axis * cache_depth + n0_axis - 1) // 
                n0_axis) + a2 * ((n_axis * cache_depth + n0_axis - 1) // n0_axis) + (((n1 * n0_axis + n0) +
                axis_dim) // n0_axis)) // n1_axis % a2_axis
        n1_1 = (a1_axis * a2_axis * n1_axis + a1 * a2_axis * ((n_axis * cache_depth + n0_axis - 1) // 
                n0_axis) + a2 * ((n_axis * cache_depth + n0_axis - 1) // n0_axis) + (((n1 * n0_axis + n0) +
                axis_dim) // n0_axis)) % n1_axis
        n0_1 = ((n1 * n0_axis + n0) + axis_dim) % n0_axis
        n1_2 = ((n1 * n0_axis + n0) - axis_dim * (cache_depth - 1)) // n0_axis
        n0_2 = ((n1 * n0_axis + n0) - axis_dim * (cache_depth - 1)) % n0_axis

        tensor = tvm.select(
            clean_cache[0] < tvm.const(1, clean_cache.dtype),
            tvm.select(rotation_axis < axis_dim * (cache_depth - 1),
                input_tensor(a1_1, a2_1, n1_1, m1_1, m0_1, n0_1),
                input_tensor(a1_2, a2_2, n1_2, m1_2, m0_2, n0_2)
            ),
            tvm.select(rotation_axis < axis_dim * (cache_depth - 1),
                tvm.const(0, input_tensor.dtype),
                input_tensor(a1_2, a2_2, n1_2, m1_2, m0_2, n0_2)
            )
        )
        return tensor
    input_tensor = attrs.get("input_tensor")
    clean_cache = attrs.get("clean_cache_tensor")
    x_ori_shape = attrs.get("x_ori_shape")
    y_combine_shape = attrs.get("y_combine_shape")
    x_combine_shape = attrs.get("x_combine_shape")
    axis_dim = x_ori_shape[axis]
    a1_axis, a2_axis, n1_axis, *_, n0_axis = x_combine_shape
    *_, n_axis = x_ori_shape
    fifo_tensor = tvm.compute(
        y_combine_shape, lambda *indices: lambda_last_dim_align_compute(*indices), name="fifo_tensor"
    )
    return fifo_tensor


def fill_window_cache_nz_compute(attrs):
    def lambda_nz_compute(*indices):
        a1, a2, n1, m1, m0, n0 = indices
        a1_1, a2_1, n1_1, m1_1, m0_1, n0_1 = a1_2, a2_2, n1_2, m1_2, m0_2, n0_2 = a1, a2, n1, m1, m0, n0
        if axis == -2:
            rotation_axis = m1 * m0_axis + m0
            a1_1 = (a1_axis * a2_axis * n1_axis * m1_axis + a1 * a2_axis * n1_axis * (m_axis * cache_depth +
                    m0_axis - 1) // m0_axis + a2 * n1_axis * (m_axis * cache_depth + m0_axis - 1) // m0_axis +
                    n1 * (m_axis * cache_depth + m0_axis - 1) // m0_axis + (m1 * m0_axis + m0 + axis_dim) //
                    m0_axis) // m1_axis // n1_axis // a2_axis
            a2_1 = (a1_axis * a2_axis * n1_axis * m1_axis + a1 * a2_axis * n1_axis * (m_axis * cache_depth +
                    m0_axis - 1) // m0_axis + a2 * n1_axis * (m_axis * cache_depth + m0_axis - 1) // m0_axis +
                    n1 * (m_axis * cache_depth + m0_axis - 1) // m0_axis + (m1 * m0_axis + m0 + axis_dim) //
                    m0_axis) // m1_axis // n1_axis % a2_axis
            n1_1 = (a1_axis * a2_axis * n1_axis * m1_axis + a1 * a2_axis * n1_axis * (m_axis * cache_depth +
                    m0_axis - 1) // m0_axis + a2 * n1_axis * (m_axis * cache_depth + m0_axis - 1) // m0_axis +
                    n1 * (m_axis * cache_depth + m0_axis - 1) // m0_axis + (m1 * m0_axis + m0 + axis_dim) //
                    m0_axis) // m1_axis % n1_axis
            m1_1 = (a1_axis * a2_axis * n1_axis * m1_axis + a1 * a2_axis * n1_axis * (m_axis * cache_depth +
                    m0_axis - 1) // m0_axis + a2 * n1_axis * (m_axis * cache_depth + m0_axis - 1) // m0_axis +
                    n1 * (m_axis * cache_depth + m0_axis - 1) // m0_axis + (m1 * m0_axis + m0 + axis_dim) //
                    m0_axis) % m1_axis
            m0_1 = (((m1 * m0_axis + m0) + axis_dim) % m0_axis)
            m1_2 = ((m1 * m0_axis + m0) - axis_dim * (cache_depth - 1)) // m0_axis
            m0_2 = ((m1 * m0_axis + m0) - axis_dim * (cache_depth - 1)) % m0_axis
        else:
            rotation_axis = a2
            a1_1 = (a1_axis * a2_axis + a1 * (a2_axis * cache_depth) + (a2 + axis_dim)) // a2_axis
            a2_1 = (a1_axis * a2_axis + a1 * (a2_axis * cache_depth) + (a2 + axis_dim)) % a2_axis
            a2_2 = a2 - axis_dim * (cache_depth - 1)
        return tvm.select(
            clean_cache_tensor[0] < tvm.const(1, clean_cache_tensor.dtype),
            tvm.select(
                rotation_axis < axis_dim * (cache_depth - 1),
                input_tensor(a1_1, a2_1, n1_1, m1_1, m0_1, n0_1),
                input_tensor(a1_2, a2_2, n1_2, m1_2, m0_2, n0_2)
            ),
            tvm.select(
                rotation_axis < axis_dim * (cache_depth - 1),
                tvm.const(0, input_tensor.dtype),
                input_tensor(a1_2, a2_2, n1_2, m1_2, m0_2, n0_2)
            )
        )
    input_tensor = attrs.get("input_tensor")
    clean_cache_tensor = attrs.get("clean_cache_tensor")
    y_infos = attrs.get("y_infos")
    axis = attrs.get("axis")
    cache_depth = attrs.get("cache_depth")
    x_combine_shape = attrs.get("x_combine_shape")

    y_shape, y_ori_shape = y_infos.get("shape"), y_infos.get("ori_shape")
    x_ori_shape = list(y_ori_shape)
    x_ori_shape[axis] = x_ori_shape[axis] // cache_depth
    *_, m_axis, _ = x_ori_shape
    if axis < -2:
        axis_dim = x_combine_shape[-5]
    else:
        axis_dim = x_ori_shape[axis]
    a1_axis, a2_axis, n1_axis, m1_axis, m0_axis, _ = x_combine_shape
    y_combine_shape = combine_nz_dims(y_shape, axis)
    attrs.update(y_combine_shape=y_combine_shape)
    attrs.update(x_ori_shape=x_ori_shape)
    if axis == -1:
        if axis_dim * common_util.get_data_size(input_tensor.dtype) % BLOCK_SIZE != 0:
            fifo_tensor = nz_last_dim_not_align_compute(attrs)
        else:
            fifo_tensor = nz_last_dim_align_compute(attrs)
    else:
        fifo_tensor = tvm.compute(y_combine_shape, lambda *indices: lambda_nz_compute(*indices), name="fifo_tensor")
    reshape_tensor = reshape_fifo_tensor(fifo_tensor, y_infos, axis)
    return reshape_tensor


def nchw_c_dim_not_align_compute(attrs):
    def fifo_lambda_compute(n, c1, c0, hw):
        c_indices = c1 * c0_dim + c0
        return tvm.select(hw < hw_dim,
                   tvm.select(
                       clean_cache[0] < tvm.const(1, clean_cache.dtype),
                       tvm.select(
                           c1 * c0_dim + c0 < axis_dim * (cache_depth - 1),
                           y_transpose_tensor(n, (c_indices + axis_dim) // c0_dim, (c_indices + axis_dim) % c0_dim, hw),
                           x_transpose_tensor(n, (c_indices - axis_dim * (cache_depth - 1)) // c0_dim,
                                              (c_indices - axis_dim * (cache_depth - 1)) % c0_dim, hw)
                       ),
                       tvm.select(
                           c1 * c0_dim + c0 < axis_dim * (cache_depth - 1),
                           tvm.const(0, input_tensor.dtype),
                           x_transpose_tensor(n, (c_indices - axis_dim * (cache_depth - 1)) // c0_dim,
                                              (c_indices - axis_dim * (cache_depth - 1)) % c0_dim, hw)
                       )
                   )
               )

    input_tensor = attrs.get("input_tensor")
    cache_depth = attrs.get("cache_depth")
    axis = attrs.get("axis")
    clean_cache = attrs.get("clean_cache_tensor")
    y_infos = attrs.get("y_infos")
    y_shape = y_infos.get("shape")
    y_ori_shape = y_infos.get("ori_shape")
    n_dim, c1_dim, h_dim, w_dim, c0_dim = y_shape
    axis_dim = list(y_ori_shape)[axis] // cache_depth
    input_shape = input_tensor.shape
    data_size = common_util.get_data_size(input_tensor.dtype)
    vnchw_align = BLOCK_SIZE // data_size
    hw_dim = input_shape[-3] * input_shape[-2]

    x_ub_tensor = tvm.compute(input_shape, lambda *indices: input_tensor(*indices), name="x_ub_tensor")
    # [n, c1, h, w, c0] -> [n, c1, h * w, c0]
    x_reshape_tensor_shape = [*input_shape[:-3], hw_dim,
                              input_shape[-1]]
    x_reshape_tensor = tvm.compute(x_reshape_tensor_shape,
        lambda n, c1, hw, c0: x_ub_tensor(n, c1, hw // w_dim, hw % w_dim, c0),
        name="x_reshape_tensor")
    # [n, c1, h * w, c0] -> [n, c1, c0, h * w]
    x_transpose_tensor_shape = [*x_reshape_tensor_shape[:-2], x_reshape_tensor_shape[-1], x_reshape_tensor_shape[-2]]
    x_transpose_tensor = tvm.compute(x_transpose_tensor_shape,
        lambda n, c1, c0, hw:  x_reshape_tensor(n, c1, hw, c0), name="x_transpose_tensor")

    y_ub_tensor = tvm.compute(y_shape, lambda n, c1, h, w, c0: input_tensor(n + 1, c1, h, w, c0),
                              name="y_ub_tensor")
    # [n, c1, h, w, c0] -> [n, c1, h * w, c0]
    y_reshape_tensor_shape = [*y_shape[:-3], hw_dim, y_shape[-1]]
    y_reshape_tensor = tvm.compute(y_reshape_tensor_shape,
        lambda n, c1, hw, c0: y_ub_tensor(n, c1, hw // w_dim, hw % w_dim, c0),
        name="y_reshape_tensor")
    # [n, c1, h * w, c0] -> [n, c1, c0, h * w]
    y_transpose_tensor_shape = [*y_reshape_tensor_shape[:-2], y_reshape_tensor_shape[-1], y_reshape_tensor_shape[-2]]
    y_transpose_tensor = tvm.compute(y_transpose_tensor_shape,
        lambda n, c1, c0, hw: y_reshape_tensor(n, c1, hw, c0), name="y_transpose_tensor")

    fifo_tensor_shape = [*y_shape[:-3], y_shape[-1], hw_dim]
    fifo_tensor = tvm.compute(fifo_tensor_shape, lambda *indices: fifo_lambda_compute(*indices), name="fifo_tensor")
    # [n, c1, c0, h * w] -> [n, c1, h * w, c0]
    fifo_transpose_tensor_shape = [*fifo_tensor_shape[:-2], fifo_tensor_shape[-1], fifo_tensor_shape[-2]]
    fifo_transpose_tensor = tvm.compute(fifo_transpose_tensor_shape,
        lambda n, c1, hw, c0: fifo_tensor(n, c1, c0, hw), name="fifo_transpose_tensor")
    # [n, c1, h * w, c0] -> [n, c1, h, w, c0]
    fifo_reshape_tensor = tvm.compute(y_shape,
        lambda n, c1, h, w, c0: fifo_transpose_tensor(n, c1, h * w_dim + w, c0),
        name="fwc_res",
        tag="fwc",
        attrs={"not_align_flag": True})
    return fifo_reshape_tensor


def fill_window_cache_5hd_compute(attrs):
    def lambda_compute(n, c1, h, w, c0):
        n_cache, c1_cache, h_cache, w_cache = n_new, c1_new, h_new, w_new = n, c1, h, w
        op_axis = n
        if axis == NCHW_N:
            n_cache, n_new = n_dim + n + axis_dim, n - axis_dim * (cache_depth - 1)
        elif axis == NCHW_C:
            c1_axis_dim = (axis_dim + c0_dim - 1) // c0_dim
            n_cache = n_dim + n * cache_depth + (c1 + c1_axis_dim) // c1_dim
            c1_cache, c1_new = (c1 + c1_axis_dim) % c1_dim, c1 - (c1_axis_dim * (cache_depth - 1))
        elif axis == NCHW_H:
            n_cache = n_dim + n * cache_depth + (c1 * cache_depth + (h + axis_dim) // h_dim) // c1_dim
            c1_cache = (c1 * cache_depth + (h + axis_dim) // h_dim) % c1_dim
            op_axis, h_cache, h_new = h, (h + axis_dim) % h_dim, h - axis_dim * (cache_depth - 1)
        elif axis == NCHW_W:
            n_cache = n_dim + n * cache_depth + (c1 * cache_depth) // c1_dim + (h * cache_depth +
                  (w + axis_dim) // w_dim) // c1_dim // h_dim
            c1_cache = (c1 * cache_depth + ((h * cache_depth + (w + axis_dim) // w_dim) // h_dim)) % c1_dim
            h_cache = (h * cache_depth + (w + axis_dim) // w_dim) % h_dim
            op_axis, w_cache, w_new = w, (w + axis_dim) % w_dim, w - axis_dim * (cache_depth - 1)

        cond = op_axis < axis_dim * (cache_depth - 1)
        if axis == NCHW_C:
            cond = c1 * c0_dim + c0 < axis_dim * (cache_depth - 1)
        return tvm.select(
            clean_cache[0] < tvm.const(1, clean_cache.dtype),
            tvm.select(
                cond,
                input_tensor(n_cache, c1_cache, h_cache, w_cache, c0),
                input_tensor(n_new, c1_new, h_new, w_new, c0)
            ),
            tvm.select(
                cond,
                tvm.const(0, input_tensor.dtype),
                input_tensor(n_new, c1_new, h_new, w_new, c0)
            )
        )

    input_tensor = attrs.get("input_tensor")
    cache_depth = attrs.get("cache_depth")
    axis = attrs.get("axis")
    clean_cache = attrs.get("clean_cache_tensor")
    y_infos = attrs.get("y_infos")
    y_shape = y_infos.get("shape")
    y_ori_shape = y_infos.get("ori_shape")
    n_dim, c1_dim, h_dim, w_dim, c0_dim = input_tensor.shape
    axis_dim = list(y_ori_shape)[axis] // cache_depth
    if axis == 1 and axis_dim * common_util.get_data_size(input_tensor.dtype) % BLOCK_SIZE != 0:
        res = nchw_c_dim_not_align_compute(attrs)
    else:
        res = tvm.compute(
            y_shape,
            lambda *indices: lambda_compute(*indices),
            name="fwc_res",
            tag="fwc"
        )
    return res


@register_operator_compute("FillWindowCache", op_mode="dynamic", support_fusion=True)
def fill_window_cache_compute(x, clean_cache, y, axis, cache_depth, kernel_name="fill_window_cache"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    clean_cache : TVM tensor
        the placeholder of clean_cache
    y : dict
        shape and dtype of output
    axis : a number of int
    cache_depth : a number of int
    kernel_name : str
        kernel name, default value is "fill_window_cache"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    def x_combine_reshape(*indices):
        a1, a2, n1, m1, m0, n0 = indices
        total_address = a1 * a2_axis + a2
        x_shape = x.shape
        batch_shapes = x_shape[:NZ_N1]
        batch_shapes_lens = len(batch_shapes)

        if batch_shapes_lens >= 2:
            index = []
            for i in range(batch_shapes_lens):
                if i != batch_shapes_lens - 1:
                    idx = total_address % batch_shapes[-i - 1]
                    index.append(idx)
                    total_address = total_address // batch_shapes[-i - 1]
                else:
                    index.append(total_address)
            index = index[::-1]
            index.extend([n1, m1, m0, n0])
        elif batch_shapes_lens == 1:
            index = [total_address, n1, m1, m0, n0]
        else:
            index = [total_address * n1_axis + n1, m1, m0, n0]
        return x(*index)
    attrs_dict = {}
    attrs_dict.update(input_tensor=x)
    attrs_dict.update(clean_cache_tensor=clean_cache)
    attrs_dict.update(y_infos=y)
    attrs_dict.update(axis=axis)
    attrs_dict.update(cache_depth=cache_depth)
    data_format = y.get("format")
    if data_format == "NC1HWC0":
        res = fill_window_cache_5hd_compute(attrs_dict)
    else:
        axis = axis - len(y.get("ori_shape")) if axis >= 0 else axis
        x_combine_shape = combine_nz_dims(x.shape, axis)
        a1_axis, a2_axis, n1_axis, *_ = x_combine_shape
        x_size, y_size = int(np.prod(x.shape)), int(np.prod(y.get("shape")))
        data_frames = (x_size + y_size) // x_size
        input_shape = [data_frames * a1_axis, *x_combine_shape[1:]]
        tensor_x_ub = tvm.compute(input_shape, lambda *indices: x_combine_reshape(*indices), name="tensor_x_ub")
        attrs_dict.update(axis=axis)
        attrs_dict.update(input_tensor=tensor_x_ub)
        attrs_dict.update(x_combine_shape=x_combine_shape)
        res = fill_window_cache_nz_compute(attrs_dict)
    return res


@register_operator("FillWindowCache")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def fill_window_cache(x, clean_cache, y, axis, cache_depth, kernel_name="fill_window_cache"):
    """
    do  fill_window_cache operation

    Parameters:
    ----------
    x : dict
        shape and dtype of input
    clean_cache : dict
        shape and dtype of clean_cache
    y : dict
        shape and dtype of output
    axis : a number of int
    cache_depth : a number of int
    kernel_name : str
        kernel name, default value is "fill_window_cache"

    Returns
    -------
    None
    """
    check_params(x, y, axis, cache_depth)
    x_shape = x.get("shape")
    x_dtype = x.get("dtype")
    clean_cache_shape = clean_cache.get("shape")
    clean_cache_dtype = clean_cache.get("dtype")
    input_tensor = tvm.placeholder(x_shape, dtype=x_dtype, name="x_tensor")
    clean_cache = tvm.placeholder(clean_cache_shape, dtype=clean_cache_dtype, name="clean_cache_tensor")
    res = fill_window_cache_compute(input_tensor, clean_cache, y, axis, cache_depth, kernel_name)
