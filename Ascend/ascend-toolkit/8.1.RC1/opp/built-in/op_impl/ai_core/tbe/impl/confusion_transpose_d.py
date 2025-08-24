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
confusion_transpose_d
"""
from copy import deepcopy
from collections import defaultdict
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_select_op_base
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.transpose_d import transpose_d
from impl import constant_util as constant


# 'pylint: disable=too-many-locals,too-many-arguments,invalid-name,unused-argument
# 'pylint: disable=simplifiable-if-statement,no-else-return


def _prod(input_shape):
    """
    Calculate the product of all elements
    """
    prod = 1
    for i in input_shape:
        prod *= i

    return prod


def _reshape_frac(shape_in, shape_out):
    """
    This algorithm can only used in the case
    where out shape can be obtained by spliting and merging input shape.
    For example:
    input shape = [(a*b),(c*d*e),f,(g*h)]
    output shape = [a,(b*c),d,e,(f*g),h)]
    fractal shape = [a,b,c,d,e,f,g,h]

    The last dimension of input and output shape
    should be the multiples of w0(usually 16)

    params:
        shape_in:  input shape of Reshape op
        shape_out: output shape of Reshape op
        w0:  minimum length in width for fractal format

    return:
        a splited shape which can be used to interpret fractal Reshape
        as a Reshape + Transpose calculation
    """
    if _prod(shape_in) != _prod(shape_out):
        error_manager_vector.raise_err_inputs_shape_not_equal("confusion_transpose_d", "_prod(shape_in)",
                                                              "_prod(shape_out)", _prod(shape_in),
                                                              _prod(shape_out), _prod(shape_out))
    idx_in = len(shape_in) - 1
    idx_out = len(shape_out) - 1
    shape_frac = []
    res_in = shape_in[idx_in]
    res_out = shape_out[idx_out]
    while idx_in >= 0 or idx_out >= 0:
        frac_elmt = min(res_in, res_out)
        shape_frac.insert(0, frac_elmt)
        if res_in % frac_elmt != 0 or res_out % frac_elmt != 0:
            param_name = "res_in " + "%" + "frac_elmt, res_out" + "%" + "frac_elmt"
            error_manager_vector.raise_err_input_value_invalid("confusion_transpose_d", param_name,
                                                               "0", res_in % frac_elmt + ", " + res_out % frac_elmt)
        res_in //= frac_elmt
        res_out //= frac_elmt
        if res_in <= 1:
            idx_in -= 1
            if idx_in >= 0:
                res_in = shape_in[idx_in]
        if res_out <= 1:
            idx_out -= 1
            if idx_out >= 0:
                res_out = shape_out[idx_out]

    return shape_frac


def _split_shape_in(input_shape, axis_list):
    """
    split the dimension in axis_list to 16

    params:
        input_shape: the shape to be processed
        axis_list: dimensions
    return:
        a splited shape
    """
    split_axis = []
    for index in axis_list:
        if len(input_shape[index]) == 1:
            input_shape[index].insert(0, input_shape[index][-1] // 16)
            input_shape[index][1] = 16
            split_axis.append(index + len(input_shape))
        else:
            if input_shape[index][-1] > 16:
                input_shape[index].insert(len(input_shape[index]) - 1,
                                          input_shape[index][-1] // 16)
                input_shape[index][-1] = 16
                split_axis.append(index + len(input_shape))
            else:
                pass

    return input_shape, split_axis


def _merge_frac(shape_tensor, frac_shape):
    """
    merge a splited shape accoding to shape_tensor

    params:
        shape_tensor:  a merged shape, should merge frac_shape accoding to this
        frac_shape: a splited shape to be merged
    return:
        merged shape
    """
    if len(shape_tensor) == len(frac_shape):
        frac_shape = [[i] for i in frac_shape]
        return frac_shape
    if 1 not in shape_tensor:
        return _merge_frac_1(shape_tensor, frac_shape)
    return _merge_frac_2(shape_tensor, frac_shape)


def _merge_frac_1(shape_tensor, frac_shape):
    """
    merge a splited shape accoding to shape_tensor
    1 not in shape_tensor
    params:
        frac_shape: a splited shape to be merged
        shape_tensor:  a merged shape, should merge frac_shape accoding to this
    return:
        merged shape
    """
    tensor_idx = len(shape_tensor) - 1
    frac_idx = len(frac_shape) - 1
    merged_list = []
    while tensor_idx >= 0:
        out_elmt = shape_tensor[tensor_idx]
        shape_list = [frac_shape[frac_idx]]
        list_product = frac_shape[frac_idx]
        while list_product < out_elmt or frac_shape[frac_idx-1] == 1:
            frac_idx -= 1
            shape_list.insert(0, frac_shape[frac_idx])
            list_product *= frac_shape[frac_idx]

        merged_list.insert(0, shape_list)
        tensor_idx -= 1
        frac_idx -= 1
    while frac_idx >= 0:
        merged_list.insert(0, [frac_shape[frac_idx]])
        frac_idx -= 1
    return merged_list


def _merge_frac_2(shape_tensor, frac_shape):
    """
    merge a splited shape accoding to shape_tensor
    1 in shape_tensor
    params:
        frac_shape: a splited shape to be merged
        shape_tensor:  a merged shape, should merge frac_shape accoding to this
    return:
        merged shape
    """
    tensor_idx = len(shape_tensor) - 1
    frac_idx = len(frac_shape) - 1
    merged_list = []
    while tensor_idx >= 0:
        out_elmt = shape_tensor[tensor_idx]
        shape_list = [frac_shape[frac_idx]]
        list_product = frac_shape[frac_idx]
        while list_product < out_elmt or frac_shape[frac_idx-1] == 1:
            if frac_shape[frac_idx] == 1 and shape_tensor[tensor_idx] == 1:
                break
            if frac_shape[:frac_idx].count(1) == \
                    shape_tensor[:tensor_idx].count(1):
                if list_product == out_elmt:
                    break
            if shape_tensor[tensor_idx-1] == 1 and frac_shape[frac_idx-1] == 1:
                if frac_shape[frac_idx] == 1:
                    break
                if list_product == out_elmt:
                    if (frac_idx-2) >= 0 and frac_shape[frac_idx-2] == 1:
                        frac_idx -= 1
                        shape_list.insert(0, frac_shape[frac_idx])
                        list_product *= frac_shape[frac_idx]
                    break
            frac_idx -= 1
            shape_list.insert(0, frac_shape[frac_idx])
            list_product *= frac_shape[frac_idx]
            if frac_idx == 0:
                break

        merged_list.insert(0, shape_list)
        tensor_idx -= 1
        frac_idx -= 1
    while frac_idx >= 0:
        merged_list.insert(0, [frac_shape[frac_idx]])
        frac_idx -= 1
    return merged_list


def _perm_nz_to_orig(merged_orig, split_idx):
    """
    calculate the perm used in Nz to ND format transpose

    params:
        merged_orig: the shape to be transposed
        split_idx: the index of spliting the last dimention
    return:
        1. the perm used in Nz to ND format transpose
        2. the shape in Nz format
    """
    perm_to_orig = []
    frac_nz_in = []
    cnt = 0
    for _, i in enumerate(range(len(merged_orig) - 2)):
        frac = merged_orig[i]
        for _, j in enumerate(range(len(frac))):
            perm_to_orig += [cnt]
            cnt += 1
            frac_nz_in += [frac[j]]

    #perm for H in W1HW0
    cnt_start = cnt
    for _, i in enumerate(range(len(merged_orig[-2]))):
        perm_to_orig += [cnt+len(merged_orig[-1])+split_idx]
        cnt += 1
        frac_nz_in += [merged_orig[-2][i]]

    #perm for W1 in W1HW0
    frac = merged_orig[-1]
    for _, i in enumerate(range(len(merged_orig[-1])+split_idx)):
        perm_to_orig += [cnt_start]
        cnt_start += 1
        cnt += 1
        frac_nz_in.insert(cnt_start-1, frac[i])

    for _, i in enumerate(range(len(merged_orig[-1])+split_idx,
                                len(merged_orig[-1]))):
        perm_to_orig += [cnt]
        cnt += 1
        frac_nz_in += [frac[i]]

    perm_to_orig = [int(i) for i in perm_to_orig]
    frac_nz_in = [int(i) for i in frac_nz_in]

    return perm_to_orig, frac_nz_in


def _merge_perm(src_list, dst_list):
    """
    merge a perm accoding to a given shape

    params:
        src_list: the perm to be merged
        dst_list: the given shape
    return:
        merged perm
    """
    len_list = []
    result_list = []
    for _, shape_list in enumerate(dst_list):
        len_list.append(len(shape_list))

    for index in len_list:
        list_tmp = src_list[:index]
        result_list.append(list_tmp)
        del src_list[:index]

    return result_list


def _flat_perm(merged_perm):
    """
    flatten a merged perm

    params:
        merged_perm: the perm to be flattened
    return:
        flat perm
    """
    flaten_perm = []
    for _, i in enumerate(range(len(merged_perm))):
        for _, j in enumerate(range(len(merged_perm[i]))):
            flaten_perm.append(merged_perm[i][j])
    return flaten_perm


def _shape_after_transpose(input_shape, trans_perm):
    """
    Transpose a merged_perm according to the given perm.
    The result is a new perm that fuses two transpose operations
    by fusing the two perms

    params:
        input_shape: the perm to be transposed
        trans_perm: the given perm
    return:
        transposed perm
    """
    if len(input_shape) != len(trans_perm):
        error_manager_vector.raise_err_inputs_shape_not_equal("confusion_transpose_d", "input_shape", "trans_perm",
                                                              input_shape, trans_perm, trans_perm)
    transposed_merged_perm = []
    for _, i in enumerate(range(len(trans_perm))):
        transposed_merged_perm.append(input_shape[trans_perm[i]])
    return transposed_merged_perm


def _perm_orig_to_nz(perm_merged, split_idx):
    """
    Only non-last-dimension transpose is considered in this situation.
    If the last dimension is transposed, it can only output a ND format output
    rather than a fractal one after fusing.

    Calculate the perm used when a tensor is transposed from ND to Nz format

    params:
        perm_merged: the perm to be transposed
        split_idx: the idx indicating the location to split the last dimension
    return:
        transposed perm
    """
    perm_to_nz = []

    for _, i in enumerate(range(len(perm_merged) - 2)):
        perm_to_nz += perm_merged[i]
    perm_to_nz += perm_merged[len(perm_merged) - 1][:split_idx]
    perm_to_nz += perm_merged[len(perm_merged) - 2]
    perm_to_nz += perm_merged[len(perm_merged) - 1][split_idx:]

    return perm_to_nz


def _shape_before_transpose(merged_frac, transpose_perm):
    """
    In a transpose before reshape case, calculate the shape before transpose.

    params:
        merged_frac: merged input shape of Reshape op
        transpose_perm: perm of the transpose opertation
    return:
        the shape before transpose
    """
    shape_before = []
    if len(merged_frac) != len(transpose_perm):
        error_manager_vector.raise_err_inputs_shape_not_equal("confusion_transpose_d", "merged_frac", "transpose_perm",
                                                              merged_frac, transpose_perm, transpose_perm)
    for _, i in enumerate(range(len(merged_frac))):
        shape_before.append([])

    for _, i in enumerate(range(len(transpose_perm))):
        shape_before[transpose_perm[i]] = merged_frac[i]

    return shape_before


def _reshape_transpose(transpose_perm, reshape_in, reshape_out):
    """
    The case where Reshape is before a transpose.

    params:
        reshape_in: input shape of Reshape op
        reshape_out: output shape of Reshape op
        transpose_perm: perm of Transpose op
    return:
        final input shape and perm of Transpose op
    """
    trans_out = _shape_after_transpose(reshape_out, transpose_perm)
    idx = len(trans_out)
    trans_out_merge = [[trans_out[i]] for _, i in enumerate(range(idx))]
    trans_out_split, _ = _split_shape_in(trans_out_merge, [-1, -2])

    merged_frac = _shape_before_transpose(trans_out_split, transpose_perm)
    merged_frac_flat = _flat_perm(merged_frac)
    frac_res = _reshape_frac(merged_frac_flat, reshape_in)
    merged_frac_in = _merge_frac(reshape_in, frac_res)
    merged_frac_in_split, _ = _split_shape_in(merged_frac_in, [-1, -2])
    merged_frac_in_split_flat = _flat_perm(merged_frac_in_split)

    merged_frac_split = _merge_frac(reshape_out, merged_frac_in_split_flat)

    nz_nd_perm, frac_nz_in = _perm_nz_to_orig(merged_frac_in_split, -1)

    perm_merged = _merge_perm(deepcopy(nz_nd_perm), merged_frac_split)
    trans_perm = _shape_after_transpose(perm_merged, transpose_perm)
    final_perm = _perm_orig_to_nz(trans_perm, -1)
    merged_frac_out_split = _shape_after_transpose(merged_frac_split, transpose_perm)

    res = [frac_nz_in, final_perm, merged_frac_out_split, nz_nd_perm, merged_frac_in_split]
    return res


def _transpose_reshape(transpose_perm, reshape_in, reshape_out):
    """
    The case where Reshape is after a transpose.

    params:
        reshape_in: input shape of Reshape op
        reshape_out: output shape of Reshape op
        transpose_perm: perm of Transpose op
    return:
        final input shape and perm of Transpose op
    """
    frac_res = _reshape_frac(reshape_out, reshape_in)
    merged_frac_out = _merge_frac(reshape_out, frac_res)
    merged_frac_out, _ = _split_shape_in(merged_frac_out, [-1, -2])
    merged_frac_out = _flat_perm(merged_frac_out)
    merged_frac_in = _merge_frac(reshape_in, merged_frac_out)
    frac_merged = _shape_before_transpose(merged_frac_in, transpose_perm)

    trans_in, _ = _split_shape_in(frac_merged, [-1, -2])
    merged_frac_in_split = _shape_after_transpose(trans_in, transpose_perm)
    merged_frac_in_split = _flat_perm(merged_frac_in_split)

    merged_frac_out_splt = _merge_frac(reshape_out, merged_frac_in_split)

    perm, frac_nz_in = _perm_nz_to_orig(trans_in, -1)
    perm_trans_in = _merge_perm(deepcopy(perm), trans_in)
    perm_transpose = _shape_after_transpose(perm_trans_in, transpose_perm)
    perm_transpose = _flat_perm(perm_transpose)

    perm_merged = _merge_perm(perm_transpose, merged_frac_out_splt)
    perm_final = _perm_orig_to_nz(perm_merged, -1)

    res = [frac_nz_in, perm_final, merged_frac_out_splt, perm, trans_in]
    return res


def _division_sixteen(shape):
    if len(shape) < 2:
        if shape[-1] == 0:
            error_detail = "value of shape is illegal, shape[-1] == 0"
            error_manager_vector.raise_err_specific_reson("confusion_transpose_d", error_detail)
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        error_detail = "value of shape is illegal, shape[-1]:%s, shape[-2]:%s" % (shape[-1], shape[-2])
        error_manager_vector.raise_err_specific_reson("confusion_transpose_d", error_detail)

    if shape[-1] % constant.SIZE_SIXTEEN == 0 and shape[-2] % constant.SIZE_SIXTEEN == 0:
        return True
    else:
        return False


def check_shape_4_3(shape_a, shape_b):
    flag_check = (shape_a[1] * shape_a[2] == shape_b[1] and shape_a[3] == shape_b[2])

    return flag_check


def check_shape_2_4(shape_a, shape_b):
    flag_check = (shape_b[0] * shape_b[1] == shape_a[0] and shape_b[2] * shape_b[3] == shape_a[1])

    return flag_check


def check_reshape(in_shape, out_shape):
    if len(in_shape) == 2 and len(out_shape) == 4 and check_shape_2_4(in_shape, out_shape):
        return True
    if len(in_shape) == 4 and len(out_shape) == 2 and check_shape_2_4(out_shape, in_shape):
        return True
    if (len(in_shape) == 4 and len(out_shape) == 3 and in_shape[0] == out_shape[0] and
        check_shape_4_3(in_shape, out_shape)):
        return True
    if (len(in_shape) == 3 and len(out_shape) == 4 and out_shape[0] == in_shape[0] and
        check_shape_4_3(out_shape, in_shape)):
        return True

    return False


def _condition(x, perm, shape, transpose_first):
    shape_x = shape_util.scalar2tensor_one(x.get("ori_shape"))

    if transpose_first:
        shape_reshapein = _shape_after_transpose(shape_x, perm)
    else:
        shape_reshapein = shape_x
        if not _division_sixteen(_shape_after_transpose(shape, perm)):
            return False

    if (len(perm) == 4 and _division_sixteen(shape_x) and perm[3] == 3):
        return check_reshape(shape_reshapein, shape)

    return False


def _get_nz_split_shape_and_perm(x, perm, shape, transpose_first):
    """
    Get NZ split shape and perm.
    For example:
    Input:
    `x = {"shape": (32, 4, 48, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 768, 64),`
         `"ori_format": "NHWC"}`
    `perm = [2, 0, 1, 3]`
    shape = [32, 48, 16, 64]`
    `transpose_first = False`
    `y = {"shape": (16, 32, 4, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 32, 48, 64),`
         `"ori_format": "NHWC"}`


    Output:
    `frac_nz_in = [32, 4, 3, 16, 16, 16]`
    reshape the dim of 48 to 3 and 16 in shape(NZ) of x.

    `merged_frac_in_split = [[32], [3, 16, 16], [4, 16]]`
    reshape the dim of 768 into [3, 16, 16], and 64 to [4, 16], in ori_shape(ND) of x.

    `nz_nd_perm = [0, 2, 3, 4, 1, 5]`
    transpose the NZ shape of input: [32, 4, 3, 16, 16, 16] to ND shape: [32, 3, 16, 16, 4, 16]

    `merged_frac_out_split = [[16], [32], [3, 16], [4, 16]]`
    reshape the dim of 48 into [3, 16], and 64 to [4, 16], in ori_shape(ND) of y.

    `final_perm = [4, 0, 1, 2, 3, 5]`
    transpose the NZ shape of input: [32, 4, 3, 16, 16, 16] to the NZ shape of output: [16, 32, 4, 3, 16, 16].
    """
    input_shape = x.get("shape")
    nd_shape = []
    nd_shape += \
        [input_shape[i] for _, i in enumerate(range(len(input_shape)-4))]
    nd_shape += [input_shape[-3] * input_shape[-2]]
    nd_shape += [input_shape[-1] * input_shape[-4]]
    if transpose_first:
        transpose_in = nd_shape
        reshape_in = _shape_after_transpose(transpose_in, perm)
        frac_nz_in, final_perm, merged_frac_out_split, nz_nd_perm, \
            merged_frac_in_split = _transpose_reshape(perm, reshape_in, shape)
    else:
        reshape_in = nd_shape
        frac_nz_in, final_perm, merged_frac_out_split, nz_nd_perm, \
            merged_frac_in_split = _reshape_transpose(perm, reshape_in, shape)
    res = [frac_nz_in, final_perm, merged_frac_out_split, nz_nd_perm, merged_frac_in_split]
    return res


def _get_dim_index(perm, dim_val):
    """
    Get the index of dim in perm which value is equal dim_val.
    """
    for idx, val in enumerate(perm):
        if val == dim_val:
            return idx

    return 0


def _get_output_slice_dim_of_nd(reshape_in, reshape_out, input_slice_dim):
    """
    Get slice dim of ND shape.
    For example:
    `reshape_in = [B, E, A, D, C]`
    `reshape_out = [B1, F1, E1, 1, A1, A2, D1, C1, W1]`
    `input_slice_dim = 2; means slice A dim in reshape_in.`
    `If B*E = B1*F1*E1*1 and A % A1 = 0, the slice_dim of A1 in reshape_out is 4; else -1.`
    """
    slice_dim = 0
    if input_slice_dim > 0:
        prod_shape_before_slice_dim = _prod(reshape_in[0:input_slice_dim])
        prod_reshape_out = 1
        for idx, val in enumerate(reshape_out):
            prod_reshape_out = prod_reshape_out * val
            if prod_reshape_out > prod_shape_before_slice_dim:
                if _prod(reshape_out[0:idx]) == prod_shape_before_slice_dim:
                    slice_dim = idx
                    break
                return -1

    if reshape_out[slice_dim] != 0 and reshape_in[input_slice_dim] % reshape_out[slice_dim] == 0:
        return slice_dim

    return -1


def _merge_nd_2_nz(nd_shape):
    """
    Change ND shape to NZ shape.
    For example:
    `nd_shape = [[B1, F1, E1], [A1, A2], [D1, C1, W1]]`
    `nz_shape = [[B1, F1, E1], [D1, C1], A1, A2, W1]`
    """
    nz_shape = []
    for _, i in enumerate(range(len(nd_shape) - 2)):
        nz_shape.append(nd_shape[i])

    nz_shape.append(nd_shape[-1][0:-1])
    nz_shape.append(nd_shape[-2][0:-1])
    nz_shape.append(nd_shape[-2][-1])
    nz_shape.append(nd_shape[-1][-1])

    return nz_shape


def _get_slice_dim_of_nz(nd_shape, perm_idx):
    """
    Get slice dim of NZ shape.
    For example:
    `nd_shape = [[B1, F1, E1], [A1, A2], [D1, C1, W1]]`
    Change nd_shape to nz_shape:
    `nz_shape = [[B1, F1, E1], [D1, C1], A1, A2, W1]`

    `perm_idx = 5, means slice the dim of A1 in nz_shape.The perm_idx of B1 in nz_shape is 0, F1 is 1, etc.`

    Then slice_dim of A1 in nz_shape is 2.The first dim of nz_shape is [B1, F1, E1], etc.
    """
    slice_dim = -1
    nz_shape = _merge_nd_2_nz(nd_shape)
    dim_num = 0
    for _, val in enumerate(range(len(nz_shape) - 2)):
        if perm_idx == dim_num:
            slice_dim = val
            break
        else:
            dim_num += len(nz_shape[val])

    return slice_dim


# 'pylint: disable = unused-argument,too-many-arguments,too-many-locals
def get_op_support_info(x, y, perm, shape, transpose_first, kernel_name="confusion_transpose_d"):
    """
    get_op_support_info
    """
    input_format = x.get("format").upper()
    input_shape = x.get("shape")

    axis_reduce_list = []
    axis_split_matrix = []

    if input_format == "FRACTAL_NZ":
        _, final_perm, merged_frac_out_split, nz_nd_perm, merged_frac_in_split = \
            _get_nz_split_shape_and_perm(x, perm, shape, transpose_first)
        input_slice_dim = _get_slice_dim_of_nz(merged_frac_in_split, nz_nd_perm[0])
        output_slice_dim = _get_slice_dim_of_nz(merged_frac_out_split, _get_dim_index(final_perm, nz_nd_perm[0]))
        if input_slice_dim != -1 and output_slice_dim != -1:
            split_0 = [SplitInput([0, [input_slice_dim], [-1], [-1]]), SplitOutput([0, [output_slice_dim]])]
            axis_split_matrix.append(split_0)
    else:
        perm_index = _get_dim_index(perm, 0)
        output_slice_dim = 0
        reshape_slice_dim = -1
        if transpose_first:
            reshape_in = _shape_after_transpose(input_shape, perm)
            reshape_slice_dim = _get_output_slice_dim_of_nd(reshape_in, shape, perm_index)
            output_slice_dim = reshape_slice_dim
        else:
            reshape_slice_dim = _get_output_slice_dim_of_nd(input_shape, shape, 0)
            output_slice_dim = perm_index
        if reshape_slice_dim != -1:
            split_0 = [SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [output_slice_dim]])]
            axis_split_matrix.append(split_0)

    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def op_select_format(x, y, perm, shape, transpose_first,
                     kernel_name="confusion_transpose_d"):
    """
    _division_sixteen : judge whether the last two dimensions are divided by 16
    scalar2tensor_one : convert scalar to tensor
    """
    condition = _condition(x, perm, shape, transpose_first)

    if condition:
        # NZ+ND
        input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                               datatype="float16,float,int8,int16,int32,int64,"
                                                        "uint8,uint16,uint32,uint64,"
                                                        "float16,float,int8,int16,int32,int64,"
                                                        "uint8,uint16,uint32,uint64,bool,bool,bfloat16,bfloat16",
                                               format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,"
                                                      "FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,"
                                                      "FRACTAL_NZ,FRACTAL_NZ,"
                                                      "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,FRACTAL_NZ,ND,FRACTAL_NZ,ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,float,int8,int16,int32,int64,"
                                                         "uint8,uint16,uint32,uint64,"
                                                         "float16,float,int8,int16,int32,int64,"
                                                         "uint8,uint16,uint32,uint64,bool,bool,bfloat16,bfloat16",
                                                format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,"
                                                       "FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,"
                                                       "FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,"
                                                       "FRACTAL_NZ,"
                                                       "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,FRACTAL_NZ,ND,FRACTAL_NZ,ND")
    else:
        # ND+ND
        input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                               datatype="float16,float,int8,int16,int32,int64,"
                                                        "uint8,uint16,uint32,uint64,bool,bfloat16",
                                               format="ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,float,int8,int16,int32,int64,"
                                                         "uint8,uint16,uint32,uint64,bool,bfloat16",
                                                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND")

    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _special_situation_check(para_dict, seq_length, batch_size, support_core_num):
    """
    check if it is case of fusion with matmul when special situation
    """
    trans_a, trans_b = para_dict.get("trans_a"), para_dict.get("trans_b")
    if not trans_a or not trans_b:
        seq_length, batch_size = [], []
        if tbe_platform.get_soc_spec("CORE_NUM") not in support_core_num:
            seq_length = [128, 256]
            batch_size = [60, 30]
    return seq_length, batch_size


# 'pylint: disable=too-many-boolean-expressions,too-many-return-statements,too-many-branches
def _is_matmul_fusion_case(y, perm, shape, transpose_first, para_dict):
    """
    check if it is case of fusion with matmul
    """
    soc_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    valid = isinstance(y, dict) and (isinstance(perm, (list, tuple))) and \
            (isinstance(shape, (list, tuple))) and list(perm) == [0, 2, 1, 3]
    if not valid:
        return False
    if soc_version == "Ascend910":
        if not transpose_first:
            seq_length = [128, 128, 224, 256]
            batch_size = [1, 60, 30, 30]
            support_core_num = [32, ]
            if tbe_platform.get_soc_spec("CORE_NUM") in support_core_num:
                seq_length = seq_length + [160, 192, 224, 256]
                batch_size = batch_size + [1] * 4
            seq_length, batch_size = _special_situation_check(para_dict, seq_length, batch_size, support_core_num)
            supported_shape = [[batch, seq] for batch, seq in zip(batch_size, seq_length)]
            supported_shape = supported_shape + [[i, 128] for i in range(16, 176, 16)] + \
                            [[i, 512] for i in range(4, 28, 4)]
            supported_shape_temp = defaultdict(list)
            for value in supported_shape:
                supported_shape_temp[value[1]].append(value[0])
            is_support_fusion = list(shape)[1] in supported_shape_temp.keys() and \
                                list(shape)[2:] == [16, 64] and \
                                list(y.get("shape"))[1:] == [16, 4, list(shape)[1] // constant.C0_SIZE, \
                                constant.C0_SIZE, constant.C0_SIZE] and \
                                y.get("shape")[0] in supported_shape_temp[list(shape)[1]] and \
                                y.get("shape")[0] == shape[0]
            if is_support_fusion:
                return True
        else:
            batch_supported_block = [i * 128 for i in range(1, 11)]
            batch_supported = [i * 16 for i in batch_supported_block]
            if y.get("shape")[0] == 64 and list(y.get("shape"))[2:] == [16, 16] and shape[1] == 1024 and \
                y.get("shape")[1] in batch_supported_block and shape[0] in batch_supported:
                return True
    elif soc_version == "Ascend310P":
        if not transpose_first:
            batch_supported = [1, 8, 16, 32, 64]
            if list(y.get("shape"))[1:] == [12, 4, 8, 16, 16] and list(shape)[1:] ==  [128, 12, 64] and \
                y.get("shape")[0] in batch_supported and shape[0] in batch_supported:
                return True
        else:
            batch_supported_block = [8, 32, 64, 128, 256, 512]
            batch_supported = [i * 16 for i in batch_supported_block]
            if y.get("shape")[0] == 48 and list(y.get("shape"))[2:] == [16, 16] and shape[1] == 768 and \
                y.get("shape")[1] in batch_supported_block and shape[0] in batch_supported:
                return True
    else:
        return False


@register_operator_compute("confusion_transpose_d", op_mode="static", support_fusion=True)
def confusion_transpose_d_compute(x, y, perm, shape, transpose_first,
                                  kernel_name="confusion_transpose_d"):
    """
    compute for matmul + confusion_transpose_d fusion
    """
    para_dict = getattr(x, "para_dict")
    if _is_matmul_fusion_case(y, perm, shape, transpose_first, para_dict):
        tensor_a = getattr(x, "tensor_a")
        tensor_b = getattr(x, "tensor_b")
        para_dict["confusion_transpose"] = True
        x = tbe.gemm(tensor_a=tensor_a, tensor_b=tensor_b, para_dict=para_dict)
        setattr(x, "matmul_with_transpose", True)
        setattr(x, "transpose_shape", y.get("shape"))
    else:
        error_detail = "This case does not support fusion with matmul now."
        error_manager_vector.raise_err_specific_reson("confusion_transpose_d", error_detail)

    return x


def confusion_transpose_d_dsl_compute(x, perm, y, kernel_name):
    """
    algorithm: confusion_transpose_d
    calculating: permute the dimensions according to new_perm and new_shape

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output
    perm : list or tuple
        permutation of the dimension of tensor
    kernel_name : str
        kernel name, default value is "confusion_transpose_d"

    Returns
    -------
    res : placeholder and res
    """
    res = tbe.transpose(x, perm)
    return res


def confusion_transpose_d_dsl(x, y, perm, shape, transpose_first,
                              kernel_name="confusion_transpose_d"):
    """
    confusion_transpose_d interface for dsl
    """
    input_dtype = x.get("dtype").lower()
    x["range"] = [[i, i] for i in x["shape"]]
    extra_params = {"axes": perm}
    ins = classify([x], "transpose", extra_params)
    schedules, tensors = [], []
    for (input_x, perm) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([input_x], "transpose")
            data_x = tvm.placeholder(x_shape, name="data_x", dtype=input_dtype)
            res = confusion_transpose_d_dsl_compute(data_x, perm, y, kernel_name)
            tensors.append([data_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_BOOL, para_check.KERNEL_NAME)
def confusion_transpose_d(x, y, perm, shape, transpose_first,
                          kernel_name="confusion_transpose_d"):
    """
    algorithm: confusion transpose
    calculating: permute the dimensions according to new_perm and new_shape

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output
    perm : list or tuple
        permutation of the dimension of tensor
    shape : list or tuple
        shape of reshape out
    transpose_first : bool
        transpode node is first or not
    kernel_name : str
        kernel name, default value is "confusion_transpose_d"

    Returns
    -------
    None
    """
    input_shape = x.get("shape")
    input_ori_shape = x.get("ori_shape")
    input_dtype = x.get("dtype").lower()
    input_format = x.get("format")
    input_ori_format = x.get("ori_format")

    para_check.check_shape(input_shape, param_name="x")

    check_list = ("int8", "int16", "int32", "int64", "uint8",
                  "uint16", "uint32", "uint64", "float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="x")

    if input_format == "FRACTAL_NZ":
        final_shape, final_perm, _, _, _ = _get_nz_split_shape_and_perm(x, perm, shape, transpose_first)
        x["shape"] = final_shape
        perm = final_perm
    else:
        if not transpose_first:
            x["shape"] = shape

    if tbe_platform.api_check_support("tik.vcopy") or tbe_platform.api_check_support("tik.vbi"):
        context = tbe_context.op_context.get_context()
        if context is not None:
            context.set_op_mode("static")
            confusion_transpose_d_dsl(x, y, perm, shape, transpose_first, kernel_name)
        else:
            with tbe_context.op_context.OpContext("static"):
                confusion_transpose_d_dsl(x, y, perm, shape, transpose_first, kernel_name)
        x.pop("range")  # in order to enable pattern fusion
    else:
        x_dict = {"shape": input_shape, "ori_shape": input_ori_shape,
                  "format": input_format, "ori_format": input_ori_format,
                  "dtype": input_dtype}
        transpose_d(x, x_dict, perm, kernel_name)
    tbe_platform.fusion_manager.set_current_op_pattern("confusiontranspose")
