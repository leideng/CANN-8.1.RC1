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
concat_offset_d
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant
    """
    # 256B can store up to 64 numbers when the data is int32 type
    NUM64 = 64
    # const vlaue 2
    VALUE_TWO = 2
    # The maximum value of the mask corresponding to 8 blocks
    MAX_MASK8 = 255


# 'pylint: disable = unused-argument
# 'pylint: disable=invalid-name,useless-object-inheritance,too-few-public-methods
def get_op_support_info(x, y, concat_dim, kernel_name="concat_offset_d"):
    """
    get_op_support_info
    """
    axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=locally-disabled,unused-argument, invalid-name
@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.DYNAMIC_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def concat_offset_d(x, y, concat_dim, kernel_name="concat_offset_d"):
    """
    Compute the concat offset of the input tensor along `concat_dim`.

    Parameters
    ----------
    concat_dim: a number of int32, The dimension along which to concatenate,
                must be in the range [-rank(shape), rank(shape))
    x: list of dict, dict include shape and dtype, dtype must be in ('int32')
    y: list of dict, dict include shape and dtype, dtype must be in ('int32')
    kernel_name: kernel name

    Returns
    -------
    concat_offset_d_compute.compute(): the result of compute

    """
    dict_num = len(x)
    input0_rank = x[0].get("shape")[0]
    if concat_dim < 0:
        concat_dim = input0_rank + concat_dim

    concat_offset_d_check(x, concat_dim, input0_rank, dict_num, kernel_name)
    concat_offset_d_compute = ConcatOffsetDCompute(concat_dim,
                                                   input0_rank, dict_num,
                                                   kernel_name)
    return concat_offset_d_compute.compute()


def concat_offset_d_check(x, concat_dim, input0_rank, dict_num, kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error
    Parameters
    ----------
    x: list of tensor
    concat_dim: The dimension along which to concatenate,
                must be in the range [-rank(shape), rank(shape))
    input0_rank: The rank of the first input shape in the list
    dict_num: The number of tensor in the list
    kernel_name: kernel_name
    Returns
    -------
    None
    """

    if dict_num < 2:
        rule_desc = "The number of elements in the list should be no less than two"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "x", dict_num)
    for i in range(0, dict_num):
        shape_input = x[i].get("shape")
        if shape_input[0] != input0_rank:
            error_detail = "input_%d : should contain %d elements,but got %d" % (i, input0_rank, shape_input[0])
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
        elif shape_input[0] > 8:
            error_detail = "the shape of input_%d should be not bigger than 8" % i
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
        elif concat_dim >= shape_input[0]:
            rule_desc = "Concat dim should be less than input0_rank,the input0_rank is %d" % input0_rank
            error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "concat_dim", concat_dim)
        elif concat_dim < 0:
            concat_dim = concat_dim - input0_rank
            rule_desc = "Concat dim should be larger or equal to 0,the input0_rank is %d" % input0_rank
            error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "concat_dim", concat_dim)
        para_check.check_shape(shape_input, max_rank=1, param_name="x")
        para_check.check_dtype(x[i].get("dtype").lower(), ("int32",), param_name="x")


class ConcatOffsetDCompute(object):
    """
    ConcatOffsetDCompute
    """
    def __init__(self, concat_dim, input0_rank, dict_num, kernel_name):
        """
        init the input param

        Parameters
        ----------
        input0_rank: The rank of the first input shape in the list
        dict_num: the number of tensor
        kernel_name: kernel name

        """
        self.concat_dim = concat_dim
        self.kernel_name = kernel_name
        self.input0_rank = input0_rank
        self.dict_num = dict_num
        self.dtype = "int32"

    def compute(self):
        """
        describe the concat_offset calculation process

        Returns
        -------
        tik_instance: the instance of tik

        """
        data_input = []
        data_output = []
        tik_instance = tik.Tik()
        cdim_mask = Constant.MAX_MASK8 - pow(Constant.VALUE_TWO, int(self.concat_dim))
        for i in range(self.dict_num):
            data_input.append(tik_instance.Tensor(
                self.dtype, [self.input0_rank],
                name="".join(["data_input", str(i)]), scope=tik.scope_gm))
            data_output.append(tik_instance.Tensor(
                self.dtype, [self.input0_rank],
                name="".join(["data_output", str(i)]), scope=tik.scope_gm))
        data_row1_ub = tik_instance.Tensor(self.dtype, [Constant.NUM64],
                                           name="data_row1_ub",
                                           scope=tik.scope_ubuf)
        data_row2_ub = tik_instance.Tensor(self.dtype, [Constant.NUM64],
                                           name="data_row2_ub",
                                           scope=tik.scope_ubuf)
        tik_instance.vector_dup(Constant.NUM64, data_row1_ub, 0, 1, 1, 1)

        for m in range(self.dict_num):
            tik_instance.data_move(data_output[m], data_row1_ub, 0, 1, 1, 0, 0)
            tik_instance.data_move(data_row2_ub, data_input[m], 0, 1, 1, 0, 0)
            tik_instance.vadd(8, data_row1_ub, data_row1_ub, data_row2_ub,
                              1, 1, 1, 1, 0, 0, 0)
            tik_instance.vector_dup([0, cdim_mask], data_row1_ub,
                                    0, 1, 0, 0)

        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=data_input,
                              outputs=data_output)
        return tik_instance
