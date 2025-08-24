#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
error manager vector.
"""
from .error_manager_util import get_error_message


def raise_err_input_value_invalid(op_name, param_name, excepted_value, real_value):
    """
    "In op[%s], the parameter[%s] should be [%s], but actually is [%s]." %
    (op_name,param_name,excepted_value, real_value)
    :param op_name
    :param param_name
    :param excepted_value
    :param real_value
    :return
    """
    args_dict = {
        "errCode": "E80000",
        "op_name": op_name,
        "param_name": param_name,
        "excepted_value": excepted_value,
        "real_value": real_value
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_miss_mandatory_parameter(op_name, param_name):
    """
    "In op[%s], the mandatory parameter[%s] is missed." %
    (op_name,param_name)
    :param op_name
    :param param_name
    :return
    """
    args_dict = {
        "errCode": "E80001",
        "op_name": op_name,
        "param_name": param_name,
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_input_param_not_in_range(op_name, param_name, min_value, max_value, value):
    """
    "In op[%s], the parameter[%s] should be in the range of [%s, %s], but actually is [%s]." %
    (op_name,param_name,min_value,max_value,value)
    :param op_name
    :param param_name
    :param min_value
    :param max_value
    :param value
    :return
    """
    args_dict = {
        "errCode": "E80001",
        "op_name": op_name,
        "param_name": param_name,
        "min_value": min_value,
        "max_value": max_value,
        "value": value,
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_input_dtype_not_supported(op_name, param_name, excepted_dtype_list, dtype):
    """
    "In op[%s], the parameter[%s]'s dtype should be one of [%s], but actually is [%s]." %
    (param,op_name,expected_data_type_list,data_type)
    :param op_name
    :param param_name
    :param excepted_dtype_list
    :param data_type
    :return
    """
    args_dict = {
        "errCode": "E80008",
        "op_name": op_name,
        "param_name": param_name,
        "expected_data_type_list": excepted_dtype_list,
        "dtype": dtype
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_check_params_rules(op_name, rule_desc, param_name, param_value):
    """
    "Op[{op_name}] has rule: {rule_desc}, but [{param_name}] is [{param_value}]."
    :param op_name
    :param rule_desc
    :param param_name
    :param param1_dtype
    :param param_value
    :return
    """
    args_dict = {
        "errCode": "E80009",
        "op_name": op_name,
        "rule_desc": rule_desc,
        "param_name": param_name,
        "param_value": param_value
    }

    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_input_format_invalid(op_name, param_name, excepted_format_list, actual_format):
    """
    In op[{op_name}], the format of input[{param_name}] should be
    one of [{excepted_format_list}], but actually is [{format}]
    :param op_name:
    :param param_name:
    :param excepted_format_list:
    :param actual_format:
    :return:
    """
    args_dict = {
        'errCode': "E80015",
        'op_name': op_name,
        'param_name': param_name,
        'excepted_format_list': excepted_format_list,
        'format': actual_format
    }

    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_inputs_shape_not_equal(op_name, param_name1, param_name2, param1_shape, param2_shape, expect_shape):
    """
    "In op[%s], the parameter[%s][%s] is not match with the parameter[%s][%s],it should be [%s]." %
    (op_name,param_name1,param_name2,param1_shape,param2_shape,expect_shape)
    :param op_name:
    :param param_name1:
    :param param_name2:
    :param param1_shape:
    :param param2_shape:
    :param expect_shape:
    :return:
    """
    args_dict = {
        'errCode': "E80017",
        'op_name': op_name,
        'param_name1': param_name1,
        'param_name2': param_name2,
        'param1_shape': param1_shape,
        'param2_shape': param2_shape,
        'expect_shape': expect_shape
    }

    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_inputs_dtype_not_equal(op_name, param_name1, param_name2, param1_dtype, param2_dtype):
    """
    "In op[%s], the parameter[%s][%s] are not equal in dtype with dtype[%s][%s]." %
    (op_name,param_name1,param_name2,param1_dtype, param2_dtype)
    :param op_name
    :param param_name1
    :param param_name2
    :param param1_dtype
    :param param2_dtype
    :return
    """
    args_dict = {
        "errCode": "E80018",
        "op_name": op_name,
        "param_name1": param_name1,
        "param_name2": param_name2,
        "param1_dtype": param1_dtype,
        "param2_dtype": param2_dtype
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_input_shape_invalid(op_name, param_name, error_detail):
    """
    "In op[%s], the shape of input[%s] is invalid, [%s]." %
    (op_name,param_name,error_detail)
    :param op_name
    :param param_name
    :param error_detail
    :return
    """
    args_dict = {"errCode": "E80028", "op_name": op_name, "param_name": param_name, "error_detail": error_detail}
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_two_input_shape_invalid(op_name, param_name1, param_name2, error_detail):
    """
    "In op[%s], the shape of inputs[%s][%s] are invalid, [%s]." %
    (op_name,param_name1,param_name2,error_detail)
    :param op_name
    :param param_name1
    :param param_name2
    :param error_detail
    :return
    """
    args_dict = {
        "errCode": "E80029",
        "op_name": op_name,
        "param_name1": param_name1,
        "param_name2": param_name2,
        "error_detail": error_detail
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_two_input_dtype_invalid(op_name, param_name1, param_name2, error_detail):
    """
    "In op[%s], the dtype of inputs[%s][%s] are invalid, [%s]." %
    (op_name,param_name1,param_name2,error_detail)
    :param op_name
    :param param_name1
    :param param_name2
    :param error_detail
    :return
    """
    args_dict = {
        "errCode": "E80030",
        "op_name": op_name,
        "param_name1": param_name1,
        "param_name2": param_name2,
        "error_detail": error_detail
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_two_input_format_invalid(op_name, param_name1, param_name2, error_detail):
    """
    "In op[%s], the format of inputs[%s][%s] are invalid, [%s]." %
    (op_name,param_name1,param_name2,error_detail)
    :param op_name
    :param param_name1
    :param param_name2
    :param error_detail
    :return
    """
    args_dict = {
        "errCode": "E80031",
        "op_name": op_name,
        "param_name1": param_name1,
        "param_name2": param_name2,
        "error_detail": error_detail
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_specific_reson(op_name, reason):
    """
    In op[%s], [%s] % (op_name,reason)
    :param op_name
    :param reason
    :return
    """
    args_dict = {"errCode": "E61001", "op_name": op_name, "reason": reason}
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_pad_mode_invalid(op_name, expected_pad_mode, actual_pad_mode):
    """
    "In op [%s], only support pads model [%s], actual is [%s]." %
    (op_name, expected_pad_mode, actual_pad_mode)
    :param op_name
    :param expected_pad_mode
    :param actual_pad_mode
    :return
    """
    args_dict = {
        "errCode": "E60021",
        "op_name": op_name,
        "expected_pad_mode": expected_pad_mode,
        "actual_pad_mode": actual_pad_mode
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_input_param_range_invalid(op_name, param_name, min_value, max_value, real_value):
    """
    "In op[%s], the num of dimensions of input[%s] should be in the range of [%s, %s], but actually is [%s]." %
    (op_name,param_name,max_value,min_value,real_value)
    :param op_name
    :param param_name
    :param min_value
    :param max_value
    :param real_value
    :return
    """
    args_dict = {
        "errCode": "E80012",
        "op_name": op_name,
        "param_name": param_name,
        "min_value": min_value,
        "max_value": max_value,
        "real_value": real_value
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_dtype_invalid(op_name, param_name, expected_list, dtype):
    """
    "The dtype of [%s] of op[%s] must in [%s], actual dtype is [%s]"
    :param op_name
    :param param_name
    :param expected_list
    :param dtype
    :return
    """
    args_dict = {
        "errCode": "E60005",
        "op_name": op_name,
        "param_name": param_name,
        "expected_dtype_list": expected_list,
        "dtype": dtype,
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)
