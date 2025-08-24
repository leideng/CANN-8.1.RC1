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
error manager conv2d.
"""
from te.utils.error_manager.error_manager_util import get_error_message
from te.utils.error_manager.error_manager_util import raise_runtime_error_cube


def raise_err_one_para(err_code, op_name, para1):
    """
    E62301: "In op[%s], the second value of BL1_shape should be a factor of
    the block num, actual input is [%s]."
    E62302: "In op[%s], the second value of BL1_shape should be even number,
    but actually it is [%s]."
    E62501: "In op[%s], [%s] should be 6d list"
    E62509: "In op[%s], the type of axis of [%s] must be positive int."
    E62511: "In op[%s], C0 must be 16,  but actually it is [%s]."
    """
    args_dict = {
        "errCode": err_code,
        "op_name": op_name,
        "para1": para1
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_two_paras(err_code, op_name, para1, para2):
    """
    E62004: "In op[%s],filter D must > Pad D, actual are [%s] and [%s]"
    E62005: "In op[%s], x D(after pad) must >= stride D, actual are [%s] and [%s]"
    E62303: "In op[%s], the value of AL1_shape & BL1_shape are  not reasonable,
    which are [%s] and [%s]."
    E62502: "In op[%s], there is a division by zero operation during calculating,
    the oprands are [%s] and [%s]."
    E62503: "In op[%s], the  Fmap's batch must be equal to the batch in backpropagation
    it should be [%s], but actually it is [%s]."
    E62504: "In op[%s], the  Dedy's channel must be equal to the Filter's batch in
    backpropagation it should be [%s], but actually it is [%s]."
    E62505: "In op[%s], the  input's channel must be equal to the Filter's batch in
    backpropagation it should be [%s], but actually it is [%s]."
    E62506: "In op[%s], the  Dedy's channel must be equal to the Filter's channel in
    backpropagation it should be [%s], but actually it is [%s]."
    """
    args_dict = {
        "errCode": err_code,
        "op_name": op_name,
        "para1": para1,
        "para2": para2
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_three_paras(err_code, op_name, para1, para2, para3):
    """
    E62001: "In op[%s],dilation_h, dilation_w and dilation_d must be 1,
    actual are [%s], [%s] and  [%s]"
    E62304: "In op[%s], the dim of [%s] should be [%s], but it is  [%s]."
    E62305: "In op[%s], the value of [%s] should be [%s], but it is  [%s]."
    E62507: "In op[%s], the [%s] dim of Filter(after dilation) must be less than
    the corresponding dim of input(after padding), they are [%s] and [%s]."
    """
    args_dict = {
        "errCode": err_code,
        "op_name": op_name,
        "para1": para1,
        "para2": para2,
        "para3": para3
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_four_paras(err_code, op_name, para1, para2, para3, para4):
    """
    E62003: "In op[%s], the size of [%s] on [%s] dimension should be in range [%s],
    but it is [%s]."
    """
    args_dict = {
        "errCode": err_code,
        "op_name": op_name,
        "para1": para1,
        "para2": para2,
        "para3": para3,
        "para4": para4
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_input_params_not_expected(op_name, param_name,
                                        expected_value, input_value):
    """
    The op[%s] input parameter[%s] should be [%s], actual the input is [%s] %
    (op_name,param_name,excepted_value,input_value)
    :param op_name
    :param param_name
    :param expected_value
    :param input_value
    :return
    """
    args_dict = {
        "errCode": "E60000",
        "op_name": op_name,
        "param_name": param_name,
        "expected_value": expected_value,
        "input_value": input_value
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_input_params_not_supported(op_name, scale_value, sqrt_mode):
    """
    In op[%s], quant model only surport scale == 1 and sqrt == 0,
    but scale is [%s], sqrt is [%s] %
    (op_name,scale_value,sqrt_mode)
    :param op_name
    :param scale_value
    :param sqrt_mode
    :return
    """
    args_dict = {
        "errCode": "E60036",
        "op_name": op_name,
        "expected_value": scale_value,
        "input_value": sqrt_mode
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_input_format_invalid(op_name, param_name,
                                   expected_format_list, input_format):
    """
    The format of [%s] of op[%s] must in [%s], actual format is [%s] %
    (param_name, op_name, excepted_format_list, format)
    :param op_name
    :param param_name
    :param expected_format_list
    :param input_format
    :return
    """
    args_dict = {
        "errCode": "E60004",
        "op_name": op_name,
        "param_name": param_name,
        "expected_format_list": expected_format_list,
        "format": input_format
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_attr_range_invalid(op_name, attr_range, attr_name, value):
    """
    In op[%s], the [%s] must in range [%s], actual is [%s] %
    (op_name,range,attr_name,value)
    :param op_name
    :param attr_range
    :param attr_name
    :param value
    :return
    """
    args_dict = {
        "errCode": "E60011",
        "op_name": op_name,
        "range": attr_range,
        "attr_name": attr_name,
        "value": value
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_should_4d(op_name, param_name):
    """
    In op[%s], [%s] should be 4d list % (op_name, param_name)
    :param op_name
    :param param_name
    :return
    """
    args_dict = {
        "errCode": "E60107",
        "op_name": op_name,
        "param_name": param_name
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_specific(op_name, reason):
    """
    In op[%s], [%s] % (op_name,reason)
    :param op_name
    :param reason
    :return
    """
    args_dict = {
        "errCode": "E60108",
        "op_name": op_name,
        "reason": reason
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_common(op_name, reason, value):
    """
    In op[%s], [%s],actual is [%s] % (op_name,reason,value)
    :param op_name
    :param reason
    :param value
    :return
    """
    args_dict = {
        "errCode": "E60114",
        "op_name": op_name,
        "reason": reason,
        "value": value
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_should_be_4d(op_name, param_name):
    """
    In op[%s], [%s] should be 4d list % (op_name, param_name)
    :param op_name
    :param param_name
    :return
    """
    args_dict = {
        "errCode": "E61000",
        "op_name": op_name,
        "param_name": param_name
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_specific_user(op_name, reason):
    """
    In op[%s], [%s] % (op_name,reason)
    :param op_name
    :param reason
    :return
    """
    args_dict = {
        "errCode": "E61001",
        "op_name": op_name,
        "reason": reason
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_input_mem_type(op_name, input_memory_type):
    """
    In op[%s], [%s] % (op_name,reason)
    :param op_name
    :param input_memory_type
    :return
    """
    args_dict = {
        "errCode": "E61500",
        "op_name": op_name,
        "input_memory_type": input_memory_type
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_output_mem_type(op_name, output_memory_type):
    """
    In op[%s], [%s] % (op_name,reason)
    :param op_name
    :param output_memory_type
    :return
    """
    args_dict = {
        "errCode": "E61501",
        "op_name": op_name,
        "output_memory_type": output_memory_type
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_check_the_validity_of_variable(op_name, sence_params,
                                             param_1, param_2):
    """
    The op [%s]. [%s] , the value is [%s] and [%s] %
    (op_name, sence_params, param_1, param_2)
    :param op_name
    :param sence_params
    :param param_1
    :param param_2
    :return
    """
    args_dict = {
        "errCode": "E61203",
        "op_name": op_name,
        "sence_params": sence_params,
        "param_1": param_1,
        "param_2": param_2
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_check_the_validity_of_one_variable(op_name,
                                                 sence_params, param_1):
    """
    The op [%s]. [%s] , the value is [%s] %
    (op_name, sence_params, param_1)
    :param op_name
    :param sence_params
    :param param_1
    :param param_2
    :return
    """
    args_dict = {
        "errCode": "E61204",
        "op_name": op_name,
        "sence_params": sence_params,
        "param_1": param_1
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_specific_input_shape(op_name, reason):
    """
    In op[%s], [%s] % (op_name,reason)
    :param op_name
    :param reason
    :return
    """
    args_dict = {
        "errCode": "E61205",
        "op_name": op_name,
        "reason": reason
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_value_or_format_invalid(op_name, param_name,
                                      expect_value, condition):
    """
    wrong tiling in op[%s]: [%s] must be equal to [%s] when [%s] %
    (op_name, param_name, expect_value, condition)
    :param op_name
    :param param_name
    :param expect_value
    :param condition
    :return
    """
    args_dict = {
        "errCode": "E61300",
        "op_name": op_name,
        "param_name": param_name,
        "expect_value": expect_value,
        "condition": condition
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_equal_invalid(op_name, param_name_1, param_name_2):
    """
    wrong tiling in op[%s]: [%s] must be equal to [%s] %
    (op_name, param_name_1, param_name_2)
    :param op_name
    :param param_name_1
    :param param_name_2
    :return
    """
    args_dict = {
        "errCode": "E61301",
        "op_name": op_name,
        "param_name_1": param_name_1,
        "param_name_2": param_name_2
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_scene_limitation(op_name, scene, param_name, claim):
    """
    The op [%s], if it is the [%s] cut shape, the [%s] must be [%s] %
    (op_name, scene, param_name, claim)
    :param op_name
    :param scene
    :param param_name
    :return
    """
    args_dict = {
        "errCode": "E61601",
        "op_name": op_name,
        "scene": scene,
        "param_name": param_name,
        "claim": claim
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_check_type(op_name, param_name, optype_1, optype_2):
    """
    The op [%s] input parameter [%s] should be [%s] type,
    but the type you enter is [%s] %
    (op_name, param_name, optype_1, optype_2)
    :param op_name
    :param param_name
    :param optype_1
    :param optype_2
    :return
    """
    args_dict = {
        "errCode": "E61602",
        "op_name": op_name,
        "param_name": param_name,
        "optype_1": optype_1,
        "optype_2": optype_2
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_scene_equal_limitation(op_name, param_1, param_2):
    """
    The op [%s] [%s] must equal to [%s] % (op_name, param_1, param_2)
    :param op_name
    :param param_1
    :param param_2
    :return
    """
    args_dict = {
        "errCode": "E61603",
        "op_name": op_name,
        "optype_1": param_1,
        "optype_2": param_2
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_err_contain_key_invalid(op_name, param_name, key):
    """
    The op [%s] [%s] input does not contain the [%s] key %
    (op_name,param_name,key)
    :param op_name
    :param param_name
    :param key
    :return
    """
    args_dict = {
        "errCode": "E60029",
        "op_name": op_name,
        "param_name": param_name,
        "key": key
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)


def raise_invalid_range(op_name, attr_name, attr_range, name):
    """
    In op[%s], the lower range of [%s]'s [%s] must greater than 0, the upper range must
    greater or equal to lower range, actual is [%s] (op_name, attr_name, attr_range)
    :param op_name
    :param attr_name
    :param attr_range
    :param name
    :return
    """
    args_dict = {
        "errCode": "E67016",
        "op_name": op_name,
        "attr_name": attr_name,
        "range": attr_range,
        "name": name,
    }
    msg = get_error_message(args_dict)
    raise_runtime_error_cube(args_dict, msg)
