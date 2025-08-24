# -*- coding: UTF-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
_check.py
"""
import inspect


def _check_condition(condition, error_msg):
    if not condition:
        raise RuntimeError(error_msg)


def check_list_tuple(datas, elm_type, error_str):
    """
    check list
    """
    _check_condition((isinstance(error_str, str) and len(error_str) > 0), "invalid error str:{}".format(error_str))
    _check_condition((isinstance(datas, (list, tuple)) and len(datas) > 0), "{}:{}".format(error_str, type(datas)))

    if elm_type is not None:
        _check_condition((all(isinstance(d, elm_type) for d in datas)),
                         "{} elm not all {}:{}".format(error_str, elm_type, datas))


def check_dict(data, key_types, value_types, error_str):
    """
    check dict
    """
    _check_condition((isinstance(error_str, str) and len(error_str) > 0), "{}:{}".format(error_str, type(data)))
    _check_condition((isinstance(data, dict)), "{}:data type {}".format(error_str, type(data)))

    for key, value in data.items():
        if key_types is not None:
            _check_condition((isinstance(key, key_types)), "{}:key type {} not in {}".format(error_str,
                                                                                        type(key), key_types))
        if value_types is not None:
            _check_condition((isinstance(value, value_types)), "{}:v type {} not in {}".format(error_str,
                                                                                          type(value), value_types))


def check_param_type(param, type_list, error_str):
    """
    check param
    """
    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    if not isinstance(param, type_list):
        raise RuntimeError(error_str)


def check_param_range(param, low, high, error_str):
    """
    check range
    """
    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    _check_condition((high > low), "high {} should > low {}".format(high, low))
    if param < low or param > high:
        raise RuntimeError(error_str)


def check_param_low(param, low, error_str):
    """
    check param
    """
    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    if param < low:
        raise RuntimeError(error_str)


def check_param_high(param, high, error_str):
    """
    check param
    """
    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    if param > high:
        raise RuntimeError(error_str)


def check_param_not_equal(param, num, error_str):
    """
    check param
    """
    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    if param == num:
        raise RuntimeError(error_str)


def check_param_equal(param, num, error_str):
    """
    check param
    """
    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    if param != num:
        raise RuntimeError(error_str)


def check_param_mod(param, base_num, error_str):
    """
    check param
    """
    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    _check_condition((isinstance(param, int) and isinstance(base_num, int)),
                     "err type:{}, {}".format(type(param), type(base_num)))
    if param % base_num != 0:
        raise RuntimeError(error_str)


def check_func(func, error_str):
    """
    check function
    """
    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    if not inspect.isfunction(func):
        raise RuntimeError(error_str)


# --------------------------- tik related -------------------------------
# 'pylint: disable=too-few-public-methods
class TikDebug:
    """
    flag: if debug
    """
    flag = False

    def __init__(self):
        pass


def set_tik_debug_flag(b_flag):
    """
    set debug flag
    :param b_flag:
    :return:
    """
    TikDebug.flag = b_flag


def check_tik_tensor(tensor, scope_list, dtype_list, tik):
    """
    check tik tensor
    """
    if hasattr(tik, "ir_builder_lib"):
        tensor_type = tik.ir_builder_lib.ib_Tensor
    elif hasattr(tik, "api"):
        tensor_type = tik.api.tik_Tensor
    else:
        raise RuntimeError("not supported tik version")
    _check_condition((isinstance(tensor, tensor_type)), "invalid tensor:{}".format(type(tensor)))

    if scope_list is not None:
        _check_condition((isinstance(scope_list, (list, tuple)) and len(scope_list) > 0),
                         "scope_list should be list or tuple")
        _check_condition((tensor.scope in scope_list), "{} not in {}".format(tensor.scope, scope_list))
    if dtype_list is not None:
        _check_condition((isinstance(dtype_list, (list, tuple)) and len(dtype_list) > 0),
                         "dtype_list should be list or tuple")
        _check_condition((tensor.dtype in dtype_list), "{} not in {}".format(tensor.dtype, dtype_list))


def is_tik_dynamic(param, tik):
    """
    check dynamic
    """
    if hasattr(tik, "ir_builder_lib"):
        # C3X
        return (isinstance(param, (tik.ir_builder_lib.ib_tensor.Expr,
                                   tik.ir_builder_lib.ib_tensor.Scalar,
                                   tik.ir_builder_lib.ib_Tensor)))
    if hasattr(tik, "tik_lib"):
        # C7X
        return isinstance(param, (tik.tik_lib.tik_expr.Expr,
                                  tik.api.tik_scalar.Scalar,
                                  tik.api.tik_scalar.InputScalar,
                                  tik.api.tik_tensor.Tensor))

    raise RuntimeError("not supported tik version")


def check_tik_param_low(param, tik, tinst, low, error_str):
    """
    check param
    """
    if not TikDebug.flag:
        return
    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    if is_tik_dynamic(param, tik):
        with tinst.if_scope(param < low):
            tinst.tik_return()
        with tinst.else_scope():
            pass
    else:
        _check_condition((not param < low), error_str)


def check_tik_param_high(param, tik, tinst, high, error_str):
    """
    check param
    """
    if not TikDebug.flag:
        return

    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    if is_tik_dynamic(param, tik):
        with tinst.if_scope(param > high):
            tinst.tik_return()
        with tinst.else_scope():
            pass
    else:
        _check_condition((not param > high), error_str)


def check_tik_param_not_equal(param, tik, tinst, num, error_str):
    """
    check param
    """
    if not TikDebug.flag:
        return

    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    if is_tik_dynamic(param, tik):
        with tinst.if_scope(param == num):
            tinst.tik_return()
        with tinst.else_scope():
            pass
    else:
        _check_condition((param != num), error_str)


def check_tik_param_mod(param, tik, tinst, num, error_str):
    """
    check tik param
    """
    if not TikDebug.flag:
        return

    _check_condition((isinstance(error_str, str)), "err type:{}".format(type(error_str)))
    if is_tik_dynamic(param, tik):
        with tinst.if_scope(param % num != 0):
            tinst.tik_return()
        with tinst.else_scope():
            pass
    else:
        _check_condition((isinstance(param, int) and isinstance(num, int)), "param and num should be int")
        _check_condition((param % num == 0), error_str)


def check_tik_param_dtype(param, dtype_list, tik):
    """
    check dtype
    :param param:
    :param dtype_list:
    :param tik:
    :return:
    """
    if not TikDebug.flag:
        return

    _check_condition((isinstance(dtype_list, (list, tuple)) and len(dtype_list) > 0),
                     "dtype_list should be list or tuple")
    if not is_tik_dynamic(param, tik):
        raise ValueError("param:{} is not tik related".format(param))
    if param.dtype not in dtype_list:
        raise RuntimeError("invalid param dtype:{}".format(param.dtype))
