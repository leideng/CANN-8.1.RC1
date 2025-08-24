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
common function
"""
import re
import math
from functools import reduce as functools_reduce
from functools import wraps
import warnings

from te.utils import shape_util

SHAPE_SIZE_LIMIT = 2 ** 31 - 1
SHAPE_SIZE_ZERO = 0
RANK_ZERO = 0
RANK_LIMIT = 8
DIM_LIMIT = 2 ** 31 - 1
ZERO_DIM = 0
# the max len of kernel_name
MAX_KERNEL_NAEM_LEN = 200
MIN_UNKOWN_SHAPE_RANK = 0
MAX_UNKOWN_SHAPE_NUM = 2 ** 31 - 1

CONST = "const"
SPECIAL = "special"
ORIGINAL = "original"
SPECIAL_SCALAR = "special_scalar"
COMMON = "common"
BROADCAST = "broadcast"

REQUIRED_INPUT = "required_input"
OPTION_INPUT = "option_input"
DYNAMIC_INPUT = "dynamic_input"

REQUIRED_OUTPUT = "required_output"
OPTION_OUTPUT = "option_output"
DYNAMIC_OUTPUT = "dynamic_output"

# in proto attr can be a Tensor/BYTES/LIST_TYPE Type, but not te fusion don't support this type
REQUIRED_ATTR_INT = "REQUIRED_ATTR_INT"
REQUIRED_ATTR_FLOAT = "REQUIRED_ATTR_FLOAT"
REQUIRED_ATTR_STR = "REQUIRED_ATTR_STR"
REQUIRED_ATTR_BOOL = "REQUIRED_ATTR_BOOL"
REQUIRED_ATTR_TYPE = "REQUIRED_ATTR_TYPE"
REQUIRED_ATTR_LIST_INT = "REQUIRED_ATTR_LIST_INT"
REQUIRED_ATTR_LIST_FLOAT = "REQUIRED_ATTR_LIST_FLOAT"
REQUIRED_ATTR_LIST_BOOL = "REQUIRED_ATTR_LIST_BOOL"
REQUIRED_ATTR_LIST_LIST_INT = "REQUIRED_ATTR_LIST_LIST_INT"

OPTION_ATTR_INT = "OPTION_ATTR_INT"
OPTION_ATTR_FLOAT = "OPTION_ATTR_FLOAT"
OPTION_ATTR_STR = "OPTION_ATTR_STR"
OPTION_ATTR_BOOL = "OPTION_ATTR_BOOL"
OPTION_ATTR_TYPE = "OPTION_ATTR_TYPE"
OPTION_ATTR_LIST_INT = "OPTION_ATTR_LIST_INT"
OPTION_ATTR_LIST_FLOAT = "OPTION_ATTR_LIST_FLOAT"
OPTION_ATTR_LIST_BOOL = "OPTION_ATTR_LIST_BOOL"
OPTION_ATTR_LIST_LIST_INT = "OPTION_ATTR_LIST_LIST_INT"

KERNEL_NAME = "kernel_name"

OP_ERROR_CODE_000 = 'E80000'
OP_ERROR_CODE_001 = 'E80001'
OP_ERROR_CODE_002 = 'E80002'
OP_ERROR_CODE_003 = 'E80003'
OP_ERROR_CODE_004 = 'E80004'
OP_ERROR_CODE_005 = 'E80005'
OP_ERROR_CODE_006 = 'E80006'
OP_ERROR_CODE_007 = 'E80007'
OP_ERROR_CODE_008 = 'E80008'
OP_ERROR_CODE_009 = 'E80009'
OP_ERROR_CODE_010 = 'E80010'
OP_ERROR_CODE_011 = 'E80011'
OP_ERROR_CODE_012 = 'E80012'
OP_ERROR_CODE_013 = 'E80013'
OP_ERROR_CODE_014 = 'E80014'
OP_ERROR_CODE_015 = 'E80015'
OP_ERROR_CODE_016 = 'E80016'
OP_ERROR_CODE_017 = 'E80017'
OP_ERROR_CODE_018 = 'E80018'
OP_ERROR_CODE_019 = 'E80019'
OP_ERROR_CODE_020 = 'E80020'
OP_ERROR_CODE_021 = 'E80021'
OP_ERROR_CODE_022 = 'E80022'
OP_ERROR_CODE_023 = 'E80023'
OP_ERROR_CODE_024 = 'E80024'
OP_ERROR_CODE_025 = 'E80025'
OP_ERROR_CODE_026 = 'E80026'
OP_ERROR_CODE_027 = 'E80027'


class OpParamInfoKey:  # 'pylint: disable=too-few-public-methods
    """
    Define op params
    """

    def __init__(self):
        pass

    SHAPE = "shape"
    FORMAT = "format"
    ORI_SHAPE = "ori_shape"
    ORI_FORMAT = "ori_format"
    D_TYPE = "dtype"
    RANGE = "range"


class TensorFormat:  # 'pylint: disable=too-few-public-methods
    """
    Define op params
    """

    def __init__(self):
        pass

    ND = "ND"
    NCHW = "NCHW"
    NHWC = "NHWC"
    NDHWC = "NDHWC"
    NCDHW = "NCDHW"
    CHWN = "CHWN"
    NC1HWC0 = "NC1HWC0"
    NC1HWC0_C04 = "NC1HWC0_C04"
    NDC1HWC0 = "NDC1HWC0"
    FRACTAL_NZ = "FRACTAL_NZ"

    HWCN = "HWCN"
    DHWCN = "DHWCN"
    FRACTAL_Z = "FRACTAL_Z"
    FRACTAL_Z_C04 = "FRACTAL_Z_C04"
    C1HWNCoC0 = "C1HWNCoC0"
    FRACTAL_Z_3D = "FRACTAL_Z_3D"
    FRACTAL_ZN_LSTM = "FRACTAL_ZN_LSTM"


ALL_FORMAT_LIST = [TensorFormat.__dict__[d_key] for d_key in TensorFormat.__dict__ if "__" not in d_key]
ALL_DTYPE_LIST = ("int8", "uint8", "int16", "uint16", "int32", "uint32",
                  "int64", "uint64", "float16", "float32", "float64", "bool", "uint1")
OP_NAME = ""
PARAM_NAME = ""


def check_op_params(*type_args,  # 'pylint: disable=too-many-locals,too-many-statements
                    **type_kwargs):  # 'pylint: disable=unused-argument,
    """
    check op params
    """
    input_params = [REQUIRED_INPUT, OPTION_INPUT, DYNAMIC_INPUT]
    output_params = [REQUIRED_OUTPUT, OPTION_OUTPUT, DYNAMIC_OUTPUT]
    required_attr_params = [REQUIRED_ATTR_STR, REQUIRED_ATTR_FLOAT,
                            REQUIRED_ATTR_INT, REQUIRED_ATTR_BOOL,
                            REQUIRED_ATTR_TYPE, REQUIRED_ATTR_LIST_INT,
                            REQUIRED_ATTR_LIST_BOOL, REQUIRED_ATTR_LIST_FLOAT,
                            REQUIRED_ATTR_LIST_LIST_INT]
    list_type_attr = [REQUIRED_ATTR_LIST_BOOL, REQUIRED_ATTR_LIST_INT,
                      REQUIRED_ATTR_LIST_FLOAT, REQUIRED_ATTR_LIST_LIST_INT,
                      OPTION_ATTR_LIST_BOOL, OPTION_ATTR_LIST_INT,
                      OPTION_ATTR_LIST_FLOAT, OPTION_ATTR_LIST_LIST_INT]

    def _check_input_output_key(op_param, param_name, op_name=OP_NAME):
        # check all necessary information(shape, format, ori_shape, ori_format, dtype)
        from tbe.dsl.base import operation
        if not isinstance(op_param, dict):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = 'dict'
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s type should be [%s],  "
                                           "but actually is [%s]." % (error_info['op_name'],
                                                                      error_info['param_name'],
                                                                      error_info['param_type'],
                                                                      error_info['actual_type']))
        if OpParamInfoKey.SHAPE not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.SHAPE
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name'], error_info['key']))
        if OpParamInfoKey.FORMAT not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.FORMAT
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name'], error_info['key']))
        if OpParamInfoKey.ORI_SHAPE not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.ORI_SHAPE
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name'], error_info['key']))
        if OpParamInfoKey.ORI_FORMAT not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.ORI_FORMAT
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name'], error_info['key']))
        if OpParamInfoKey.D_TYPE not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.D_TYPE
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name'], error_info['key']))

        if operation.in_dynamic():
            if OpParamInfoKey.RANGE not in op_param.keys():
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_004
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['key'] = OpParamInfoKey.RANGE
                raise RuntimeError(error_info,
                                   "In op[%s], the input[%s] does not contain the item[%s]."
                                   % (error_info['op_name'], error_info['param_name'], error_info['key']))

    def _check_input_output_dict(op_param, param_name, op_name=OP_NAME):
        from tbe.dsl.base import operation
        _check_input_output_key(op_param, param_name, op_name)
        if operation.in_dynamic():
            check_range(op_param[OpParamInfoKey.SHAPE], op_param[OpParamInfoKey.RANGE], param_name=param_name)
        check_shape(op_param[OpParamInfoKey.SHAPE], param_name=param_name)
        check_shape(op_param[OpParamInfoKey.ORI_SHAPE], param_name=param_name)

        if op_param[OpParamInfoKey.FORMAT] not in TensorFormat.__dict__.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_015
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['excepted_format_list'] = ",".join(ALL_FORMAT_LIST)
            error_info['format'] = op_param[OpParamInfoKey.FORMAT]

            raise RuntimeError(error_info, "In op[%s], the format of input[%s] "
                                           "should be one of [%s], but actually is [%s]."
                               % (error_info['op_name'],
                                  error_info['param_name'],
                                  error_info['excepted_format_list'],
                                  error_info['format']))

        if op_param[OpParamInfoKey.ORI_FORMAT] not in TensorFormat.__dict__.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_014
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['excepted_format_list'] = ",".join(ALL_FORMAT_LIST)
            error_info['format'] = op_param[OpParamInfoKey.ORI_FORMAT]
            raise RuntimeError(error_info,
                               "In op[%s], the ori format of input[%s] should be one of [%s]"
                               ", but actually is [%s]."
                               % (error_info['op_name'],
                                  error_info['param_name'],
                                  ",".join(ALL_FORMAT_LIST),
                                  error_info['format']))

        if not isinstance(op_param[OpParamInfoKey.D_TYPE], str):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = 'str'
            error_info['actual_type'] = op_param[OpParamInfoKey.D_TYPE].__class__.__name__
            raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s type should be [%s],  "
                                           "but actually is [%s]." % (error_info['op_name'],
                                                                      error_info['param_name'],
                                                                      error_info['param_type'],
                                                                      error_info['actual_type']))

        if op_param[OpParamInfoKey.D_TYPE] is None or op_param[OpParamInfoKey.D_TYPE].lower() not in ALL_DTYPE_LIST:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_008
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['excepted_dtype_list'] = ",".join(ALL_DTYPE_LIST)
            error_info['dtype'] = op_param[OpParamInfoKey.D_TYPE]
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s dtype should be "
                               "one of [%s], but actually is [%s]." %
                               (error_info['op_name'],
                                error_info['param_name'],
                                error_info['excepted_dtype_list'],
                                error_info['dtype']))

    def _check_input(op_param, param_name, param_type, op_name=OP_NAME):
        if param_type == REQUIRED_INPUT:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_001
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            if op_param is None:
                raise RuntimeError(error_info, "In op[%s], the mandatory "
                                               "parameter[%s] is missed."
                                   % (error_info['op_name'], error_info['param_name']))
            _check_input_output_dict(op_param, param_name, op_name)
        elif param_type == OPTION_INPUT:
            if op_param is not None:
                _check_input_output_dict(op_param, param_name, op_name)
        else:
            if not isinstance(op_param, (list, tuple)):
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_003
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['param_type'] = "list truple"
                error_info['actual_type'] = op_param.__class__.__name__
                raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s type should be [%s]"
                                               ",  but actually is [%s]."
                                   % (op_name, param_name, error_info['param_type'], error_info['actual_type']))
            if not op_param:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_001
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                raise RuntimeError(error_info, "In op[%s], the mandatory parameter[%s]"
                                               " is missed." % (op_name, param_name))
            for one_input in op_param:
                _check_input_output_dict(one_input, param_name, op_name)

    def _check_output(op_param, param_name, param_type, op_name=OP_NAME):
        if param_type == REQUIRED_OUTPUT:
            if op_param is None:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_001
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                raise RuntimeError(error_info, "In op[%s], the mandatory parameter[%s]"
                                               " is missed." % (op_name, param_name))

            _check_input_output_dict(op_param, param_name, op_name)
        elif param_type == OPTION_OUTPUT:
            if op_param is not None:
                _check_input_output_dict(op_param, param_name, op_name)
        else:
            if not isinstance(op_param, (list, tuple)):
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_003
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['param_type'] = "list tuple"
                error_info['actual_type'] = op_param.__class__.__name__
                raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s "
                                               "type should be [%s],  but actually is [%s]."
                                   % (op_name, param_name, error_info['param_type'], error_info['actual_type']))
            if not op_param:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_001
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                raise RuntimeError(error_info, "In op[%s], the mandatory"
                                               " parameter[%s] is missed."
                                   % (op_name, param_name))
            for one_input in op_param:
                _check_input_output_dict(one_input, param_name, op_name)

    def _check_attr_type(op_param, param_name, py_type, py_type_name, op_name=OP_NAME):
        if not isinstance(op_param, py_type):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = str(py_type)
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s type should be [%s],"
                               " but actually is [%s]."
                               % (error_info['op_name'], error_info['param_name'],
                                  error_info['param_type'], error_info['actual_type']))
        if py_type_name == "float":
            if math.isinf(op_param) or math.isnan(op_param):
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_000
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['excepted_value'] = "float range data"
                error_info['real_value'] = str(op_param)
                raise RuntimeError(error_info,
                                   "In op[%s], the parameter[%s] should be [%s], but actually is [%s]."
                                   % (error_info['op_name'], error_info['param_name'],
                                      error_info['excepted_value'], error_info['real_value']))

    def _check_list_attr_element(op_param, param_name, py_type, py_type_name, op_name=OP_NAME):
        if not isinstance(op_param, py_type):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = str(py_type)
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s type should be [%s],"
                               " but actually is [%s]."
                               % (error_info['op_name'], error_info['param_name'],
                                  error_info['param_type'], error_info['actual_type']))
        if py_type_name == "float":
            if math.isinf(op_param) or math.isnan(op_param):
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_000
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['excepted_value'] = "float range data"
                error_info['real_value'] = str(op_param)
                raise RuntimeError(error_info,
                                   "In op[%s], the parameter[%s] should be [%s], but actually is [%s]."
                                   % (error_info['op_name'], error_info['param_name'],
                                      error_info['excepted_value'], error_info['real_value']))

    def _check_list_attr(op_param, param_name, param_type, op_name=OP_NAME):
        if not isinstance(op_param, (list, tuple)):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = "list tuple"
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s type should be [%s],"
                               "  but actually is [%s]."
                               % (error_info['op_name'], error_info['param_name'],
                                  error_info['param_type'], error_info['actual_type']))

        if param_type in [REQUIRED_ATTR_LIST_BOOL, OPTION_ATTR_LIST_BOOL]:
            for one_attr in op_param:
                _check_list_attr_element(one_attr, param_name, bool, "bool", op_name)

        if param_type in [REQUIRED_ATTR_LIST_INT, OPTION_ATTR_LIST_INT]:
            for one_attr in op_param:
                _check_list_attr_element(one_attr, param_name, int, "int", op_name)

        if param_type in [REQUIRED_ATTR_LIST_FLOAT, OPTION_ATTR_LIST_FLOAT]:
            for one_attr in op_param:
                _check_list_attr_element(one_attr, param_name, float, "float", op_name)

        if param_type in [REQUIRED_ATTR_LIST_LIST_INT, OPTION_ATTR_LIST_LIST_INT]:
            for one_attr in op_param:
                if not isinstance(one_attr, (list, tuple)):
                    error_info = {}
                    error_info['errCode'] = OP_ERROR_CODE_003
                    error_info['op_name'] = op_name
                    error_info['param_name'] = param_name
                    error_info['param_type'] = "list tuple"
                    error_info['actual_type'] = op_param.__class__.__name__
                    raise RuntimeError(error_info,
                                       "In op[%s], the parameter[%s]'s type should be [%s],"
                                       " but actually is [%s]."
                                       % (error_info['op_name'],
                                          error_info['param_name'],
                                          error_info['param_type'],
                                          error_info['actual_type']))

                for ele in one_attr:
                    _check_list_attr_element(ele, param_name, int, "int", op_name)

    def _check_attr(op_param, param_name, param_type, op_name=OP_NAME):
        if op_param is None and param_type in required_attr_params:

            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_001
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            raise RuntimeError(error_info,
                               "In op[%s], the mandatory parameter[%s] is missed."
                               % (op_name, param_name))
        if not op_param:
            return

        if param_type in [REQUIRED_ATTR_INT, OPTION_ATTR_INT]:
            _check_attr_type(op_param, param_name, int, "int", op_name)

        if param_type in [REQUIRED_ATTR_FLOAT, OPTION_ATTR_FLOAT]:
            _check_attr_type(op_param, param_name, float, "float", op_name)

        if param_type in [REQUIRED_ATTR_STR, OPTION_ATTR_STR]:
            _check_attr_type(op_param, param_name, str, "string", op_name)

        if param_type in [REQUIRED_ATTR_BOOL, OPTION_ATTR_BOOL]:
            _check_attr_type(op_param, param_name, bool, "bool", op_name)

        if param_type in [REQUIRED_ATTR_TYPE, OPTION_ATTR_TYPE]:
            if op_param not in ALL_DTYPE_LIST:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_003
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['param_type'] = " ".join(ALL_DTYPE_LIST)
                error_info['actual_type'] = op_param.__class__.__name__
                raise RuntimeError(error_info,
                                   "In op[%s], the parameter[%s]'s dtype should"
                                   " be one of [%s], but actually is [%s]."
                                   % (error_info['op_name'],
                                      error_info['param_name'],
                                      error_info['param_type'],
                                      error_info['actual_type']))

        if param_type in list_type_attr:
            _check_list_attr(op_param, param_name, param_type, op_name)

    def _check_kernel_name(kernel_name, param_name, op_name):
        """
        check kernel_name
        """
        if not isinstance(kernel_name, str):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = "str"
            error_info['actual_type'] = kernel_name.__class__.__name__
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s type should be [%s], "
                               "but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['param_type'], error_info['actual_type']))

        if len(kernel_name) > MAX_KERNEL_NAEM_LEN:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_002
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['min_value'] = '0'
            error_info['max_value'] = str(MAX_KERNEL_NAEM_LEN)
            error_info['real_value'] = str(len(kernel_name))
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s] should be in the range of [%s, %s],"
                               "but actually is [%s]." % (error_info['op_name'],
                                                          error_info['param_name'],
                                                          error_info['min_value'],
                                                          error_info['max_value'],
                                                          error_info['real_value']))

        pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        if not pattern.match(kernel_name):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_020
            error_info['op_name'] = op_name
            raise RuntimeError(error_info,
                               "In op[%s],kernel_name can only contain letters, numbers and underscores,"
                               " and begin with underscores or letters" % (error_info['op_name']))

    def _check_one_op_param(op_param, param_name, param_type, op_name=OP_NAME):

        if param_type in input_params:
            _check_input(op_param, param_name, param_type, op_name)
        elif param_type in output_params:
            _check_output(op_param, param_name, param_type, op_name)
        elif param_type == KERNEL_NAME:
            if op_param is None:
                return
            _check_kernel_name(op_param, param_name, op_name)
        else:  # else is attr_params:
            _check_attr(op_param, param_name, param_type, op_name)

    def _out_wrapper(func):
        formal_parameter = func.__code__.co_varnames
        formal_parameter_list = list(zip(formal_parameter, type_args))

        @wraps(func)
        def _in_wrapper(*args, **kwargs):
            for i, one_args in enumerate(args):
                op_name = func.__name__
                _check_one_op_param(one_args, formal_parameter_list[i][0],
                                    formal_parameter_list[i][1], op_name)

            for arg_key in kwargs:
                op_name = func.__name__
                for name_type in formal_parameter_list:
                    if arg_key == name_type[0]:
                        _check_one_op_param(kwargs[arg_key], arg_key, name_type[1], op_name)
                        break

            return func(*args, **kwargs)

        return _in_wrapper

    return _out_wrapper


def check_range(shape, shape_range, min_dim=0,  # 'pylint: disable=too-many-arguments
                max_dim=RANK_LIMIT, max_shape_num=MAX_UNKOWN_SHAPE_NUM,  # 'pylint: disable=too-many-arguments
                param_name=PARAM_NAME):  # 'pylint: disable=too-many-arguments
    """
    check rule for tensor shape
    """
    if not isinstance(shape_range, (tuple, list)):
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_003
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['param_type'] = "list tuple"
        error_info['actual_type'] = shape_range.__class__.__name__
        raise RuntimeError(error_info,
                           "In op, the parameter[%s]'s type should be [%s],"
                           "but actually is [%s]." %
                           (error_info['param_name'],
                            error_info['param_type'], error_info['actual_type']))
    if len(shape) != len(shape_range):
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_021
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['shape_len'] = len(shape)
        error_info['range_len'] = len(shape_range)
        raise RuntimeError(error_info,
                           "In op, the length of shape[%s] and"
                           "the length of range[%s] must be the same." %
                           (error_info['shape_len'], error_info['range_len']))

    for range_i in shape_range:
        if len(range_i) == 2 and (range_i[1] is None) \
                and isinstance(range_i[0], int) \
                and 0 <= range_i[0] <= max_shape_num:
            continue
        if not isinstance(range_i[0], int):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['param_type'] = 'int'
            error_info['actual_type'] = range_i[0].__class__.__name__
            raise RuntimeError(error_info,
                               "In op, the parameter[%s]'s type should be [%s],"
                               "but actually is [%s]." %
                               (error_info['param_name'], error_info['param_type'],
                                error_info['actual_type']))
        if not isinstance(range_i[1], int):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['param_type'] = 'int'
            error_info['actual_type'] = range_i[1].__class__.__name__
            raise RuntimeError(error_info,
                               "In op, the parameter[%s]'s type should be [%s],"
                               "but actually is [%s]." %
                               (error_info['param_name'], error_info['param_type'],
                                error_info['actual_type']))
        valid_type = isinstance(range_i[0], int) and \
            isinstance(range_i[1], int)
        if len(range_i) != 2:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_023
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            raise RuntimeError(error_info,
                               "In op[%s],the length of each element"
                               "in the range must be two" %
                               (error_info['op_name']))
        valid_range = len(range_i) == 2 and 0 <= range_i[0] <= range_i[1] <= max_shape_num
        if valid_type and valid_range:
            continue
        else:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_022
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['first_real_value'] = range_i[0]
            error_info['second_real_value'] = range_i[1]
            error_info['min_range_value'] = 0
            error_info['max_range_value'] = max_shape_num
            raise RuntimeError(error_info,
                               "In op, the ndim of first range input[%s] "
                               "is less than that of the second range input[%s], "
                               "and the ndim of range should be in the range of [%s, %s]."
                               % (error_info['first_real_value'],
                                  error_info['second_real_value'],
                                  0,
                                  max_shape_num))


def check_dynamic_shape(shape, max_dim=DIM_LIMIT, max_rank=RANK_LIMIT, param_name=PARAM_NAME):
    """
    check invalid for dynamic shape
    :param shape:
    :param max_dim: default is 2 ** 31 - 1
    :param max_rank: default is 8
    :param param_name: default is empty string
    :return:
    """
    if len(shape) < MIN_UNKOWN_SHAPE_RANK or len(shape) > max_rank:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_012
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['min_value'] = MIN_UNKOWN_SHAPE_RANK
        error_info['max_value'] = max_rank
        error_info['real_value'] = len(shape)
        raise RuntimeError(error_info,
                           "In op, the num of dimensions of input[%s] should be in"
                           "the range of [%s, %s], but actually is [%s]."
                           % (error_info['param_name'], MIN_UNKOWN_SHAPE_RANK, max_rank, len(shape)))
    for _, dim in enumerate(shape):
        valid_dim = -2 <= dim <= max_dim
        if not valid_dim:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_002
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['min_value'] = "-2"
            error_info['max_value'] = max_dim
            error_info['real_value'] = dim
            raise RuntimeError(error_info,
                               "In op, the parameter[%s] should be in the range of [%s, %s],"
                               "but actually is [%s]."
                               % (error_info['param_name'], -2, max_dim, dim))


def check_shape(shape, min_dim=0, max_dim=DIM_LIMIT,  # 'pylint: disable=too-many-arguments
                min_rank=0, max_rank=RANK_LIMIT,  # 'pylint: disable=too-many-arguments
                min_size=0, max_size=SHAPE_SIZE_LIMIT, param_name=PARAM_NAME):  # 'pylint: disable=too-many-arguments
    """
    check shape size
    """
    from tbe.dsl.base import operation
    if not isinstance(shape, (tuple, list)):

        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_003
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['param_type'] = "list tuple"
        error_info['actual_type'] = shape.__class__.__name__
        raise RuntimeError(error_info,
                           "In op, the parameter[%s]'s type should be [%s], "
                           "but actually is [%s]." %
                           (error_info['param_name'],
                            error_info['param_type'], error_info['actual_type']))

    for dim in shape:
        if not isinstance(dim, int):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['param_type'] = 'int'
            error_info['actual_type'] = dim.__class__.__name__
            raise RuntimeError(error_info,
                               "In op, the parameter[%s]'s type should be [%s],  "
                               "but actually is [%s]." %
                               (error_info['param_name'], error_info['param_type'],
                                error_info['actual_type']))

    if operation.in_dynamic():
        check_dynamic_shape(shape, max_dim, max_rank, param_name)
    else:
        if len(shape) < min_rank or len(shape) > max_rank:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_012
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['min_value'] = min_rank
            error_info['max_value'] = max_rank
            error_info['real_value'] = len(shape)
            raise RuntimeError(error_info,
                               "In op, the num of dimensions of input[%s] should be in"
                               "the range of [%s, %s], but actually is [%s]."
                               % (error_info['param_name'], min_rank, max_rank, len(shape)))

        for _, dim in enumerate(shape):
            if dim < min_dim:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_002
                error_info['op_name'] = OP_NAME
                error_info['param_name'] = param_name
                error_info['min_value'] = min_dim
                error_info['real_value'] = dim
                raise RuntimeError(error_info,
                                   "In op, the dim value[%s] should more than [%s],"
                                   "but actually is [%s]."
                                   % (error_info['param_name'], min_dim, dim))
        if shape:
            shape_size = functools_reduce(lambda x, y: x * y, shape[:])
        else:
            shape_size = 1
        if shape_size < min_size:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_011
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['min_value'] = min_size
            error_info['real_value'] = shape_size
            raise RuntimeError(error_info,
                               "In op, the shape size(product of all dimensions) of "
                               "input[%s] should more than [%s], but actually is [%s]."
                               % (error_info['min_value'], min_size, shape_size))


def check_dtype(dtype, check_list=ALL_DTYPE_LIST, param_name=PARAM_NAME):
    """
    The common check rule for tensor dtype
    """
    if dtype is None:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_007
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        raise RuntimeError(error_info, "In op, the input[%s]'s dtype could not be none." %
                           (error_info['param_name']))

    if not isinstance(dtype, str):
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_003
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['param_type'] = 'str'
        error_info['actual_type'] = dtype.__class__.__name__
        raise RuntimeError(error_info, "In op, the parameter[%s]'s type should be [%s],  "
                                       "but actually is [%s]." %
                           (error_info['param_name'], error_info['param_type'], error_info['actual_type']))
    if dtype.lower() not in check_list:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_008
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['excepted_dtype_list'] = check_list
        error_info['dtype'] = dtype.lower()
        raise RuntimeError(error_info, "In op, the parameter[%s]'s dtype should be one of [%s]"
                                       ", but actually is [%s]."
                           % (error_info['param_name'],
                              error_info['excepted_dtype_list'], error_info['dtype']))


def check_format(data_format, check_list=ALL_FORMAT_LIST, param_name=PARAM_NAME):
    """
    The common check rule for tensor dtype
    """

    if data_format is None:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_017
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        raise RuntimeError(error_info, "In op, the input[%s]'s format could not be none" %
                           (error_info['param_name']))

    if data_format not in check_list:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_015
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['excepted_format_list'] = ",".join(check_list)
        error_info['format'] = data_format
        raise RuntimeError(error_info, "In op, the format of input[%s] should "
                                       "be one of [%s], but actually is [%s]."
                           % (error_info['param_name'],
                               error_info['excepted_format_list'], error_info['format']))


def check_elewise_shape_range(inputs: list, support_broadcast=False):
    """
    :param inputs: list, all inputs of operator
    :return:
    """
    from tbe.dsl.base import operation

    def _has_intersection(range0, range1):
        _range0 = list(range0)
        _range1 = list(range1)
        if _range0[1] is None:
            _range0[1] = MAX_UNKOWN_SHAPE_NUM
        if _range1[1] is None:
            _range1[1] = MAX_UNKOWN_SHAPE_NUM
        return max(_range0[0], _range1[0]) <= min(_range0[1], _range1[1])

    def _check_range_relu(shape_x, shape_y, range_x, range_y):
        size_x = len(shape_x)
        size_y = len(shape_y)
        min_size = min(size_x, size_y)
        for i in range(1, min_size + 1):
            if len(range_x[-i]) != 2 or len(range_y[-i]) != 2:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_023
                error_info['op_name'] = operation.get_context().get_op_type()
                error_info['param_name'] = PARAM_NAME
                raise RuntimeError(error_info,
                                   "In op[%s],the range of each element must be two" % (error_info['op_name']))
            if support_broadcast:
                if (shape_x[-i] != 1 and shape_y[-i] != 1) and \
                        not (_has_intersection(range_x[-i], range_y[-i])
                             or range_x[-i][0] <= 1 or range_y[-i][0] <= 1):
                    error_info = {}
                    error_info['errCode'] = OP_ERROR_CODE_024
                    error_info['op_name'] = operation.get_context().get_op_type()
                    error_info['param_name'] = PARAM_NAME
                    raise RuntimeError(error_info,
                                       "In op[%s],the range at the same location "
                                       "must have intersections" % (error_info['op_name']))
            else:
                if not _has_intersection(range_x[-i], range_y[-i]):
                    error_info = {}
                    error_info['errCode'] = OP_ERROR_CODE_024
                    error_info['op_name'] = operation.get_context().get_op_type()
                    error_info['param_name'] = PARAM_NAME
                    raise RuntimeError(error_info,
                                       "In op[%s],the range at the same location "
                                       "must have intersections" % (error_info['op_name']))

    if len(inputs) <= 1:
        return
    last_shape = None
    last_range = None
    inputs_keys = (OpParamInfoKey.SHAPE, OpParamInfoKey.RANGE)
    for index, _input in enumerate(inputs):
        if not isinstance(_input, dict):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = operation.get_context().get_op_type()
            error_info['param_name'] = PARAM_NAME
            error_info['param_type'] = 'dict'
            error_info['actual_type'] = _input.__class__.__name__
            raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s type should be [%s],  "
                                           "but actually is [%s]." % (error_info['op_name'],
                                                                      error_info['param_name'],
                                                                      error_info['param_type'],
                                                                      error_info['actual_type']))
        for key in inputs_keys:
            if key not in _input.keys():
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_004
                error_info['op_name'] = operation.get_context().get_op_type()
                error_info['param_name'] = PARAM_NAME
                error_info['key'] = OpParamInfoKey.RANGE
                raise RuntimeError(error_info,
                                   "In op[%s], the input[%s] does not contain the item[%s]."
                                   % (error_info['op_name'], error_info['param_name'], error_info['key']))
        shape = _input.get("shape")
        _range = _input.get("range")
        if index > 0:
            _check_range_relu(shape, last_shape, _range, last_range)
        last_shape = shape
        last_range = _range


def squeeze_shape(shape):
    """
    squeeze shape
    """
    warnings.warn("squeeze_shape in the file is deprecated, replace it with the same func in shape_util",
                  DeprecationWarning)
    return shape_util.squeeze_shape(shape)


def wrap_axes_to_positive(axes, rank):
    """
    wrap axis to positive
    """
    warnings.warn("wrap_axes_to_positive is deprecated, replace it with the same func in shape_util",
                  DeprecationWarning)
    return shape_util.wrap_axes_to_positive(axes, rank)


def refine_shape_axes(shape, axes):
    """
    refine shape and axes for reduce ops, fused reduced axes, and fused not reduced axes
    result is a tuple of (shape, axes)
    for example:
        input: shape is (2,3,4,5,6), axes is (1, -3)
        output: (2, 12, 30), (1,)

    Parameters
    ----------
    shape : shape which need refine

    axes : axes which need refine

    Returns
    -------
    shape : list
        refined shape

    axes : list
        refined axes

    """
    warnings.warn("refine_shape_axes is deprecated, replace it with the same func in shape_util",
                  DeprecationWarning)
    return shape_util.refine_shape_axes(shape, axes)


def broadcast_shapes(shape1, shape2, op_name=OP_NAME, param_name_input1='', param_name_input2=''):
    """
    two input shapes produce three output shape
    """
    warnings.warn("refine_shape_axes is deprecated, replace it with the same func in shape_util",
                  DeprecationWarning)
    return shape_util.broadcast_shapes(shape1, shape2, op_name, param_name_input1, param_name_input2)


def refine_shapes_for_broadcast(shape1, shape2):
    """
    Fusing the axes for the input shapes
    """
    warnings.warn("refine_shapes_for_broadcast is deprecated, replace it with the same func in shape_util",
                  DeprecationWarning)
    return shape_util.refine_shapes_for_broadcast(shape1, shape2)


def _equal(expr_a, expr_b):
    """
    :param expr_a:
    :param expr_b:
    :return:
    """
    warnings.warn("_equal is deprecated, replace it with the same func in shape_util",
                  DeprecationWarning)
    return shape_util._equal(expr_a, expr_b)


def _parse_expr(expr, elements: dict):
    warnings.warn("_parse_expr is deprecated, replace it with the same func in shape_util",
                  DeprecationWarning)
    return shape_util._parse_expr(expr, elements)


def _parse_mul(expr, elements: dict):
    warnings.warn("_parse_mul is deprecated, replace it with the same func in shape_util",
                  DeprecationWarning)
    return shape_util._parse_mul(expr, elements)
