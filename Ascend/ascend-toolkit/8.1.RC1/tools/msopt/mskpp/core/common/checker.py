#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import logging
import re
import os

valid_mem_type = ["GM", "UB", "L1", "L0A", "L0B", "L0C", "FB", "BT"]
valid_dtype = [
    # full dtype
    ["BOOL", "UINT1", "UINT2", "UINT8", "UINT16", "UINT32", "BF16",
     "UINT64", "INT4", "INT8", "INT16", "INT32", "INT64", "FP16", "FP32"],
    # basic dtype
    ["FP16", "FP32", "INT16", "INT32"]
]
valid_format = ["NCHW", "NHWC", "ND", "NZ", "NC1HWC0", "FRACTAL", "FRACTAL_NZ"]
valid_gm_type = ["HBM", "L2Cache"]
LARGE_LONG_SIZE_THRESHOLD = 2 ** 31 - 1
MAX_NAME_LENGTH = 128
NAME_PATTERN = r"^[A-Za-z0-9]+$"


def is_required_type(param, data_type):
    return isinstance(param, data_type)


def is_int_type(param):
    return (not isinstance(param, bool)) and isinstance(param, int)


def is_mem_type_valid(param):
    return param in valid_mem_type


def is_dtype_valid(param_dtype, valid_dtype_choice=0):
    return param_dtype in valid_dtype[valid_dtype_choice]


def is_shape_valid(param_shape_list):
    if not is_required_type(param_shape_list, list):
        return False
    if not param_shape_list:
        return False
    size = 1
    for dim in param_shape_list:
        if not is_int_type(dim) or dim < 0:
            return False
        size *= dim
    return check_convert_long_size(size)


def is_format_valid(param_format):
    return param_format in valid_format


def is_gm_type_valid(param_gm_type):
    return param_gm_type in valid_gm_type


def check_convert_long_size(size):
    return size <= LARGE_LONG_SIZE_THRESHOLD


def check_name_valid(name, name_type="name"):
    if not is_required_type(name, str):
        raise Exception("Parameter {} should be str, please modify it".format(name_type))
    if not name:
        raise Exception("Parameter {} should not be empty, please modify it".format(name_type))
    if len(name) > MAX_NAME_LENGTH:
        raise Exception("The length of parameter {} exceeds {}, please modify it".format(name_type, MAX_NAME_LENGTH))
    name_pattern = re.compile(NAME_PATTERN)
    match = name_pattern.match(name)
    if match is None:
        raise Exception("Parameter {} is invalid, please modify it".format(name_type))


def check_output_path(output_path):
    if output_path == "":
        raise Exception("The output path should not be empty")
    real_path = os.path.realpath(output_path)
    if not check_path_exists(real_path):
        raise Exception("The output path {} does not exist".format(real_path))
    file_owner = os.stat(real_path).st_uid
    if file_owner != os.getuid():
        raise Exception("The output path {} belong to others".format(real_path))
    if not os.access(real_path, os.W_OK):
        raise Exception("The output path {} does not have permission to write".format(real_path))


def check_path_exists(path):
    return os.path.exists(path)


class InstrInitParaCheck:
    @classmethod
    def check_args_key(cls, key_list, kwargs):
        for key in key_list:
            if kwargs[key] is None:
                raise Exception("args {} is None, please check instr input params".format(key))

    @classmethod
    def check_type(cls, args, args_key, type_name):
        cls.check_args_key([args_key], args)
        args_value = args[args_key]
        if type_name == "bool":
            if not is_required_type(args_value, bool):
                raise Exception("init param {} should be {}, but got {}".format(args_key, type_name, type(args_value)))
        if type_name == "int":
            if not is_int_type(args_value):
                raise Exception("init param {} should be {}, but got {}".format(args_key, type_name, type(args_value)))

    @classmethod
    def check_dtype_valid(cls, args, args_dtype):
        cls.check_args_key([args_dtype], args)
        if not is_dtype_valid(args[args_dtype]):
            raise Exception("instr param dtype is invalid. Please check")

    @classmethod
    def check_shape_valid(cls, args, args_fill_shape):
        cls.check_args_key([args_fill_shape], args)
        if not is_shape_valid(args[args_fill_shape]):
            raise Exception("instr param fill_shape is invalid. It should be a int element list.")

    @classmethod
    def check_tensor_is_complete(cls, tensor):
        if not tensor.is_complete():
            raise Exception("instr's input Tensor is not complete.")