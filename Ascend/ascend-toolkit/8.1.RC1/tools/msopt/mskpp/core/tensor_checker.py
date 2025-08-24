#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from mskpp._C import arch


TENSOR_DTYPE_MAP = {
    "FP32": "float32",
    "FP16": "float16",
    "BF16": "bfloat16",
    "UINT8": "uint8",
    "INT8": "int8",
    "INT4": "int4",
    "BOOL": "bool",
    "UINT1": "uint1",
    "UINT2": "unit2",
    "INT16": "int16",
    "UINT16": "uint16",
    "INT64": "int64",
    "UINT64": "uint64",
    "INT32": "int32",
    "UINT32": "uint32"
}


class Check:

    def __init__(self, name, inputs, output, support_dtype_dict):
        self.input_tuple = inputs
        self.output_tuple = output
        self.op_name = name
        self.input_dtype_list = self._get_inputs_dtype()
        self.output_dtype_list = self._get_output_dtype()
        self.matrix_dtype_dict = support_dtype_dict
        self.tensor_dtype_map = TENSOR_DTYPE_MAP
        self.input_dtype_list = self.dtype_mapping(self.input_dtype_list)
        self.output_dtype_list = self.dtype_mapping(self.output_dtype_list)

    def dtype_mapping(self, dtype_list):
        return [self.tensor_dtype_map.get(x) for x in dtype_list]

    def _check_inputs_dtype(self, dtype_list):
        if not dtype_list:
            raise Exception("The Matrix Op {} input is none, please check it ".format(self.op_name))

    def _get_inputs_dtype(self):
        if not self.input_tuple:
            raise Exception("The Matrix Op {} input is none, please check it".format(self.op_name))
        dtype_input_list = [x.dtype for x in self.input_tuple]
        self._check_inputs_dtype(dtype_input_list)
        return dtype_input_list

    def _get_output_dtype(self):
        if not self.output_tuple:
            raise Exception("The Matrix Op {} output is none, please check it".format(self.op_name))
        return [self.output_tuple[0].dtype]


class MatrixOperationCheck(Check):

    def __init__(self, name, inputs, output, support_dtype_dict):
        super().__init__(name, inputs, output, support_dtype_dict)

    def check_oder_dtype(self):
        support_dtype_list = self.matrix_dtype_dict.get(self.input_dtype_list[0])
        if support_dtype_list is None:
            raise Exception(
                "{} in Config file is None or invalid. Please check.".format(self.input_dtype_list))

        check_dtype_list = self.input_dtype_list + self.output_dtype_list
        if check_dtype_list not in support_dtype_list:
            raise Exception(
                "For Op {}, only support dtype {}, your dtype is {}. Please check.".format(self.op_name,
                                                                                             support_dtype_list,
                                                                                             check_dtype_list))


def matrix_dtype_check(op_name, input_tuple, output_tuple, support_dtype_dict):
    dtype_check = MatrixOperationCheck(op_name, input_tuple, output_tuple, support_dtype_dict)
    dtype_check.check_oder_dtype()
