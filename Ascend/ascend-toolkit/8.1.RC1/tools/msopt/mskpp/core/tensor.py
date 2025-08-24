#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from functools import reduce
from mskpp.core.memory_instruction import MemoryInstruction
from mskpp.core.common import checker


class Tensor:
    '''
    定义变量的地址和大小
    '''
    def __init__(self, mem_type, dtype=None, size=None, format=None, is_inited=False):
        self.mem_type = mem_type
        self.dtype = dtype
        self.size = size
        self.format = format
        self.is_inited = is_inited
        self.param_check(self)
        self.valid = True if mem_type == "GM" or is_inited else False
        self.father_tensor = None
        self.tensor_value = -1       # 用于标记当前tensor的写入值，初始为-1，在对应任务调度后需要写入当前tensor的值

    def __str__(self):
        msg = "{}({},{},{})".format(self.mem_type, self.size, self.dtype, self.format)
        return msg

    def __getitem__(self, key):
        """
        :param key: support slice, tuple(int), tuple(slice)
        :return:
        """
        new_shape = []
        if isinstance(key, slice):
            # key type is slice, only reshape first dimension of self.size
            if len(self.size) < 1:
                raise Exception("The size of Tensor.size should be at least 1 when key type is slice.")
            new_shape = list(self.size)
            start, stop, step = key.indices(self.size[0])
            num_item = len(range(start, stop, step))
            new_shape[0] = num_item
        elif isinstance(key, tuple):
            if len(self.size) < len(key):
                raise Exception("The size of tuple key should not be greater than the size of Tensor.size.")
            size_index = 0
            for idx, key_value in enumerate(key):
                if not isinstance(key_value, slice) and not isinstance(key_value, int):
                    raise TypeError("Not supported key type, should be slice, tuple(int), tuple(slice).")
                size_index = idx
                # key type is tuple(slice)
                if isinstance(key_value, slice):
                    start, stop, step = key_value.indices(self.size[idx])
                    num_item = len(range(start, stop, step))
                    new_shape.append(num_item)
                    continue
                # key type is tuple(int)
                if key_value < 0 or self.size[idx] <= key_value:
                    raise Exception("The value of tuple(int) should not be less than 0,"
                                    " and should be smaller than the value of Tensor.size in the same dimension.")
            if size_index < len(self.size):
                new_shape.extend(self.size[size_index + 1:])
        else:
            raise TypeError("Not supported key type, should be slice, tuple(int), tuple(slice).")
        new_tensor = Tensor(self.mem_type, self.dtype, new_shape, self.format, self.is_inited)
        new_tensor.valid = self.valid
        new_tensor.father_tensor = self
        return new_tensor

    @staticmethod
    def param_check(tensor, empty_allow=True):
        if not empty_allow:
            if tensor.size is None:
                raise Exception('The shape of tensor is None')
            if tensor.dtype is None:
                raise Exception('The dtype of tensor is None')
            if tensor.format is None:
                raise Exception('The format of tensor is None')
        if not checker.is_mem_type_valid(tensor.mem_type):
            raise Exception("Tensor parameter mem_type invalid. Only the following mem_type are supported: {}"
                            .format(checker.valid_mem_type))
        if (tensor.dtype is not None) and (not checker.is_dtype_valid(tensor.dtype, 0)):
            raise Exception("Tensor parameter dtype invalid. Only the following dtype are supported: {}"
                            .format(checker.valid_dtype[0]))
        if (tensor.size is not None) and (not checker.is_shape_valid(tensor.size)):
            raise Exception("Tensor parameter size is invalid. Please check.")
        if (tensor.format is not None) and (not checker.is_format_valid(tensor.format)):
            raise Exception("Tensor parameter format invalid. Only the following format are supported: {}"
                            .format(checker.valid_format))
        if not checker.is_required_type(tensor.is_inited, bool):
            raise Exception("Tensor parameter is_inited should be bool, but got: {}".format(type(tensor.is_inited)))

    def is_complete(self):
        return self.dtype and self.size and self.format

    def is_valid(self):
        return self.valid

    def set_invalid(self):
        self.valid = False

    def set_valid(self):
        self.valid = True

    def load(self, tensor, repeat=1, set_value=-1, expect_value=-1):
        """
        将数据从目标地址搬运到此处
        :param tensor: 需要加载的tensor，与当前的tensor构成一个mov指令
        :param repeat: mov类指令中可外部指定的repeat，默认为1
        :param set_value: 在触发当前任务后，需要向tensor中写入的值
        :param expect_value: 在触发当前任务前，需要检测tensor中写入的值是否符合预期
        :return:
        """
        if not checker.is_required_type(tensor, Tensor):
            raise Exception("Tensor.load input type should be Tensor, but got {}".format(type(tensor)))
        self.param_check(tensor, False)
        if (not checker.is_int_type(set_value)) or (not checker.is_int_type(expect_value)):
            raise Exception("Tensor.load method set_value and expect_value should be int, but got: "
                            "set_value is {} and expect_value is {}".format(type(set_value), type(expect_value)))
        trans_enable = self.trans_format(tensor)

        if self.dtype and tensor.dtype != self.dtype:
            raise Exception("dtype({} and {}) is not match".format(self.dtype, tensor.dtype))
        else:
            self.dtype = tensor.dtype
        if (not checker.is_int_type(repeat)) or (repeat < 1):  # repeat有效值为正整数
            raise Exception("input repeat = {} invalid.".format(repeat))

        if self.size is None:  # 空对象
            self.size = tensor.size

        origin_size = reduce(lambda x, y: x * y, self.size)
        expect_size = reduce(lambda x, y: x * y, tensor.size)
        if origin_size < expect_size:
            raise Exception("The size of tensor({}) is less than expect size({}).".format(self.size, tensor.size))
        MemoryInstruction((tensor), (self), trans_enable, repeat, set_value, expect_value)()

    def trans_format(self, tensor):
        if not self.format:
            # format为None表示不使能随路转换
            self.format = tensor.format
            return False
        if tensor.format == self.format:
            return False

        if tensor.mem_type == "GM" and self.mem_type == "L1":
            if tensor.format == "ND" and self.format == "NZ":
                return True
            elif tensor.format == "NHWC" and self.format == "NC1HWC0":
                return True
            else:
                raise Exception("GM's tensor format({}) can not be transformed to L1's tensor format({})"
                                .format(tensor.format, self.format))

        elif tensor.mem_type == "L0C" and self.mem_type == "GM":
            if tensor.format == "NZ" and self.format == "ND":
                return True
            elif tensor.format == "NC1HWC0" and self.format == "NHWC":
                return True
            else:
                raise Exception("L0C's tensor format({}) can not be transformed to GM's tensor format({})"
                                .format(tensor.format, self.format))

        else:
            # 除上述可随路转换的空间和format外，其余format需相同
            raise Exception("format({} and {}) do not match, and these tensors' format can not be transformed in chips"
                            .format(self.format, tensor.format))
