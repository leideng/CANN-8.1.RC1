#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from abc import ABC, abstractmethod
from ..core.computation_instruction import ComputationInstruction
from .instr_strategy import execute_func_by_instr_name
from ..core.tensor_checker import matrix_dtype_check

MMAD_SUPPORT_DTYPE_DICT = {
    "uint8": [["uint8", "uint8", "uint32", "uint32"], ["uint8", "int8", "int32", "int32"]],
    "int8": [["int8", "int8", "int32", "int32"]],
    "float16": [["float16", "float16", "float16", "float16"], ["float16", "float16", "float32", "float32"],
                ["float16", "uint2", "float16", "float16"]]}


class InstrInferBase(ABC):  # 指令在输出推导时应该实现的策略接口,用以确定指令计算后tensor输出的size、format、dtype等属性
    @abstractmethod
    def instr_infer_shape(self, inputs, outputs, attr, instr_name=None):
        pass


class MmadInfer(InstrInferBase):
    def instr_infer_shape(self, inputs, outputs, attr, instr_name=None):
        x, y, b = inputs
        if len(y.size) < 2:
            raise Exception("The dim of y is invalid for mmad. dim must be larger than 1")
        outputs[0].size = [x.size[0], y.size[1]]
        outputs[0].format = b.format if b.format != "ND" else x.format
        outputs[0].dtype = b.dtype
        matrix_dtype_check("mmad", inputs, outputs, MMAD_SUPPORT_DTYPE_DICT)


class TransInfer(InstrInferBase):
    def instr_infer_shape(self, inputs, outputs, attr, instr_name=None):
        x = inputs[0]
        res = 1
        for i in inputs[0].size:
            res *= i
        if res != 256:
            raise ValueError('input shape size not equal 256')
        outputs[0].size = x.size
        outputs[0].format = x.format
        outputs[0].dtype = x.dtype


class VecInferByType(InstrInferBase):
    @staticmethod
    def _get_type_by_dtype(attr, output):
        if "dtype" not in attr:
            raise ValueError('instr input attr lack of dtype')
        output.dtype = attr.get("dtype")

    @staticmethod
    def _get_type_by_inputs0(inputs0, output):
        if inputs0.dtype == "FP16" or inputs0.dtype == "INT16":
            output.dtype = "INT8"
        elif inputs0.dtype == "FP32":
            output.dtype = "FP16"
        else:
            raise TypeError("Input dtype is not support for addreluconv instr")

    @staticmethod
    def _get_type_by_inputs0v2(inputs0, output):
        output.dtype = "FP16" if (inputs0.dtype == "FP32") else "INT8"

    @staticmethod
    def _get_type_by_ifmix(inputs0, attr, output):
        if attr["if_mix"]:
            output.dtype = "FP32"
            return
        output.dtype = inputs0.dtype

    def instr_infer_shape(self, inputs, outputs, attr, instr_name=None):
        x = inputs[0]
        outputs[0].size = x.size
        outputs[0].format = x.format
        outputs[0].dtype = x.dtype
        instr_type_func = {
            ("VCONV", "VCONVDEQ", "VCONVVDEQ", "VMULCONV"): (self._get_type_by_dtype, attr, outputs[0]),
            ("VADDRELUCONV", ): (self._get_type_by_inputs0, x, outputs[0]),
            ("VSUBRELUCONV", ): (self._get_type_by_inputs0v2, x, outputs[0]),
            ("VAXPY", "VMLA"): (self._get_type_by_ifmix, x, attr, outputs[0])
        }
        execute_func_by_instr_name(instr_type_func, instr_name)


class VecInferBySize(InstrInferBase):
    @staticmethod
    def _set_size_by_attr(outputs, attr, name):
        if name not in attr:
            raise ValueError("instr input attr lack of {}".format(name))
        outputs.size = [attr[name]]

    @staticmethod
    def _set_size_by_double(outputs, input_size):
        res = []
        for i in input_size:
            res.append(i * 2)
        outputs.size = res

    @staticmethod
    def _set_size_by_broadcast(outputs, input_size, attr, broadcast):
        if broadcast not in attr:
            raise ValueError("instr input attr lack of {}".format(broadcast))
        outputs.size = input_size
        outputs.size[-1] *= attr.get(broadcast)

    @staticmethod
    def _set_size_by_reduce(outputs, input_size, attr, reduce):
        if reduce not in attr:
            raise ValueError("instr input attr lack of {}".format(reduce))
        outputs.size = input_size
        if attr.get(reduce) == 0:
            raise ZeroDivisionError("Reduce num can not be zero!")
        outputs.size[-1] = outputs.size[-1] // attr.get(reduce)
        if len(outputs.size) > 1 and (outputs.size[-1] <= 1):
            outputs.size.pop(-1)

    @staticmethod
    def _set_size_by_fill_shape(outputs, attr, fill_shape):
        if fill_shape not in attr:
            raise ValueError("instr input attr lack of {}".format(fill_shape))
        outputs.size = attr.get(fill_shape)

    def instr_infer_shape(self, inputs, outputs, attr, instr_name=None):
        x = inputs[0]
        outputs[0].format = x.format
        outputs[0].dtype = x.dtype
        instr_size_func = {
            ("VREDUCE", "VREDUCEV2"): (self._set_size_by_attr, outputs[0], attr, "reserve_num"),
            ("VBITSORT", ): (self._set_size_by_double, outputs[0], inputs[0].size),
            ("VBRCB", ): (self._set_size_by_broadcast, outputs[0], x.size, attr, "broadcast_num"),
            ("VCADD", "VCGADD", "VCGMAX", "VCGMIN", "VCMAX", "VCMIN", "VCPADD"):
                (self._set_size_by_reduce, outputs[0], x.size, attr, "reduce_num"),
            ("VECTORDUP", ): (self._set_size_by_fill_shape, outputs[0], attr, "fill_shape")
        }
        execute_func_by_instr_name(instr_size_func, instr_name)


class MmadInferInvoke(ComputationInstruction):
    def infer_shape(self, inputs, outputs, attr):
        mmad_infer = MmadInfer()
        mmad_infer.instr_infer_shape(inputs, outputs, attr, self.name)


class VecInferByTypeInvoke(ComputationInstruction):
    def infer_shape(self, inputs, outputs, attr):
        vec_infer_in_diff_type = VecInferByType()
        vec_infer_in_diff_type.instr_infer_shape(inputs, outputs, attr, self.name)


class TransInferInvoke(ComputationInstruction):
    def infer_shape(self, inputs, outputs, attr):
        trans_infer = TransInfer()
        trans_infer.instr_infer_shape(inputs, outputs, attr, self.name)


class VecInferBySizeInvoke(ComputationInstruction):
    def infer_shape(self, inputs, outputs, attr):
        vec_infer_in_diff_size = VecInferBySize()
        vec_infer_in_diff_size.instr_infer_shape(inputs, outputs, attr, self.name)
