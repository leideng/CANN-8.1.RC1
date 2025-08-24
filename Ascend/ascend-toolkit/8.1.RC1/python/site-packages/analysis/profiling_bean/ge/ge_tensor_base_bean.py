#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging

from common_func.ms_constant.ge_enum_constant import GeDataFormat
from common_func.ms_constant.ge_enum_constant import GeDataType
from common_func.utils import Utils
from profiling_bean.struct_info.struct_decoder import StructDecoder


class GeTensorBaseBean(StructDecoder):
    """
    class for ge tensor base bean
    """

    def __init__(self: any) -> None:
        self._input_format = []
        self._input_data_type = []
        self._input_shape = []
        self._output_format = []
        self._output_data_type = []
        self._output_shape = []

    @property
    def input_format(self: any) -> str:
        """
        for input format
        """
        return ";".join(self._process_tensor_format(self._input_format))

    @property
    def input_data_type(self: any) -> str:
        """
        for input data type
        """
        return ";".join(self._process_tensor_data_type(self._input_data_type))

    @property
    def input_shape(self: any) -> str:
        """
        for input shape
        """
        input_shape = self._reshape_and_filter(self._input_shape, 0)
        return ";".join(input_shape)

    @property
    def output_format(self: any) -> str:
        """
        for output format
        """
        return ";".join(self._process_tensor_format(self._output_format))

    @property
    def output_data_type(self: any) -> str:
        """
        for output data type
        """
        return ";".join(self._process_tensor_data_type(self._output_data_type))

    @property
    def output_shape(self: any) -> str:
        """
        for output shape
        """
        output_shape = self._reshape_and_filter(self._output_shape, 0)
        return ";".join(output_shape)

    @staticmethod
    def _process_with_sub_format(tensor_format: int) -> tuple:
        """
        get the real tensor format and tensor sub format,
        real tensor_format need operate with 0xff when tensor sub format exist
        :param tensor_format:
        :return:
        """
        if tensor_format == GeDataFormat.UNDEFINED.value:
            return tensor_format, 0
        return tensor_format & 0xff, (tensor_format & 0xffff00) >> 8

    @staticmethod
    def _process_tensor_data_type(data_type: list) -> list:
        enum_dict = GeDataType.member_map()
        return [enum_dict.get(_formate, GeDataType.UNDEFINED).name for _formate in data_type]

    @classmethod
    def _reshape_and_filter(cls: any, shape_data: list, filter_num: int) -> list:
        res_shape = []
        for single_shape in shape_data:
            _tmp_shape = []
            for _shape in single_shape:
                if _shape != filter_num:
                    _tmp_shape.append(str(_shape))
            res_shape.append(_tmp_shape)
        _res_shape_str_list = Utils.generator_to_list(",".join(i) for i in res_shape)
        return _res_shape_str_list

    @classmethod
    def _process_tensor_format(cls: any, _input_format) -> list:
        enum_dict = GeDataFormat.member_map()
        for index, _format in enumerate(_input_format):
            tensor_format, tensor_sub_format = cls._process_with_sub_format(_format)
            if tensor_format not in enum_dict:
                logging.error("Unsupported tensor format %d", tensor_format)
                _input_format[index] = str(_input_format[index])
                continue
            enum_format = enum_dict.get(tensor_format).name
            if tensor_sub_format > 0:
                enum_format = '{0}:{1}'.format(enum_format, str(tensor_sub_format))
            _input_format[index] = enum_format
        return _input_format
