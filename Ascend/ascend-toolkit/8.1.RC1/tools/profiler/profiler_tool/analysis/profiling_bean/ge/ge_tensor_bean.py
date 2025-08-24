#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.

import struct

from common_func.constant import Constant
from common_func.utils import Utils
from profiling_bean.ge.ge_tensor_base_bean import GeTensorBaseBean


class GeTensorBean(GeTensorBaseBean):
    """
    ge tensor info bean
    """
    TENSOR_LEN = 11
    TENSOR_PER_LEN = 8

    def __init__(self: any) -> None:
        super().__init__()
        self._fusion_data = ()
        self._data_tag = Constant.DEFAULT_VALUE
        self._model_id = Constant.DEFAULT_VALUE
        self._index_num = Constant.DEFAULT_VALUE
        self._stream_id = Constant.DEFAULT_VALUE
        self._task_id = Constant.DEFAULT_VALUE
        self._batch_id = Constant.DEFAULT_VALUE
        self._tensor_num = Constant.DEFAULT_VALUE
        self._tensor_type = Constant.DEFAULT_VALUE
        self._timestamp = Constant.DEFAULT_VALUE

    @property
    def model_id(self: any) -> int:
        """
        for model id
        """
        return self._model_id

    @property
    def index_num(self: any) -> int:
        """
        for task id
        """
        return self._index_num

    @property
    def batch_id(self: any) -> int:
        """
        for bacth id
        """
        return self._batch_id

    @property
    def stream_id(self: any) -> int:
        """
        for stream id
        """
        return self._stream_id

    @property
    def task_id(self: any) -> int:
        """
        for task id
        """
        return self._task_id

    @property
    def tensor_num(self: any) -> int:
        """
        for tensor num
        """
        return self._tensor_num

    @property
    def tensor_type(self: any) -> int:
        """
        for tensor type
        """
        return self._tensor_type

    @property
    def timestamp(self: any) -> int:
        """
        for timestamp
        """
        return self._timestamp

    def fusion_decode(self: any, binary_data: bytes) -> any:
        """
        decode ge tensor binary data
        :param binary_data:
        :return:
        """
        fmt = self.get_fmt()
        self.construct_bean(struct.unpack_from(fmt, binary_data))
        return self

    def construct_bean(self: any, *args: any) -> None:
        """
        refresh the ge tensor data
        :param args: ge tensor bean data
        :return: True or False
        """
        self._fusion_data = args[0]
        self._data_tag = self._fusion_data[1]
        self._model_id = self._fusion_data[2]
        self._index_num = self._fusion_data[3]
        self._stream_id = Utils.get_stream_id(self._fusion_data[4])
        self._task_id = self._fusion_data[5]
        self._batch_id = self._fusion_data[6]
        self._tensor_num = self._fusion_data[7]
        self._timestamp = self._fusion_data[63]

        # tensor data is 5, each tensor len is 11
        _tensor_datas = []
        for tensor_index in range(0, self._tensor_num):
            _tensor_datas.append(
                list(self._fusion_data[self.TENSOR_LEN * tensor_index + self.TENSOR_PER_LEN:
                                       self.TENSOR_LEN * tensor_index + (self.TENSOR_LEN + self.TENSOR_PER_LEN)]))
        for _tensor_data in _tensor_datas:
            if _tensor_data[0] == 0:
                self._input_format.append(_tensor_data[1])
                self._input_data_type.append(_tensor_data[2])
                self._input_shape.append(_tensor_data[3:])
            if _tensor_data[0] == 1:
                self._output_format.append(_tensor_data[1])
                self._output_data_type.append(_tensor_data[2])
                self._output_shape.append(_tensor_data[3:])
