#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging
import struct

from common_func.constant import Constant
from msparser.data_struct_size_constant import StructFmt
from profiling_bean.struct_info.struct_decoder import StructDecoder


class NpuMemDataBean(StructDecoder):
    """
    Npu mem data bean for the data parsing by npu mem parser
    """

    def __init__(self: any) -> None:
        self._event = None
        self._ddr = Constant.DEFAULT_VALUE
        self._hbm = Constant.DEFAULT_VALUE
        self._timestamp = Constant.DEFAULT_VALUE

    @property
    def event(self: any) -> str:
        """
        event type
        :return: event type
        """
        return self._event

    @property
    def ddr(self: any) -> int:
        """
        ddr
        :return: ddr
        """
        return self._ddr

    @property
    def hbm(self: any) -> int:
        """
        hbm
        :return: hbm
        """
        return self._hbm

    @property
    def timestamp(self: any) -> int:
        """
        timestamp
        :return: timestamp
        """
        return self._timestamp

    def npu_mem_decode(self: any, bin_data: any) -> any:
        """
        decode the npu mem bin data
        :param bin_data: npu mem bin data
        :return: instance of npu mem
        """
        if self.construct_bean(struct.unpack(StructFmt.NPU_MEM_FMT, bin_data)):
            return self
        return {}

    def construct_bean(self: any, *args: dict) -> bool:
        """
        refresh the npu mem data
        :param args: npu mem bin data
        :return: True or False
        """
        _npu_mem_data = args[0]
        if _npu_mem_data:
            self._event = _npu_mem_data[1]
            self._ddr = _npu_mem_data[3]
            self._hbm = _npu_mem_data[4]
            self._timestamp = _npu_mem_data[0]
            return True
        logging.error("NPU mem data struct is incomplete, please check the npu mem file.")
        return False
