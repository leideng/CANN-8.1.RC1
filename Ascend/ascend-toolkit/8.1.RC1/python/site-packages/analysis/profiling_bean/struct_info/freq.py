#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging
import struct

from msparser.data_struct_size_constant import StructFmt
from msparser.interface.idata_bean import IDataBean


class LpmData:
    """
    lpm data struct
    """
    def __init__(self: any, syscnt: int, freq: int) -> None:
        self._syscnt = syscnt
        self._freq = freq

    @property
    def syscnt(self: any) -> int:
        """
        cycle
        """
        return self._syscnt

    @property
    def freq(self: any) -> int:
        """
        frequency, MHz
        """
        return self._freq


class FreqLpmConvBean(IDataBean):
    """
    Frequency bean data for the data parsing by freq parser.
    """
    COUNT_INDEX = 0
    SYSCNT_BEGIN_INDEX = 2
    FREQ_BEGIN_INDEX = 3
    INTERVAL = 3
    FREQ_DATA_NUM = len(StructFmt.FREQ_FMT)

    def __init__(self: any) -> None:
        self._count = 0
        self._lpm_data = []

    @property
    def count(self: any) -> int:
        """
        lpm data report number
        """
        return self._count

    @property
    def lpm_data(self: any) -> list:
        """
        lpm data
        """
        return self._lpm_data

    def decode(self: any, bin_data: bytes) -> None:
        """
        decode the freq bin data
        :param bin_data: freq bin data
        :return: instance of freq
        """
        if not self.construct_bean(struct.unpack(StructFmt.BYTE_ORDER_CHAR + StructFmt.FREQ_FMT, bin_data)):
            logging.error("freq data struct is incomplete, please check the freq file.")

    def construct_bean(self: any, *args: any) -> bool:
        """
        refresh the freq data
        :param args: freq data
        :return: True or False
        """
        freq_data = args[0]
        if len(freq_data) != self.FREQ_DATA_NUM:
            return False
        self._count = freq_data[self.COUNT_INDEX]
        self._lpm_data = []
        for idx in range(self.count):
            syscnt = freq_data[self.SYSCNT_BEGIN_INDEX + self.INTERVAL * idx]
            freq = freq_data[self.FREQ_BEGIN_INDEX + self.INTERVAL * idx]
            self._lpm_data.append(LpmData(syscnt, freq))
        return True
