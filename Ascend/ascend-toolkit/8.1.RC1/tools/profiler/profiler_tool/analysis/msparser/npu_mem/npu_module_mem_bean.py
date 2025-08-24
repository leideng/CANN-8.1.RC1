# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging
import struct

from profiling_bean.struct_info.struct_decoder import StructDecoder
from msparser.data_struct_size_constant import StructFmt


class NpuModuleMemDataBean(StructDecoder):
    """
    Npu module mem data bean for the data parsing by npu module mem parser
    """

    def __init__(self: any, *args) -> None:
        self._module_id, _, self._cpu_cycle_count, self._total_size = args[0][:4]

    @property
    def module_id(self: any) -> int:
        """
        :return: module_id
        """
        return self._module_id

    @property
    def cpu_cycle_count(self: any) -> int:
        """
        :return: cpu_cycle_count
        """
        return self._cpu_cycle_count

    @property
    def total_size(self: any) -> int:
        """
        :return: total_size
        """
        return self._total_size

