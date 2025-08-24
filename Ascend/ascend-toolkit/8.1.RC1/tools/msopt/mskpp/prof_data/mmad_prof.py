#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.

from math import ceil
from mskpp.core.prof_data import PrefModel, ProfDataRegister
from mskpp._C import prof_data, arch
from mskpp.core.common import checker


@ProfDataRegister.register("MMAD")
class MmadPref(PrefModel):
    def __init__(self, inputs, outputs):
        super(MmadPref, self).__init__("MMAD", inputs, outputs)

    def size(self):
        x, y, b = self.inputs
        if len(x.size) < 2 or len(y.size) < 2:
            raise Exception("The dim of shape is invalid for mmad")
        tile_m = x.size[0]
        tile_k = x.size[1]
        tile_n = y.size[1]
        return tile_m * tile_k * tile_n

    def time(self):
        x, y, b = self.inputs
        granularity = self.size()
        if not checker.check_convert_long_size(granularity):
            raise Exception("The shape size is too large for mmad")
        real_perf = prof_data.MmadData().get(granularity, x.dtype)
        if real_perf <= 0:
            raise Exception("Cannot get running time of {}".format(self.name))
        cycles = ceil(arch.get_size_of(x.dtype) * granularity / real_perf)
        return cycles
