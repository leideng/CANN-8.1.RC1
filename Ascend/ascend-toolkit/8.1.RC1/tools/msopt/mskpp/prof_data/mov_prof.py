#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.


from functools import reduce
from math import ceil
from mskpp.core.prof_data import PrefModel, ProfDataRegister
from mskpp._C import prof_data, arch
from mskpp.core.common import checker


@ProfDataRegister.register("MOV")
class MovPref(PrefModel):
    def __init__(self, inputs, outputs, trans_enable, repeat):
        super(MovPref, self).__init__("MOV", inputs, outputs)
        self.trans_enable = trans_enable
        self.repeat = repeat

    @staticmethod
    def cache_hit_ratio(src, dst, bandwidth):
        if (src == "GM" and dst == "L1") or (dst == "GM"):
            cache_hit_ratio = arch.get_cache_hit_ratio()
            # 实测数据一般L2cache为0和l2cache为1时带宽差2倍+，理论值为4.5倍
            return cache_hit_ratio * bandwidth + (1 - cache_hit_ratio) * bandwidth / 4.5
        return bandwidth

    def size(self):
        src = self.inputs[0]
        shape_size = reduce(lambda x, y: int(x) * int(y), src.size)
        data_move_size = shape_size * arch.get_size_of(src.dtype)
        return data_move_size

    def is_support_repeat(self, src_mem, dst_mem):
        if arch.get() != "ascend910b1":
            return False
        access_road = src_mem + "_TO_" + dst_mem
        repeat_support_list = ["GM_TO_L0A", "GM_TO_L0B", "L1_TO_L0A", "L1_TO_L0B"]
        return access_road in repeat_support_list

    def time(self):
        src = self.inputs[0]
        dst = self.outputs[0]
        data_move_size = self.size()
        if self.is_support_repeat(src.mem_type, dst.mem_type):
            bandwidth = prof_data.MovData().get_repeat(src.mem_type, dst.mem_type, self.repeat)
        else:
            if not checker.check_convert_long_size(data_move_size):
                raise Exception("The shape size is too large for move")
            bandwidth = prof_data.MovData().get(src.mem_type, dst.mem_type, data_move_size, self.trans_enable)
        bandwidth = MovPref.cache_hit_ratio(src.mem_type, dst.mem_type, bandwidth)
        if bandwidth <= 0:
            raise Exception(f"wrong bandwidth value = {bandwidth}")
        cycles = ceil(data_move_size / bandwidth)
        return cycles
