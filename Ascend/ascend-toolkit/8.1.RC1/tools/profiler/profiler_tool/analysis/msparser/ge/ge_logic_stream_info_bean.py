#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from profiling_bean.struct_info.struct_decoder import StructDecoder
 
 
class GeLogicStreamInfoBean(StructDecoder):
    """
    ge logic stream info bean
    """
 
    def __init__(self: any, *args: any) -> None:
        data = args[0]
        self._logic_stream_id = data[6]
        physic_stream_num = data[7]
        start_index = 8   # physic_stream_id 从第八个开始存储
        self._physic_logic_stream_id = []
        for index in range(start_index, start_index + physic_stream_num):
            self._physic_logic_stream_id.append([data[index], self._logic_stream_id])
 
    @property
    def logic_stream_id(self: any) -> int:
        """
        for logic stream id
        """
        return self._logic_stream_id
 
    @property
    def physic_logic_stream_id(self: any) -> list:
        """
        for physic stream id
        """
        return self._physic_logic_stream_id

