#!/usr/bin/python3
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from profiling_bean.stars.stars_common import StarsCommon
from profiling_bean.struct_info.struct_decoder import StructDecoder


class BlockDimBean(StructDecoder):
    """
    block dim for tiling-down
    """

    def __init__(self: any, *args: any) -> None:
        block_dim = args[0]
        self._timestamp = block_dim[4]
        self._stream_id = StarsCommon.set_stream_id(block_dim[5], block_dim[6])
        self._task_id = StarsCommon.set_task_id(block_dim[5], block_dim[6])
        self._block_dim = block_dim[7]

    @property
    def stream_id(self: any) -> int:
        """
        get stream id
        :return: stream id
        """
        return self._stream_id

    @property
    def timestamp(self: any) -> int:
        """
        get timestamp
        :return: timestamp
        """
        return self._timestamp

    @property
    def task_id(self: any) -> int:
        """
        get task id
        :return: task id
        """
        return self._task_id

    @property
    def block_dim(self: any) -> int:
        """
        get block_dim
        :return: block_dim
        """
        return self._block_dim
