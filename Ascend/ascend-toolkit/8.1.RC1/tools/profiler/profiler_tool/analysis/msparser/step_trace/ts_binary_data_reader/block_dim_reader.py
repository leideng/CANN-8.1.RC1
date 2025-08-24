#!/usr/bin/python3
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from common_func.db_name_constant import DBNameConstant
from profiling_bean.struct_info.block_dim_bean import BlockDimBean


class BlockDimReader:
    """
    class for the real block dim under the scene of tiling-down
    """

    def __init__(self: any) -> None:
        self._data = []
        self._table_name = DBNameConstant.TABLE_BLOCK_DIM

    @property
    def data(self: any) -> list:
        """
        list of block dim data from binary
        """
        return self._data

    @property
    def table_name(self):
        """
        the table name for the block dim
        """
        return self._table_name

    def read_binary_data(self: any, bean_data: any) -> None:
        """
        read block dim binary data and store them into list
        :param bean_data: binary data
        :return: None
        """
        block_dim_bean = BlockDimBean.decode(bean_data)
        if block_dim_bean:
            self._data.append(
                (block_dim_bean.timestamp, block_dim_bean.stream_id,
                 block_dim_bean.task_id, block_dim_bean.block_dim))
