#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from msmodel.interface.parser_model import ParserModel
from msparser.npu_mem.npu_mem_dto import NpuMemDto


class NpuMemModel(ParserModel):
    """
    npu mem model class
    """

    def __init__(self: any, result_dir: str, db_name: str, table_list: list) -> None:
        super().__init__(result_dir, db_name, table_list)
        self.conn = None
        self.cur = None

    def flush(self: any, npu_mem_data: list) -> None:
        """
        save npu mem data to db
        :return:
        """
        self.insert_data_to_db(DBNameConstant.TABLE_NPU_MEM, npu_mem_data)

    def get_timeline_data(self: any) -> list:
        npu_mem_sql = "select timestamp, event, ddr, hbm, memory from {0} " \
            .format(DBNameConstant.TABLE_NPU_MEM)
        return DBManager.fetch_all_data(self.cur, npu_mem_sql, dto_class=NpuMemDto)

    def get_summary_data(self: any) -> list:
        return self.get_timeline_data()
