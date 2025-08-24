#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.path_manager import PathManager
from msmodel.interface.parser_model import ParserModel
from profiling_bean.db_dto.npu_op_mem_dto import NpuOpMemDto
from profiling_bean.db_dto.npu_op_mem_rec_dto import NpuOpMemRecDto
from profiling_bean.db_dto.npu_module_mem_dto import NpuModuleMemDto
from profiling_bean.db_dto.op_mem_dto import OpMemDto


class NpuAiStackMemTableSelector:
    TableDict = {
        DBNameConstant.TABLE_NPU_OP_MEM_RAW : {
            'dto' : NpuOpMemDto,
            'sql' : "select operator, addr, size, timestamp, thread_id, " \
            "total_allocate_memory, total_reserve_memory, level, type as type_, device_type from {0} " \
            .format(DBNameConstant.TABLE_NPU_OP_MEM_RAW)
        },
        DBNameConstant.TABLE_NPU_OP_MEM : {
            'dto' : OpMemDto,
            'sql' : "select operator, size, allocation_time, release_time, " \
                    "duration, allocation_total_allocated, allocation_total_reserved," \
                    "release_total_allocated, release_total_reserved, device_type, name from {0} " \
                    .format(DBNameConstant.TABLE_NPU_OP_MEM)
        },
        DBNameConstant.TABLE_NPU_OP_MEM_REC : {
            'dto': NpuOpMemRecDto,
            'sql': "select component, timestamp, total_reserve_memory, total_allocate_memory, device_type from {0} " \
                    .format(DBNameConstant.TABLE_NPU_OP_MEM_REC)
        },
        DBNameConstant.TABLE_NPU_MODULE_MEM : {
            'dto': NpuModuleMemDto,
            'sql': "select module_id, syscnt, total_size, device_type from {0} order by module_id asc, syscnt asc " \
                    .format(DBNameConstant.TABLE_NPU_MODULE_MEM)
        }
    }

    @classmethod
    def get_sql(cls: any, table_name: str) -> None:
        return cls.TableDict.get(table_name).get('sql')

    @classmethod
    def get_dto(cls: any, table_name: str) -> None:
        return cls.TableDict.get(table_name).get('dto')


class NpuAiStackMemModel(ParserModel):
    """
    npu op mem model class
    """

    def __init__(self: any, result_dir: str, db_name: str, table_list: list) -> None:
        super().__init__(result_dir, db_name, table_list)

    def flush(self: any, table_name: str, data_list: list) -> None:
        """
        save npu op mem data to db
        :return:
        """
        self.insert_data_to_db(table_name, data_list)

    def get_table_data(self: any, table_name: str) -> list:
        return DBManager.fetch_all_data(self.cur, NpuAiStackMemTableSelector.get_sql(table_name),
                                        dto_class=NpuAiStackMemTableSelector.get_dto(table_name))
