#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from common_func.constant import Constant
from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.str_constant import StrConstant
from msmodel.interface.parser_model import ParserModel
from msmodel.interface.view_model import ViewModel


class ParallelModel(ParserModel):
    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_PARALLEL,
                         [DBNameConstant.TABLE_PARALLEL_STRATEGY, DBNameConstant.TABLE_HCCL_OPERATOR_OVERLAP,
                          DBNameConstant.TABLE_COMPUTATION_TIME])

    def flush(self: any, table_name: str, data_list: list) -> None:
        """
        flush data to db
        """
        self.insert_data_to_db(table_name, data_list)


class ParallelViewModel(ViewModel):
    def __init__(self: any, path: str) -> None:
        super().__init__(path, DBNameConstant.DB_PARALLEL, [])

    def get_parallel_table_name(self: any) -> str:
        sql = "select parallel_mode from {}".format(DBNameConstant.TABLE_PARALLEL_STRATEGY)
        data_list = DBManager.fetch_all_data(self.cur, sql)
        if not data_list:
            return Constant.NA
        elif not data_list[0]:
            return Constant.NA
        return StrConstant.PARALLEL_TABLE_NAME_MAPPING.get(data_list[0][0], Constant.NA)

    def get_parallel_index_data(self: any, tabel_name: str, rank_id: any, device_id: int, hwts_freq: int) -> list:
        freq_to_us = 1000000 / hwts_freq
        if rank_id == Constant.DEFAULT_INVALID_VALUE:
            rank_id = "null"
        if tabel_name == DBNameConstant.TABLE_CLUSTER_DATA_PARALLEL:
            sql = "SELECT {0} rank_id, {1} device_id, t1.model_id, t1.index_id, " \
                  "t2.step_time*{2}, t2.computation_time*{2}, " \
                  "t1.pure_communication_time*{2}, t1.communication_time*{2}, " \
                  "(t1.all_communication_time-t1.communication_time)*{2}, " \
                  "t1.hccl_op_num FROM(" \
                  "SELECT model_id, index_id, sum( end_time - start_time) communication_time, " \
                  "sum( end_time - start_time - overlap_time) pure_communication_time, " \
                  "max(end_time) - min(start_time) all_communication_time, count(0) hccl_op_num " \
                  "FROM {3} GROUP BY model_id, index_id ) t1 LEFT JOIN {4} t2 ON t1.model_id = t2.model_id " \
                  "AND t1.index_id = t2.index_id".format(rank_id, device_id, freq_to_us,
                                                         DBNameConstant.TABLE_HCCL_OPERATOR_OVERLAP,
                                                         DBNameConstant.TABLE_COMPUTATION_TIME)
        elif tabel_name == DBNameConstant.TABLE_CLUSTER_MODEL_PARALLEL:
            sql = "select {0} rank_id, {1} device_id, t1.model_id, t1.index_id, " \
                  "t2.step_time*{2}, t2.computation_time*{2}, " \
                  "t1.pure_communication_time*{2}, t1.communication_time*{2} " \
                  "from(select model_id, index_id, " \
                  "sum(end_time-start_time) communication_time, " \
                  "sum(end_time-start_time-overlap_time) pure_communication_time FROM {3} group by " \
                  "model_id, index_id ) t1 left join {4} t2 ON t1.model_id = t2.model_id and " \
                  "t1.index_id = t2.index_id".format(rank_id, device_id, freq_to_us,
                                                     DBNameConstant.TABLE_HCCL_OPERATOR_OVERLAP,
                                                     DBNameConstant.TABLE_COMPUTATION_TIME)
        else:
            sql = "select {0} rank_id, {1} device_id, t1.model_id, t1.index_id, " \
                  "t2.step_time*{2}, t2.computation_time*{2}, " \
                  "t1.pure_communication_time*{2}, t1.communication_time*{2}, " \
                  "t1.pure_communication_time_only_revice*{2}, t1.pure_communication_time_except_revice*{2} " \
                  "from(select model_id, index_id, " \
                  "sum(end_time-start_time) communication_time, " \
                  "sum(end_time-start_time-overlap_time) pure_communication_time, " \
                  "sum(case when op_type='Receive' then end_time-start_time-overlap_time else 0 end) " \
                  "pure_communication_time_only_revice, sum(case when op_type<>'Receive' then " \
                  "end_time-start_time-overlap_time else 0 end) pure_communication_time_except_revice from {3} " \
                  "group by model_id, index_id)t1 left join {4} t2 ON t1.model_id = t2.model_id " \
                  "and t1.index_id = t2.index_id".format(rank_id, device_id, freq_to_us,
                                                         DBNameConstant.TABLE_HCCL_OPERATOR_OVERLAP,
                                                         DBNameConstant.TABLE_COMPUTATION_TIME)
        return DBManager.fetch_all_data(self.cur, sql)

    def get_parallel_strategy_data(self: any) -> list:
        sql = "select * from {}".format(DBNameConstant.TABLE_PARALLEL_STRATEGY)
        return DBManager.fetch_all_data(self.cur, sql)
