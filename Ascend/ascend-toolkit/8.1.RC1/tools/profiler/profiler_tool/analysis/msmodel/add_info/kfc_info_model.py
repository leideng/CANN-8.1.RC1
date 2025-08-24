#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from collections import namedtuple
from dataclasses import dataclass

from common_func.db_name_constant import DBNameConstant
from common_func.msprof_object import CustomizedNamedtupleFactory
from common_func.db_manager import DBManager
from msmodel.interface.parser_model import ParserModel
from msmodel.interface.view_model import ViewModel
from mscalculate.flip.flip_calculator import FlipCalculator


class KfcInfoModel(ParserModel):
    """
    kfc info model class
    """

    def __init__(self: any, result_dir: str, table_list: list) -> None:
        super().__init__(result_dir, DBNameConstant.DB_KFC_INFO, table_list)

    def flush(self: any, data_list: list, table_name: str = DBNameConstant.TABLE_KFC_INFO) -> None:
        """
        insert data to table
        :param data_list: hccl information data
        :param table_name: table name
        :return:
        """
        self.insert_data_to_db(table_name, data_list)


class KfcInfoViewModel(ViewModel):
    KFC_HCCL_INFO_TYPE = CustomizedNamedtupleFactory.enhance_namedtuple(
        namedtuple("KfcInfo",
                   ["timestamp", "op_name", "ccl_tag", "group_name", "local_rank", "remote_rank",
                    "rank_size", "work_flow_mode", "plane_id", "context_id", "notify_id", "stage", "role",
                    "duration_estimated", "src_addr", "dst_addr", "size", "op_type", "data_type",
                    "link_type", "transport_type", "rdma_type", "stream_id", "task_id", "batch_id",
                    "start_time", "duration", "bandwidth", "device_task_type", "ts_virtual_batch_id"]),
        {})
    KFC_COMM_TURN_TYPE = CustomizedNamedtupleFactory.enhance_namedtuple(
        namedtuple("KfcCommTurn",
                   ["device_id", "stream_id", "task_id", "comm_turn", "current_turn", "server_start_time",
                    "wait_msg_start_time", "kfc_alg_exe_start_time", "send_task_start_time",
                    "send_sqe_finish_time", "rtsq_exe_end_time", "server_end_time"]),
        {})
    KFC_COMPUTE_TURN_TYPE = CustomizedNamedtupleFactory.enhance_namedtuple(
        namedtuple("KfcComputeTurn",
                   ["device_id", "stream_id", "task_id", "compute_turn", "current_turn", "wait_compute_start_time",
                    "compute_start_time", "compute_exe_end_time"]),
        {})

    HCCL_OP_INFO_TYPE = CustomizedNamedtupleFactory.enhance_namedtuple(
        namedtuple("HcclOpInfo",
                   ["timestamp", "relay", "retry", "data_type", "alg_type", "count",
                    "group_name", "stream_id", "task_id", "rank_size", "source"]),
        {})

    HCCL_OP_MASTER_STREAM_TYPE = CustomizedNamedtupleFactory.enhance_namedtuple(
        namedtuple("HcclOpMasterStreamType",
                   ["timestamp", "stream_id", "task_id", "hccl_stream_id", "hccl_task_id",
                    "batch_id", "hccl_batch_id", "task_type"]),
        {})

    MASTER_STREAM_HCCL_TASK_TYPE = CustomizedNamedtupleFactory.enhance_namedtuple(
        namedtuple("MasterStreamHcclTaskType",
                   ["timestamp", "aicpu_stream_id", "aicpu_task_id", "stream_id", "task_id",
                    "aicpu_batch_id", "batch_id", "task_type"]),
        {})

    AICPU_TASK_FLIP = CustomizedNamedtupleFactory.enhance_namedtuple(
        namedtuple("AicpuTaskFlip",
                   ["stream_id", "timestamp", "task_id", "flip_num"]),
        {})

    def __init__(self, result_dir: str, table_list: list):
        super().__init__(result_dir, DBNameConstant.DB_KFC_INFO, table_list)

    def get_kfc_info_data(self: any) -> list:
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_KFC_INFO):
            return []
        sql = "select timestamp, op_name, ccl_tag, group_name, local_rank, remote_rank, rank_size, work_flow_mode, " \
              "plane_id, context_id, notify_id, stage, role, duration_estimated, src_addr, dst_addr, size, op_type, " \
              "data_type, link_type, transport_type, rdma_type, stream_id, task_id, " \
              "0 as batch_id, 0 as start_time, 0 as duration, 0 as bandwidth, 'N/A' as device_task_type, " \
              "-1 as ts_virtual_batch_id from {}".format(DBNameConstant.TABLE_KFC_INFO)
        kfc_info_data = self.get_sql_data(sql)
        return [self.KFC_HCCL_INFO_TYPE(*data) for data in kfc_info_data]

    def get_kfc_comm_turn_data(self: any) -> list:
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_KFC_COMM_TURN):
            return []
        sql = "select * " \
              "from {}".format(DBNameConstant.TABLE_KFC_COMM_TURN)
        kfc_info_data = self.get_sql_data(sql)
        return [self.KFC_COMM_TURN_TYPE(*data) for data in kfc_info_data]

    def get_kfc_compute_turn_data(self: any) -> list:
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_KFC_COMPUTE_TURN):
            return []
        sql = "select * " \
              "from {}".format(DBNameConstant.TABLE_KFC_COMPUTE_TURN)
        kfc_info_data = self.get_sql_data(sql)
        return [self.KFC_COMPUTE_TURN_TYPE(*data) for data in kfc_info_data]

    def get_hccl_op_info_data(self: any) -> list:
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_DEVICE_HCCL_OP_INFO):
            return []
        sql = "select timestamp, relay, retry, data_type, alg_type, " \
              "count, group_name, stream_id, task_id, rank_size, source" \
              " from {} order by timestamp".format(DBNameConstant.TABLE_DEVICE_HCCL_OP_INFO)
        hccl_op_info = self.get_sql_data(sql)
        return [self.HCCL_OP_INFO_TYPE(*data) for data in hccl_op_info]

    def get_aicpu_master_stream_hccl_task(self: any) -> list:
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_AICPU_MASTER_STREAM_HCCL_TASK):
            return []
        sql = "select timestamp, aicpu_stream_id, aicpu_task_id, stream_id, task_id, " \
              "0 as aicpu_batch_id, 0 as batch_id, type " \
              "from {} order by timestamp".format(DBNameConstant.TABLE_AICPU_MASTER_STREAM_HCCL_TASK)
        aicpu_master_stream_hccl_task = self.get_sql_data(sql)
        aicpu_master_stream_hccl_task = [self.HCCL_OP_MASTER_STREAM_TYPE(*data)
                                         for data in aicpu_master_stream_hccl_task]
        aicpu_master_stream_hccl_task = FlipCalculator.set_device_batch_id(aicpu_master_stream_hccl_task,
                                                                           self.result_dir)
        return [self.MASTER_STREAM_HCCL_TASK_TYPE(*data) for data in aicpu_master_stream_hccl_task]

    def get_aicpu_task_flip(self: any) -> list:
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_AICPU_TASK_FLIP):
            return []
        sql = "select stream_id, timestamp, task_id, flip_num from {}".format(DBNameConstant.TABLE_AICPU_TASK_FLIP)
        aicpu_task_flip = self.get_sql_data(sql)
        return [self.AICPU_TASK_FLIP(*data) for data in aicpu_task_flip]


@dataclass
class KfcTurnData:
    op_name: str
    stream_id: int
    task_id: int
    start_time: str
    duration: float
