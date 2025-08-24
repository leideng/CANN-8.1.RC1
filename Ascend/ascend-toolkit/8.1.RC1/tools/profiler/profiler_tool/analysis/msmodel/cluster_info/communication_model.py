#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import logging

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.number_constant import NumberConstant
from common_func.msprof_exception import ProfException
from msmodel.interface.view_model import ViewModel
from mscalculate.hccl.hccl_task import HcclTask


class CommunicationModel(ViewModel):
    """
    get hccl operators data from db
    """

    def __init__(self, collection_path):
        super().__init__(collection_path, DBNameConstant.DB_HCCL_SINGLE_DEVICE,
                         [DBNameConstant.TABLE_HCCL_TASK_SINGLE_DEVICE, DBNameConstant.TABLE_KFC_TASK])

    def get_all_events_from_db(self: any, conditions: dict, top_hccl_ops: tuple = None) -> list:
        """
        get hccl op names
        :return:
        """
        if top_hccl_ops is not None:
            if len(top_hccl_ops) < 2:
                condition = "op_name='{}'".format(top_hccl_ops[0])
            else:
                condition = "op_name IN {}".format(top_hccl_ops)
            sql = "select * from {} where timestamp < ? and timestamp >= ? " \
                  "and {condititon}".format(DBNameConstant.TABLE_HCCL_TASK_SINGLE_DEVICE, condititon=condition)
        else:
            sql = "select * from {} where timestamp < ? and timestamp >= ?"
        data = []
        if DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_HCCL_TASK_SINGLE_DEVICE):
            data = DBManager.fetch_all_data(self.cur, sql.format(DBNameConstant.TABLE_HCCL_TASK_SINGLE_DEVICE),
                                            (conditions.get('iter_end', 0) * NumberConstant.NS_TO_US,
                                             conditions.get('iter_start', float('inf')) * NumberConstant.NS_TO_US),
                                            dto_class=HcclTask)
        if DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_KFC_TASK):
            data += DBManager.fetch_all_data(self.cur, sql.format(DBNameConstant.TABLE_KFC_TASK),
                                             (conditions.get('iter_end', 0) * NumberConstant.NS_TO_US,
                                              conditions.get('iter_start', float('inf')) * NumberConstant.NS_TO_US),
                                             dto_class=HcclTask)
        if not data:
            logging.error("Fail to connect %s, hccl parser is interrupted", DBNameConstant.DB_HCCL_SINGLE_DEVICE)
            raise ProfException(ProfException.PROF_INVALID_CONNECT_ERROR)
        return data

    def get_all_communication_data(self: any) -> list:
        data = []
        if DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_KFC_TASK):
            sql = "select * from {}".format(DBNameConstant.TABLE_HCCL_TASK_SINGLE_DEVICE)
            data = DBManager.fetch_all_data(self.cur, sql, dto_class=HcclTask)
        if DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_KFC_TASK):
            sql = "select * from {}".format(DBNameConstant.TABLE_KFC_INFO)
            data += DBManager.fetch_all_data(self.cur, sql, dto_class=HcclTask)
        return data
