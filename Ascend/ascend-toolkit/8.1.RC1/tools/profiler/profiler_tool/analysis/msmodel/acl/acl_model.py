#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.str_constant import StrConstant
from msmodel.interface.ianalysis_model import IAnalysisModel
from msmodel.interface.view_model import ViewModel
from profiling_bean.db_dto.acl_dto import AclDto


class AclModel(ViewModel, IAnalysisModel):
    """
    Model of viewer for acl data
    """

    def __init__(self, params: dict) -> None:
        self._result_dir = params.get(StrConstant.PARAM_RESULT_DIR)
        self._iter_range = params.get(StrConstant.PARAM_ITER_ID)
        super().__init__(self._result_dir, DBNameConstant.DB_ACL_MODULE,
                         [DBNameConstant.TABLE_ACL_DATA])

    def get_timeline_data(self: any) -> list:
        sql = "select api_name, start_time, (end_time-start_time) " \
              "as output_duration, process_id, thread_id, api_type " \
              "from {0} order by start_time".format(DBNameConstant.TABLE_ACL_DATA)
        return DBManager.fetch_all_data(self.cur, sql)

    def get_summary_data(self: any) -> list:
        sql = "select api_name, api_type, start_time, (end_time-start_time) " \
              "as output_duration, process_id, thread_id " \
              "from {0} order by start_time asc".format(DBNameConstant.TABLE_ACL_DATA)
        return DBManager.fetch_all_data(self.cur, sql)

    def get_acl_op_execute_data(self):
        sql = f"select start_time, end_time, thread_id from {DBNameConstant.TABLE_ACL_DATA} where " \
              f"api_name='aclopCompileAndExecute' or api_name='aclopCompileAndExecuteV2' order by start_time"
        return DBManager.fetch_all_data(self.cur, sql, dto_class=AclDto)

    def get_acl_op_compile_data(self):
        sql = f"select start_time, end_time, thread_id from {DBNameConstant.TABLE_ACL_DATA} where " \
              f"api_name='OpCompile' order by start_time"
        return DBManager.fetch_all_data(self.cur, sql, dto_class=AclDto)
