#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
import os

from common_func.constant import Constant
from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.path_manager import PathManager
from common_func.profiling_scene import ProfilingScene
from common_func.msprof_object import CustomizedNamedtupleFactory
from common_func.ms_constant.str_constant import StrConstant
from msmodel.interface.ianalysis_model import IAnalysisModel
from msmodel.interface.view_model import ViewModel
from msmodel.add_info.mc2_comm_info_model import Mc2CommInfoViewModel
from profiling_bean.db_dto.time_section_dto import TimeSectionDto
from profiling_bean.db_dto.time_section_dto import CommunicationTimeSection


def filter_aiv_communication(data_list: list) -> list:
    compute_data = []
    for data in data_list:
        if data.op_name.endswith(StrConstant.AIV_KERNEL):
            continue
        compute_data.append(data)
    return compute_data


class OpSummaryModel(ViewModel, IAnalysisModel):
    """
    class used to operate summary db
    """

    COMPUTED_TASK_TYPE = (
        Constant.TASK_TYPE_AI_CORE, Constant.TASK_TYPE_AI_CPU,
        Constant.TASK_TYPE_AIV, Constant.TASK_TYPE_MIX_AIV,
        Constant.TASK_TYPE_MIX_AIC
    )

    COMMUNICATION_TIME_SECTION_TYPE = CustomizedNamedtupleFactory.generate_named_tuple_from_dto(
        CommunicationTimeSection, [])

    def __init__(self: any, sample_config: dict) -> None:
        super().__init__(sample_config.get("result_dir"), DBNameConstant.DB_AICORE_OP_SUMMARY, [])
        self.conn = None
        self.cur = None
        self.result_dir = sample_config.get("result_dir")
        self.iter_id = sample_config.get("iter_id")
        self.model_id = sample_config.get("model_id")

    def get_summary_data(self: any) -> str:
        return self.__module__

    def get_timeline_data(self: any) -> str:
        return self.__module__

    def separate_kfc_stream(self: any, data_list: list) -> list:
        if not os.path.exists(PathManager.get_db_path(self.result_dir, DBNameConstant.DB_MC2_COMM_INFO)):
            return data_list
        with Mc2CommInfoViewModel(self.result_dir, [DBNameConstant.TABLE_MC2_COMM_INFO]) as model:
            comm_info = model.get_kfc_stream(DBNameConstant.TABLE_MC2_COMM_INFO)
        if not comm_info:
            return data_list
        kfc_stream_id = []
        for info in comm_info:
            kfc_stream_id.append(info[0])  # 0-aicpu_kfc_stream_id
        compute_data = []
        for data in data_list:
            if data.stream_id in kfc_stream_id or data.op_name.endswith(StrConstant.AICPU_KERNEL):
                continue
            compute_data.append(data)
        return compute_data

    def get_operator_data_by_task_type(self: any, task_type: tuple = COMPUTED_TASK_TYPE) -> list:
        db_path = PathManager.get_db_path(self.result_dir, DBNameConstant.DB_AICORE_OP_SUMMARY)
        if not DBManager.check_tables_in_db(db_path, DBNameConstant.TABLE_SUMMARY_TASK_TIME,
                                            DBNameConstant.TABLE_SUMMARY_GE):
            return []
        ge_summary_headers = DBManager.get_table_headers(self.cur, DBNameConstant.TABLE_SUMMARY_GE)
        task_time_headers = DBManager.get_table_headers(self.cur, DBNameConstant.TABLE_SUMMARY_TASK_TIME)
        inner_join_condition = ""
        if "model_id" in ge_summary_headers and "model_id" in task_time_headers:
            inner_join_condition += " and a.model_id=b.model_id"
        if "index_id" in ge_summary_headers and "index_id" in task_time_headers:
            if ProfilingScene().is_graph_export():
                inner_join_condition += " and (a.index_id=b.index_id or b.index_id=0)"
            else:
                inner_join_condition += " "
        sql = "SELECT a.stream_id, op_name, b.task_type, start_time, duration_time, " \
              "start_time+duration_time as end_time FROM {0} a INNER JOIN {1} b " \
              "on a.stream_id=b.stream_id and a.task_id=b.task_id and a.batch_id=b.batch_id " \
              "and a.subtask_id=b.context_id {2} " \
              "and a.task_type<>'{unknown}'".format(
                DBNameConstant.TABLE_SUMMARY_TASK_TIME,
                DBNameConstant.TABLE_SUMMARY_GE,
                inner_join_condition,
                unknown=Constant.TASK_TYPE_UNKNOWN)
        return DBManager.fetch_all_data(self.cur, sql, dto_class=TimeSectionDto)

    def get_operator_data_separated_by_kfc_stream(self: any) -> list:
        operator_data = self.get_operator_data_by_task_type()
        return filter_aiv_communication(self.separate_kfc_stream(operator_data))
