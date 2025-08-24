#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import json
import logging
import os
from collections import defaultdict

from common_func.common import print_msg
from common_func.common import warn
from common_func.data_check_manager import DataCheckManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.number_constant import NumberConstant
from common_func.ms_constant.str_constant import CommunicationMatrixInfo
from common_func.ms_constant.str_constant import StrConstant
from common_func.msprof_common import get_path_dir
from common_func.msprof_common import prepare_for_analyze
from common_func.msprof_common import prepare_log
from common_func.msprof_common import get_valid_sub_path
from common_func.msprof_exception import ProfException
from common_func.msvp_common import create_json_for_dict
from common_func.path_manager import PathManager
from framework.load_info_manager import LoadInfoManager
from msmodel.cluster_info.communication_model import CommunicationModel
from msparser.cluster.communication_matrix_parser import CommunicationMatrixParser
from msmodel.cluster_info.communication_analyzer_model import CommunicationAnalyzerModel


class CommunicationMatrixAnalyzer:
    """
    single NPU communication data analyzer
    """
    FILE_NAME = os.path.basename(__file__)
    HOST_PATH = 'host'
    TRANSPORT_TYPE = {
        0: 'HCCS',
        1: 'PCIE',
        2: 'RDMA',
        3: 'LOCAL',
        4: 'SIO'
    }

    def __init__(self: any, collection_path: any, export_type: any) -> None:
        self.collection_path = collection_path
        self.hccl_op_data = defaultdict(list)
        self.export_type = export_type

    def process(self):
        """Analyzing Communication Data"""
        self._process_sub_dirs()

    def _process_output(self, output_data: list) -> dict:
        """Delete unnecessary fields in dict"""
        proc_dict = {}
        for data in output_data:
            proc_dict[data.get(StrConstant.OP_NAME)] = {}
            link_info = data.get(StrConstant.LINK_INFO)
            for info in link_info:
                src_rank = info.get(CommunicationMatrixInfo.SRC_RANK)
                dst_rank = info.get(CommunicationMatrixInfo.DST_RANK)
                link_key = f"{src_rank}-{dst_rank}"
                proc_dict[data.get(StrConstant.OP_NAME)][link_key] = {
                    CommunicationMatrixInfo.TRANSPORT_TYPE:
                        self.TRANSPORT_TYPE.get(info.get(CommunicationMatrixInfo.TRANSPORT_TYPE)),
                    CommunicationMatrixInfo.TRANSIT_SIZE_MB: info.get(CommunicationMatrixInfo.TRANSIT_SIZE_MB),
                    CommunicationMatrixInfo.TRANSIT_TIME_MS: info.get(CommunicationMatrixInfo.TRANSIT_TIME_MS),
                    CommunicationMatrixInfo.BANDWIDTH_GB_S: info.get(CommunicationMatrixInfo.BANDWIDTH_GB_S)
                }
        return proc_dict

    def _get_hccl_data_from_db(self: any, rank_path: str) -> None:
        """
        get op events of all rank by iteration start and end time
        """
        with CommunicationModel(rank_path) as _model:
            conditions = {
                'iter_start': NumberConstant.DEFAULT_START_TIME,
                'iter_end': NumberConstant.DEFAULT_END_TIME
            }
            events_all = _model.get_all_events_from_db(conditions)
            if not events_all:
                print_msg(f"Fail to get hccl events, "
                          f"please check hccl_single_device.db from {rank_path}")
                raise ProfException(ProfException.PROF_INVALID_DATA_ERROR)
            for event in events_all:
                # get group_name from hccl_single_device.db
                op_name = event.op_name + "@" + event.group_name
                self.hccl_op_data[op_name].append(event)

    def _generate_output(self, rank_path: str) -> None:
        self._get_hccl_data_from_db(rank_path)
        if not self.hccl_op_data:
            message = f"fail to get hccl data"
            logging.error("Can't get hccl events!")
            raise ProfException(ProfException.PROF_INVALID_DATA_ERROR, message)

        communication_matrix_parser = CommunicationMatrixParser(self.hccl_op_data)
        op_info = communication_matrix_parser.run()

        if self.export_type == 'text':
            self._dump_dict_to_json(self._process_output(op_info))
        elif self.export_type == 'db':
            self._dump_dict_to_db(self._process_output(op_info))

    def _dump_dict_to_db(self, output_result: dict) -> object:
        output_file_path = PathManager.get_analyze_dir(self.collection_path)
        with CommunicationAnalyzerModel(output_file_path, [DBNameConstant.TABLE_COMM_ANALYZER_MATRIX]) as _model:
            _model.flush_communication_data_to_db(output_result)

    def _dump_dict_to_json(self, output_result: dict):
        output_file_name = "communication_matrix.json"
        output_file_path = PathManager.get_analyze_result_path(self.collection_path, output_file_name)
        result = create_json_for_dict(output_file_path, output_result)
        result_json = json.loads(result)
        if result_json["status"] == NumberConstant.SUCCESS:
            print_msg(result)
        else:
            print_msg(json.dumps(
                {'status': NumberConstant.ERROR,
                 'info': f'communication matrix data generation failed, '
                         f'maybe you can check the directory({self.collection_path}) permissions.',
                 'data': ''}))

    def _process_sub_dirs(self: any, sub_path: str = '', is_cluster: bool = False) -> None:
        collect_path = self.collection_path
        if sub_path:
            collect_path = os.path.join(self.collection_path, sub_path)
        sub_dirs = sorted(get_path_dir(collect_path), reverse=True)
        for sub_dir in sub_dirs:  # result_dir
            if sub_dir == StrConstant.TIMELINE_PATH or sub_dir == self.HOST_PATH:
                continue

            sub_path = get_valid_sub_path(collect_path, sub_dir, False)
            if DataCheckManager.contain_info_json_data(sub_path):
                LoadInfoManager.load_info(sub_path)
                self._communication_matrix_analyze(sub_path)
            elif sub_path and is_cluster:
                warn(self.FILE_NAME, 'Invalid parsing dir("%s"), -dir must be profiling data dir '
                                     'such as PROF_XXX_XXX_XXX' % collect_path)
            else:
                self._process_sub_dirs(sub_dir, is_cluster=True)

    def _communication_matrix_analyze(self, sub_path: str) -> None:
        """ communication matrix analyzer"""
        if not os.path.exists(PathManager.get_db_path(sub_path, DBNameConstant.DB_HCCL_SINGLE_DEVICE)):
            prepare_log(PathManager.get_analyze_dir(os.path.join(sub_path, '..')))
            logging.warning('There is no hccl data to analyze communication matrix. '
                            'Please export first or check whether single device.')
            return
        prepare_for_analyze(os.path.join(sub_path, '..'))
        self._generate_output(sub_path)
