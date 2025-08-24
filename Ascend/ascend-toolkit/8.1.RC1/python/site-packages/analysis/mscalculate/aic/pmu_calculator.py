#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import os

from common_func.constant import Constant
from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.file_manager import FileManager
from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.str_constant import StrConstant
from common_func.msprof_iteration import MsprofIteration
from common_func.path_manager import PathManager
from common_func.profiling_scene import ProfilingScene
from mscalculate.interface.icalculator import ICalculator
from profiling_bean.db_dto.ge_task_dto import GeTaskDto


class PmuCalculator(ICalculator):
    """
    class used to parse aicore data
    """
    AICORE_LOG_SIZE = 128
    STREAM_TASK_KEY_FMT = "{0}-{1}"

    _project_path = None
    _sample_json = None
    _iter_range = None
    _block_dims = None
    _freq = None
    _core_num_dict = None
    sample_config = None

    def calculate(self: any) -> None:
        """
        calculate the ai core
        :return: None
        """

    def save(self: any) -> None:
        """
        save ai core data
        :return: None
        """

    def init_params(self):
        self._project_path = self.sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)
        self._iter_range = self.sample_config.get(StrConstant.PARAM_ITER_ID)
        self._block_dims = {'block_dim': {}, 'mix_block_dim': {}, 'block_dim_group': {}}
        self._core_num_dict = {'aic': 0, 'aiv': 0}
        self._freq = 0
        self.get_block_dim_from_ge()
        self.get_config_params()

    def get_block_dim_from_ge(self: any) -> None:
        """
        get ge data from ge info in the format of [task_id, stream_id, blockid]
        :return: {"task_id-stream_id":blockdim}
        """
        db_path = PathManager.get_db_path(self._project_path, DBNameConstant.DB_GE_INFO)
        if not DBManager.check_tables_in_db(db_path, DBNameConstant.TABLE_GE_TASK):
            return
        ge_conn, ge_curs = DBManager.check_connect_db_path(db_path)
        if ge_conn and ge_curs:
            ge_data = self.__get_block_dim_data(ge_curs)
            if not ge_data:
                DBManager.destroy_db_connect(ge_conn, ge_curs)
                return
            self._format_ge_data(ge_data)
        DBManager.destroy_db_connect(ge_conn, ge_curs)

    def get_config_params(self: any) -> None:
        """
        get core num and ai core freq from config file
        """
        for file_name in os.listdir(self._project_path):
            if FileManager.is_info_json_file(file_name):
                self._core_num_dict['aic'] = InfoConfReader().get_data_under_device('ai_core_num')
                self._core_num_dict['aiv'] = InfoConfReader().get_data_under_device('aiv_num')
                self._freq = InfoConfReader().get_freq(StrConstant.AIC)

    def _get_current_block(self: any, block_type: str, ai_core_data: any) -> int:
        """
        get the current block dim when stream id and task id occurs again
        :param ai_core_data: ai core pmu data
        :return: block dim
        """
        block = self._block_dims.get(block_type, {}).get(
            self.STREAM_TASK_KEY_FMT.format(ai_core_data.task_id, ai_core_data.stream_id))
        if not block:
            return 0
        return block.pop(0) if len(block) > 1 else block[0]

    def _format_ge_data(self: any, ge_data: list) -> None:
        for data in ge_data:
            if data.task_type not in (Constant.TASK_TYPE_AI_CORE, Constant.TASK_TYPE_AIV, Constant.TASK_TYPE_HCCL,
                                      Constant.TASK_TYPE_MIX_AIC, Constant.TASK_TYPE_MIX_AIV):
                continue
            _key = self.STREAM_TASK_KEY_FMT.format(data.task_id, data.stream_id)
            self._block_dims.get('block_dim', {}).setdefault(_key, []).append(int(data.block_dim))
            if data.task_type in (Constant.TASK_TYPE_MIX_AIV, Constant.TASK_TYPE_MIX_AIC):
                self._block_dims.get('mix_block_dim', {}).setdefault(_key, []).append(int(data.mix_block_dim))

    def __get_block_dim_data(self: any, ge_curs: any) -> list:
        device_id = InfoConfReader().get_device_id()
        if ProfilingScene().is_all_export() or ProfilingScene().is_step_export():
            sql = "select task_id, stream_id, context_id, task_type, block_dim, mix_block_dim from {0} " \
                  "where device_id={1} " \
                  "order by timestamp".format(DBNameConstant.TABLE_GE_TASK, device_id)
            return DBManager.fetch_all_data(ge_curs, sql, dto_class=GeTaskDto)
        ge_data = []
        iter_list = MsprofIteration(self._project_path).get_index_id_list_with_index_and_model(self._iter_range)
        sql = "select task_id, stream_id, context_id, task_type, block_dim, mix_block_dim from {0} " \
              "where model_id=? and (index_id=0 or index_id=?) and device_id={1} " \
              " order by timestamp".format(DBNameConstant.TABLE_GE_TASK, device_id)
        for iter_id, model_id in iter_list:
            ge_data.extend(DBManager.fetch_all_data(ge_curs, sql, (model_id, iter_id), dto_class=GeTaskDto))
        return ge_data
