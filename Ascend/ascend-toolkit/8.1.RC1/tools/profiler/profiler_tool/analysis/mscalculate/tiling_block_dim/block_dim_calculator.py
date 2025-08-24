#!/usr/bin/python3
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.str_constant import StrConstant
from common_func.ms_multi_process import MsMultiProcess
from common_func.path_manager import PathManager
from mscalculate.flip.flip_calculator import FlipCalculator
from mscalculate.interface.icalculator import ICalculator
from msmodel.ge.ge_info_model import GeInfoViewModel
from msmodel.step_trace.ts_track_model import TsTrackViewModel


class BlockDimCalculator(ICalculator, MsMultiProcess):
    BITS_FOR_BLOCK_DIM = 16
    INVALID_BLOCK_DIM_VALUE = 65535

    def __init__(self: any, file_list: dict, sample_config: dict):
        super().__init__(sample_config)
        self._file_list = file_list
        self._project_path = sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)
        self._ge_model = GeInfoViewModel(self._project_path, [DBNameConstant.TABLE_GE_TASK])
        self._ts_model = TsTrackViewModel(self._project_path, [DBNameConstant.TABLE_BLOCK_DIM])
        self._data = []

    @staticmethod
    def _process_block_dim_data(data):
        return {(datum.stream_id, datum.task_id, datum.batch_id): datum for datum in data}

    def calculate(self: any) -> None:
        with self._ge_model as _ge_model:
            if not _ge_model.check_table():
                return
            ge_task_data = _ge_model.get_ge_info_by_device_id(DBNameConstant.TABLE_GE_TASK,
                                                              InfoConfReader().get_device_id())
        if not ge_task_data:
            return

        with self._ts_model as _ts_model:
            if not _ts_model.check_table():
                return
            tiling_block_dim_data = _ts_model.get_tiling_block_dim_data()
        if not tiling_block_dim_data:
            return

        tiling_block_dim_data = FlipCalculator.set_device_batch_id(tiling_block_dim_data, self._project_path)
        processed_block_dim_data = self._process_block_dim_data(tiling_block_dim_data)
        for ge_data in ge_task_data:
            search_key = (ge_data.stream_id, ge_data.task_id, ge_data.batch_id)
            if search_key in processed_block_dim_data:
                tiling_blcok_dim = processed_block_dim_data.get(search_key).block_dim
                self._data.append(ge_data.replace(block_dim=tiling_blcok_dim & self.INVALID_BLOCK_DIM_VALUE,
                                                  mix_block_dim=(tiling_blcok_dim & self.INVALID_BLOCK_DIM_VALUE) * (
                                                          tiling_blcok_dim >> self.BITS_FOR_BLOCK_DIM)))
            else:
                self._data.append(ge_data)

    def save(self: any) -> None:
        if not self._data:
            return
        with self._ge_model as _ge_model:
            delete_sql = f"delete from {DBNameConstant.TABLE_GE_TASK} " \
                         f"where device_id={InfoConfReader().get_device_id()}"
            DBManager.execute_sql(_ge_model.conn, delete_sql)
            _ge_model.insert_data_to_db(DBNameConstant.TABLE_GE_TASK, self._data)

    def ms_run(self: any) -> None:
        if not os.path.exists(PathManager.get_db_path(self._project_path, DBNameConstant.DB_GE_INFO)):
            return
        self.calculate()
        self.save()
