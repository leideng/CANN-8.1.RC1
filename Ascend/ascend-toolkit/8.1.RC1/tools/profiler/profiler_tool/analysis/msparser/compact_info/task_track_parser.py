#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging
from typing import List

from common_func.db_name_constant import DBNameConstant
from common_func.hash_dict_constant import HashDictData
from common_func.ms_constant.str_constant import StrConstant
from common_func.ms_multi_process import MsMultiProcess
from common_func.info_conf_reader import InfoConfReader
from common_func.platform.chip_manager import ChipManager
from msmodel.compact_info.task_track_model import TaskTrackModel
from msparser.compact_info.task_track_bean import TaskTrackBean
from msparser.data_struct_size_constant import StructFmt
from msparser.interface.data_parser import DataParser
from profiling_bean.prof_enum.data_tag import DataTag
from mscalculate.flip.flip_calculator import FlipCalculator


class TaskTrackParser(DataParser, MsMultiProcess):
    """
    task track data parser
    """
    TASK_FLIP = "FLIP_TASK"
    TASK_MAINTENANCE = "MAINTENANCE"

    def __init__(self: any, file_list: dict, sample_config: dict) -> None:
        super().__init__(sample_config)
        super(DataParser, self).__init__(sample_config)
        self._file_list = file_list
        self._project_path = sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)
        self._task_track_data = []
        self._task_flip_data = []

    def reformat_data(self: any, bean_data: List[TaskTrackBean]) -> None:
        """
        transform bean to data
        """
        hash_dict_data = HashDictData(self._project_path)
        type_hash_dict = hash_dict_data.get_type_hash_dict()
        for data in bean_data:
            if type_hash_dict.get(data.level, {}).get(data.task_type, data.task_type) == self.TASK_FLIP:
                setattr(data, 'flip_num', data.batch_id)
                self._task_flip_data.append(data)
                continue
            if type_hash_dict.get(data.level, {}).get(data.task_type, data.task_type) == self.TASK_MAINTENANCE:
                continue
            self._task_track_data.append(data)
        if ChipManager().is_chip_all_data_export() and InfoConfReader().is_all_export_version():
            self._task_track_data = FlipCalculator.compute_batch_id(self._task_track_data, self._task_flip_data)
        self._task_track_data = [
            [
                bean.device_id,
                bean.timestamp,
                type_hash_dict.get(bean.level, {}).get(bean.task_type, bean.task_type),  # task type
                bean.stream_id,
                bean.task_id,
                bean.thread_id,
                bean.batch_id,
                type_hash_dict.get(bean.level, {}).get(bean.struct_type, bean.struct_type),  # task track type
                bean.level,
                bean.data_len,
            ] for bean in self._task_track_data
        ]
        self._task_flip_data = [
            [
                bean.stream_id,
                bean.timestamp,
                bean.task_id,
                bean.batch_id,
            ] for bean in self._task_flip_data
        ]

    def save(self: any) -> None:
        """
        save task track data
        """
        if self._task_track_data:
            with TaskTrackModel(self._project_path, [DBNameConstant.TABLE_TASK_TRACK]) as model:
                model.flush(self._task_track_data)
        if self._task_flip_data:
            with TaskTrackModel(self._project_path, [DBNameConstant.TABLE_HOST_TASK_FLIP]) as model:
                model.flush(self._task_flip_data)

    def parse(self: any) -> None:
        """
        parse task track data
        """
        track_files = self._file_list.get(DataTag.TASK_TRACK, [])
        track_files = self.group_aging_file(track_files)
        if not track_files:
            return
        bean_data = []
        for files in track_files.values():
            bean_data += self.parse_bean_data(
                files,
                StructFmt.TASK_TRACK_DATA_SIZE,
                TaskTrackBean,
                format_func=lambda x: x,
                check_func=self.check_magic_num,
            )
        self.reformat_data(bean_data)

    def ms_run(self: any) -> None:
        """
        parse and save task track data
        :return:
        """
        if not self._file_list.get(DataTag.TASK_TRACK, []):
            return
        logging.info("start parsing task track data, files: %s", str(self._file_list.get(DataTag.TASK_TRACK, [])))
        self.parse()
        self.save()
