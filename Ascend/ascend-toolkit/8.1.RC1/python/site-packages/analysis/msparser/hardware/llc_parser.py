#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2018-2019. All rights reserved.

import logging
import os
import struct

from common_func.constant import Constant
from common_func.db_name_constant import DBNameConstant
from common_func.file_manager import FileManager
from common_func.file_manager import FileOpen
from common_func.ms_multi_process import MsMultiProcess
from common_func.msvp_common import is_valid_original_data
from common_func.path_manager import PathManager
from common_func.platform.chip_manager import ChipManager
from framework.offset_calculator import OffsetCalculator
from msmodel.hardware.llc_model import LlcModel
from msparser.data_struct_size_constant import StructFmt
from profiling_bean.prof_enum.data_tag import DataTag


class NonMiniLLCParser(MsMultiProcess):
    """
    parsing LLC data class
    """

    def __init__(self: any, file_list: dict, sample_config: dict) -> None:
        super().__init__(sample_config)
        self.sample_config = sample_config
        self.device_id = self.sample_config.get("device_id", "0")
        self._file_list = file_list.get(DataTag.LLC, [])
        self.project_path = self.sample_config.get("result_dir", "")
        self._model = LlcModel(self.project_path, DBNameConstant.DB_LLC,
                               [DBNameConstant.TABLE_LLC_ORIGIN, DBNameConstant.TABLE_LLC_EVENTS,
                                DBNameConstant.TABLE_LLC_METRICS])
        self.calculate = OffsetCalculator(self._file_list, StructFmt.LLC_FMT_SIZE, self.project_path)
        self.origin_data = []
        self._file_list.sort(key=lambda x: int(x.split("_")[-1]))

    def read_binary_data(self: any, file_name: str) -> None:
        """
        parsing llc data and insert into llc.db
        :param file_name:
        :return: None
        """
        _file_path = PathManager.get_data_file_path(self.project_path, file_name)
        _file_size = os.path.getsize(_file_path)
        try:
            with FileOpen(_file_path, "rb") as llc_file:
                self._read_binary_helper(llc_file.file_reader, _file_size)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError) as err:
            logging.error("%s: %s", file_name, err)
        finally:
            pass

    def start_parsing_data_file(self: any) -> None:
        """
        parsing data file
        """
        try:
            for file_name in self._file_list:
                self._original_data_handler(file_name)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError) as err:
            logging.error(str(err), exc_info=Constant.TRACE_BACK_SWITCH)
        finally:
            pass

    def save(self: any) -> None:
        """
        save llc data to db
        :return: None
        """
        if self.origin_data and self._model:
            self._model.init()
            self._model.create_table()
            self._model.create_events_trigger(self.sample_config.get("llc_profiling", ""))
            self._model.flush(self.origin_data)
            self._model.insert_metrics_data()
            self._model.finalize()

    def ms_run(self: any) -> None:
        """
        Entry for LLC binaries parse.
        """
        try:
            if self._file_list:
                self.start_parsing_data_file()
                self.save()
        except (OSError, SystemError, ValueError, TypeError, RuntimeError) as llc_err:
            logging.error(str(llc_err), exc_info=Constant.TRACE_BACK_SWITCH)

    def _read_binary_helper(self: any, llc_file: any, _file_size: int) -> None:
        llc_data = self.calculate.pre_process(llc_file, _file_size)
        for _index in range(_file_size // StructFmt.LLC_FMT_SIZE):
            one_slice = llc_data[_index * StructFmt.LLC_FMT_SIZE:(_index + 1) * StructFmt.LLC_FMT_SIZE]

            if one_slice:
                timestamp, count, event_id, l3t_id = struct.unpack(StructFmt.LLC_FMT, one_slice)
                self.origin_data.append(
                    (self.device_id, timestamp, count, event_id, l3t_id))
            else:
                break

    def _original_data_handler(self: any, file_name: str) -> None:
        if is_valid_original_data(file_name, self.project_path):
            logging.info(
                "start parsing llc data file: %s", file_name)
            self.read_binary_data(file_name)
            if not ChipManager().is_chip_v3():
                FileManager.add_complete_file(self.project_path, file_name)
            logging.info("Create LLC DB finished!")
