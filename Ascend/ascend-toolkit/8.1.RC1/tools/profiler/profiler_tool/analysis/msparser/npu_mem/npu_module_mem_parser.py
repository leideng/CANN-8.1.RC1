#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging
import os
import re

from common_func.constant import Constant
from common_func.db_name_constant import DBNameConstant
from common_func.file_manager import FileManager
from common_func.file_manager import FileOpen
from common_func.ms_multi_process import MsMultiProcess
from common_func.path_manager import PathManager
from common_func.msvp_common import is_valid_original_data
from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant
from framework.offset_calculator import OffsetCalculator
from msparser.data_struct_size_constant import StructFmt
from msparser.interface.data_parser import DataParser
from msmodel.npu_mem.npu_ai_stack_mem_model import NpuAiStackMemModel
from profiling_bean.prof_enum.data_tag import DataTag
from msparser.npu_mem.npu_module_mem_bean import NpuModuleMemDataBean


class NpuModuleMemParser(DataParser, MsMultiProcess):
    """
    parsing npu module mem data
    """

    def __init__(self: any, file_list: dict, sample_config: dict) -> None:
        super().__init__(sample_config)
        super(DataParser, self).__init__(sample_config)
        self._file_list = file_list
        self._npu_module_mem_data = []
        self._model = NpuAiStackMemModel(self._project_path,
                                                DBNameConstant.DB_NPU_MODULE_MEM,
                                                [DBNameConstant.TABLE_NPU_MODULE_MEM])

    def parse(self: any) -> None:
        """
        start parsing the data
        """
        num_module_mem_files = self._file_list.get(DataTag.NPU_MODULE_MEM, [])
        if not num_module_mem_files:
            return
        for _file in num_module_mem_files:
            if not is_valid_original_data(_file, self._project_path):
                continue
            _file_path = PathManager.get_data_file_path(self._project_path, _file)
            _file_size = os.path.getsize(_file_path)
            if not _file_size:
                logging.warning(
                    "The size of file: %s is zero. Check whether the file size is correct.", _file)
                continue
            logging.info(
                "start parsing npu module mem data file: %s", _file)
            self._process_npu_module_mem_data(_file_path, _file_size, num_module_mem_files)
            FileManager.add_complete_file(self._project_path, _file)

    def save(self: any) -> bool:
        """
        save npu module mem data
        """
        if self._npu_module_mem_data:
            with self._model as _model:
                _model.flush(DBNameConstant.TABLE_NPU_MODULE_MEM, self._npu_module_mem_data)

    def ms_run(self: any) -> None:
        """
        run function
        """
        if not self._file_list.get(DataTag.NPU_MODULE_MEM, []):
            return
        try:
            self.parse()
        except (OSError, SystemError, ValueError, TypeError, RuntimeError) as err:
            logging.error(str(err), exc_info=Constant.TRACE_BACK_SWITCH)
            return
        self.save()

    def _process_npu_module_mem_data(self: any, file_path: str, file_size: int, file_list: list) -> None:
        offset_calculator = OffsetCalculator(file_list, StructFmt.NPU_MODULE_MEM_SIZE,
                                             self._project_path)
        with FileOpen(file_path, "rb") as _npu_module_mem_file:
            _all_data = offset_calculator.pre_process(_npu_module_mem_file.file_reader, file_size)
            device_type = "NPU" + ':' + self._sample_config.get('device_id')
            for _index in range(file_size // StructFmt.NPU_MODULE_MEM_SIZE):
                data = _all_data[_index * StructFmt.NPU_MODULE_MEM_SIZE:(_index + 1) * StructFmt.NPU_MODULE_MEM_SIZE]
                npu_module_mem_data_bean = NpuModuleMemDataBean.decode(data)
                if npu_module_mem_data_bean is None:
                    continue
                if npu_module_mem_data_bean.total_size > NumberConstant.DEFAULT_NUMBER:
                    # negative reversed
                    logging.warning(
                        "The total_size %d greater than integer maximum of sqlite,"
                        " please confirm wheather the total_size is reversed.",
                        npu_module_mem_data_bean.total_size)
                    self._npu_module_mem_data.append([
                        npu_module_mem_data_bean.module_id,
                        npu_module_mem_data_bean.cpu_cycle_count,
                        -1,
                        device_type
                    ])
                    continue
                self._npu_module_mem_data.append([
                    npu_module_mem_data_bean.module_id,
                    npu_module_mem_data_bean.cpu_cycle_count,
                    npu_module_mem_data_bean.total_size,
                    device_type
                ])
