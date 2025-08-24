#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2018-2019. All rights reserved.

import logging
import multiprocessing
import os
import time
from abc import abstractmethod

from common_func.common import init_log, warn
from common_func.constant import Constant
from common_func.msprof_exception import ProfException
from common_func.profiling_scene import ProfilingScene
from common_func.profiling_scene import ExportMode
from common_func.ms_constant.str_constant import StrConstant
from framework.load_info_manager import LoadInfoManager


class MsMultiProcess(multiprocessing.Process):
    """
    ms multi process base class
    """

    FILE_NAME = os.path.basename(__file__)

    def __init__(self: any, sample_config: dict) -> None:
        super().__init__()
        self.sample_config = sample_config

    def process_init(self: any) -> None:
        """
        init process
        """
        init_log(self.sample_config.get("result_dir"))
        LoadInfoManager.load_info(self.sample_config.get("result_dir"))
        ProfilingScene().set_mode(self.sample_config.get(StrConstant.EXPORT_MODE, ExportMode.ALL_EXPORT))

    @abstractmethod
    def ms_run(self: any) -> None:
        """
        subclass need to implement this method
        """

    def run(self: any) -> None:
        """
        implement run method
        """
        start_time = time.time()
        self.process_init()
        try:
            self.ms_run()
        except ProfException as err:
            if err.message:
                err.callback(self.FILE_NAME, err.message)
            else:
                result_dir = self.sample_config.get("result_dir")
                warn(self.FILE_NAME, 'Analysis data in "%s" failed. Maybe the data is incomplete.' % result_dir)
        except Exception as err:
            logging.error(str(err), exc_info=Constant.TRACE_BACK_SWITCH)
        logging.info("%s process data finished, execute time is %.3f s",
                      self.__class__.__name__, (time.time() - start_time))
