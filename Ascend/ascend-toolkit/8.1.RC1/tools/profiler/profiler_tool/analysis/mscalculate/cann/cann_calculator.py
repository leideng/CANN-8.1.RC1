#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging
from typing import List

from common_func.constant import Constant
from common_func.ms_constant.str_constant import StrConstant
from common_func.ms_multi_process import MsMultiProcess
from common_func.profiling_scene import ProfilingScene
from mscalculate.cann.cann_analysis_chain import CANNAnalysisChain
from mscalculate.cann.cann_analysis_gear import HCCLGear, ACLGear
from mscalculate.cann.cann_analysis_gear import ModelGear
from mscalculate.cann.cann_analysis_gear import NodeGear
from mscalculate.cann.cann_analysis_gear import RootGear
from mscalculate.cann.cann_analysis_gear import TaskGear
from mscalculate.cann.cann_event_generator import CANNEventGenerator
from mscalculate.interface.icalculator import ICalculator
from msinterface.msprof_c_interface import dump_cann_trace


class CANNCalculator(ICalculator, MsMultiProcess):
    """
    calculator for total cann
    """

    def __init__(self: any, file_list: dict, sample_config: dict) -> None:
        super().__init__(sample_config)
        self._project_path = sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)
        self.event_generator = CANNEventGenerator(self._project_path)
        self.analysis_chains: List[CANNAnalysisChain] = []
        self.gears = dict()

    def calculate(self: any) -> None:
        """
        run the data calculators
        """
        # for future multi-process
        self.gears = {
            Constant.ROOT_LEVEL: RootGear(self._project_path),
            Constant.ACL_LEVEL: ACLGear(self._project_path),
            Constant.MODEL_LEVEL: ModelGear(self._project_path),
            Constant.NODE_LEVEL: NodeGear(self._project_path),
            Constant.TASK_LEVEL: TaskGear(self._project_path),
            Constant.HCCL_LEVEL: HCCLGear(self._project_path),
        }
        thread_dbs = self.event_generator.run()
        for db in thread_dbs:
            for gear in self.gears.values():
                gear.set_db(db)
            chain = CANNAnalysisChain(db.thread_id, db, self.gears)
            self.analysis_chains.append(chain)
            chain.start()

    def save(self: any) -> None:
        """
        save data to database
        """
        for gear in self.gears.values():
            gear.flush_data()

    def ms_run(self: any) -> None:
        """
        main
        :return: None
        """
        logging.info("start to analysis cann software callstack")
        if ProfilingScene().is_cpp_parse_enable():
            dump_cann_trace(self._project_path)
            return
        else:
            logging.warning("Data will not be parsed by msprof_analysis.so!")
        self.calculate()
        self.save()
