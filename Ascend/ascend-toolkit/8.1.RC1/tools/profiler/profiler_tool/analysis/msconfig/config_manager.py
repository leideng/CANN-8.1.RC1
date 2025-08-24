#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import logging

from msconfig.ai_core_config import AICoreConfig
from msconfig.ai_cpu_config import AICPUConfig
from msconfig.constant_config import ConstantConfig
from msconfig.ctrl_cpu_config import CtrlCPUConfig
from msconfig.data_calculator_config import DataCalculatorConfig
from msconfig.data_parsers_config import DataParsersConfig
from msconfig.filename_introduction_config import FilenameIntroductionConfig
from msconfig.l2_cache_config import L2CacheConfig
from msconfig.meta_config import MetaConfig
from msconfig.msprof_export_data_config import MsProfExportDataConfig
from msconfig.prof_condition_config import ProfConditionConfig
from msconfig.stars_config import StarsConfig
from msconfig.tables_config import TablesConfig
from msconfig.tables_operator_config import TablesOperatorConfig
from msconfig.tables_training_config import TablesTrainingConfig
from msconfig.ts_cpu_config import TsCPUConfig


class ConfigManager:
    PROF_CONDITION = 'ProfConditionConfig'
    DATA_PARSERS = 'DataParsersConfig'
    STARS = 'StarsConfig'
    TABLES = 'TablesConfig'
    TABLES_TRAINING = 'TablesTrainingConfig'
    TABLES_OPERATOR = 'TablesOperatorConfig'
    MSPROF_EXPORT_DATA = 'MsProfExportDataConfig'
    DATA_CALCULATOR = 'DataCalculatorConfig'
    AI_CORE = "AICoreConfig"
    AI_CPU = "AICPUConfig"
    CTRL_CPU = "CtrlCPUConfig"
    TS_CPU = "TsCPUConfig"
    CONSTANT = "ConstantConfig"
    L2_CACHE = "L2CacheConfig"
    FILENAME_INTRODUCTION = "FilenameIntroductionConfig"

    CONFIG_MAP = {
        PROF_CONDITION: ProfConditionConfig,
        DATA_PARSERS: DataParsersConfig,
        STARS: StarsConfig,
        TABLES: TablesConfig,
        TABLES_TRAINING: TablesTrainingConfig,
        TABLES_OPERATOR: TablesOperatorConfig,
        MSPROF_EXPORT_DATA: MsProfExportDataConfig,
        DATA_CALCULATOR: DataCalculatorConfig,
        AI_CORE: AICoreConfig,
        AI_CPU: AICPUConfig,
        CTRL_CPU: CtrlCPUConfig,
        TS_CPU: TsCPUConfig,
        CONSTANT: ConstantConfig,
        L2_CACHE: L2CacheConfig,
        FILENAME_INTRODUCTION: FilenameIntroductionConfig,
    }

    @classmethod
    def get(cls: any, config_name: str):
        config_class = cls.CONFIG_MAP.get(config_name)
        if not config_class:
            config_class = MetaConfig
            logging.error('%s not found', config_name)
        return config_class()
