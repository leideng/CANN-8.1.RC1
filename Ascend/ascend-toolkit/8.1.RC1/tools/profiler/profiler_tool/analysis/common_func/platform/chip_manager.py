#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import os

from common_func.constant import Constant
from common_func.info_conf_reader import InfoConfReader
from common_func.msprof_exception import ProfException
from common_func.singleton import singleton
from profiling_bean.prof_enum.chip_model import ChipModel, ChipCoreNum


@singleton
class ChipManager:
    """
    class used to get chip info.
    """
    CHIP_RELATION_MAP = {
        Constant.CHIP_V1_1_0: ChipModel.CHIP_V1_1_0,
        Constant.CHIP_V2_1_0: ChipModel.CHIP_V2_1_0,
        Constant.CHIP_V3_1_0: ChipModel.CHIP_V3_1_0,
        Constant.CHIP_V3_2_0: ChipModel.CHIP_V3_2_0,
        Constant.CHIP_V3_3_0: ChipModel.CHIP_V3_3_0,
        Constant.CHIP_V4_1_0: ChipModel.CHIP_V4_1_0,
        Constant.CHIP_V1_1_1: ChipModel.CHIP_V1_1_1,
        Constant.CHIP_V1_1_2: ChipModel.CHIP_V1_1_2,
        Constant.CHIP_V1_1_3: ChipModel.CHIP_V1_1_3
    }
    CHIP_CORE_NUM_MAP = {
        ChipModel.CHIP_V4_1_0: ChipCoreNum.CHIP_V4_1_0,
        ChipModel.CHIP_V1_1_1: ChipCoreNum.CHIP_V1_1_1,
        ChipModel.CHIP_V1_1_2: ChipCoreNum.CHIP_V1_1_2,
        ChipModel.CHIP_V1_1_3: ChipCoreNum.CHIP_V1_1_3
    }

    ALL_DATA_EXPORT_CHIP_BLACKLIST = [
        ChipModel.CHIP_V1_1_0,
        ChipModel.CHIP_V3_1_0,
        ChipModel.CHIP_V1_1_3,
    ]

    FILE_NAME = os.path.basename(__file__)

    def __init__(self: any) -> None:
        self.chip_id = ChipModel.CHIP_V1_1_0

    @staticmethod
    def is_ffts_type() -> bool:
        """
        check if ffts type
        :return: True or False
        """
        return True

    @staticmethod
    def is_ffts_plus_type() -> bool:
        """
        check if ffts plus type
        :return:
        """
        return True

    @classmethod
    def get_chip_id(cls: any) -> any:
        """
        get chip id
        :return: chip id
        """
        return cls.CHIP_RELATION_MAP.get(InfoConfReader().get_root_data(Constant.PLATFORM_VERSION))

    def load_chip_info(self: any) -> None:
        """
        load chip info
        :return:
        """
        if InfoConfReader().get_root_data(Constant.PLATFORM_VERSION) not in self.CHIP_RELATION_MAP.keys():
            message = "Can't get platform version or platform version isn't identified from info.json, " \
                      "please check the file."
            raise ProfException(ProfException.PROF_SYSTEM_EXIT, message)
        self.chip_id = self.CHIP_RELATION_MAP.get(InfoConfReader().get_root_data(Constant.PLATFORM_VERSION))

    def is_chip_v1(self: any) -> bool:
        """
        check the scene of mini
        :return: True or False
        """
        return self.chip_id == ChipModel.CHIP_V1_1_0

    def is_chip_v2(self: any) -> bool:
        """
        check the scene of cloud
        :return: True or False
        """
        return self.chip_id == ChipModel.CHIP_V2_1_0

    def is_chip_v3(self: any) -> bool:
        """
        check the scene of dc or mdc or lhisi
        :return: True or False
        """
        return self.chip_id in (ChipModel.CHIP_V3_1_0, ChipModel.CHIP_V3_2_0, ChipModel.CHIP_V3_3_0)

    def is_chip_v3_1(self: any) -> bool:
        """
        check the scene of mdc
        :return: True or False
        """
        return self.chip_id == ChipModel.CHIP_V3_1_0

    def is_chip_v3_2(self: any) -> bool:
        """
        check the scene of lhisi
        :return: True or False
        """
        return self.chip_id == ChipModel.CHIP_V3_2_0

    def is_chip_v3_3(self: any) -> bool:
        """
        check the scene of dc
        :return: True or False
        """
        return self.chip_id == ChipModel.CHIP_V3_3_0

    def is_chip_v4(self: any) -> bool:
        """
        check the scene of chip.v4.1.0
        :return: True or False
        """
        return self.chip_id == ChipModel.CHIP_V4_1_0

    def is_chip_v1_1(self: any) -> bool:
        """
        check the scene of chip.v1.1.1 or chip.v1.1.2 or chip.v1.1.3
        :return: True or False
        """
        return self.chip_id in (ChipModel.CHIP_V1_1_1, ChipModel.CHIP_V1_1_2, ChipModel.CHIP_V1_1_3)

    def is_chip_v1_1_1(self: any) -> bool:
        """
        check the scene of chip.v1.1.1
        :return: True or False
        """
        return self.chip_id == ChipModel.CHIP_V1_1_1

    def is_stars_chip(self) -> bool:
        """
        check the scene of chip.v4.1.0 or chip.v1.1.x
        :return: True or False
        """
        return self.is_chip_v1_1() or self.is_chip_v4()

    def is_chip_all_data_export(self: any) -> bool:
        """
        check the all data export scene of chip
        :return: True or False
        """
        return self.chip_id not in self.ALL_DATA_EXPORT_CHIP_BLACKLIST

    def get_max_core_id(self) -> ChipCoreNum:
        if self.chip_id not in self.CHIP_CORE_NUM_MAP:
            message = "Can't get ai core num or platform version isn't identified from info.json, " \
                      "please check the file."
            raise ProfException(ProfException.PROF_SYSTEM_EXIT, message)
        return self.CHIP_CORE_NUM_MAP.get(self.chip_id).value
