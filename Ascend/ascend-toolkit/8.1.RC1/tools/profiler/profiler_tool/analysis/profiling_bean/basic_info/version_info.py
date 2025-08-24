#!/usr/bin/python3
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from common_func.info_conf_reader import InfoConfReader
from profiling_bean.basic_info.base_info import BaseInfo


class VersionInfo(BaseInfo):
    def __init__(self: any) -> None:
        super(VersionInfo, self).__init__()
        self.collection_version = ""
        self.analysis_version = ""
        self.drv_version = 0

    def run(self, _):
        self.merge_data()

    def merge_data(self: any) -> any:
        self.collection_version = InfoConfReader().get_collection_version()
        self.analysis_version = InfoConfReader().ANALYSIS_VERSION
        self.drv_version = InfoConfReader().get_drv_version()
