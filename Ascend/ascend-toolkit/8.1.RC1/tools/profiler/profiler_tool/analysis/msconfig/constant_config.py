#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from msconfig.meta_config import MetaConfig


class ConstantConfig(MetaConfig):
    DATA = {
        'events': [
            ('ai_ctrl_cpu_profiling_events', '0x11,0x8'),
            ('ts_cpu_profiling_events', '0x11,0x8'),
            ('hbm_profiling_events', 'read,write'),
            ('ddr_profiling_events', 'read,write')
        ],
        'GENERAL': [
            ('dangerous_app_list', 'rm,mv,reboot,shutdown,halt')
        ]
    }

    def __init__(self):
        super().__init__()

