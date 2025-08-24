#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.

import importlib
import os
import sys

if __name__ == '__main__':
    sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
    MODEL_PATH = "msinterface.msprof_entrance"
    MSPROF_ENTRANCE_CLASS = "MsprofEntrance"
    os.umask(0o027)
    msprof_entrance_module = importlib.import_module(MODEL_PATH)
    if hasattr(msprof_entrance_module, MSPROF_ENTRANCE_CLASS):
        getattr(msprof_entrance_module, MSPROF_ENTRANCE_CLASS)().main()
