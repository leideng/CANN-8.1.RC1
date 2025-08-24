#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import os
import shutil
import tarfile
from common import FileOperate as f
from params import ParamDict


def compress_output_dir_tar():
    """
    Compress the output directory using tar.
    """

    output_dir = ParamDict().asys_output_timestamp_dir
    if not (output_dir and f.check_dir(output_dir)):
        return
    with tarfile.open(os.path.join(os.path.dirname(output_dir), os.path.basename(output_dir) + ".tar.gz"), "w:gz") \
            as tar:
        tar.add(output_dir, arcname=os.path.basename(output_dir))

    # remove output dir
    shutil.rmtree(output_dir)
