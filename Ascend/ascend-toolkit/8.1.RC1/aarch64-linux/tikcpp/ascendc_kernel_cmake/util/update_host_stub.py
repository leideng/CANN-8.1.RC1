#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Purpose: support to repack standardized named packages
# Copyright Huawei Technologies Co., Ltd. 2022-2030. All rights reserved.
# ----------------------------------------------------------------------------

import os
import math
import re
import stat
import argparse
from io import FileIO
from typing import Iterator, List, Optional


def ceil_4(file_len: int) -> int:
    """Multiples of 4 are rounded up."""
    multiple = 4
    return int(math.ceil(file_len / multiple)) * multiple


def get_ascendc_kernel(soc_version: str, target_name: str) -> str:
    return f'__ascend_kernel_{soc_version}_{target_name}'


def get_ascendc_section(soc_version: str, target_name: str) -> str:
    return rf'.ascend.kernel.{soc_version}.{target_name}'


def update_source_section(origin_content: str,
                        soc_version: str,
                        target_name: str) -> str:

    update_content = origin_content

    ascendc_kernel = get_ascendc_kernel(soc_version, target_name)
    replaced_ascend_kernel = '__replaced_ascend_kernel'
    update_content = re.sub(replaced_ascend_kernel, ascendc_kernel, update_content)

    ascendc_section = get_ascendc_section(soc_version, target_name)
    replaced_ascend_section = '__replaced_ascend_section'
    update_content = re.sub(replaced_ascend_section, ascendc_section, update_content)

    replaced_ascend_kernel_soc_version = "__replaced_ascend_compile_soc_version"
    update_content = re.sub(replaced_ascend_kernel_soc_version, soc_version, update_content)

    return update_content


def update_source_content(obj_path: str, origin_content: str) -> str:
    """Update source content."""
    update_flags = {'aic': 'device_aic.o',
                    'aiv': 'device_aiv.o',
                    'mix': 'device.o'}

    update_content = origin_content

    for op_type, obj in update_flags.items():
        flag_file = os.path.join(obj_path, f'{op_type}_build.flag')
        obj_file = os.path.join(obj_path, obj)
        if os.path.exists(flag_file):
            obj_size = os.stat(obj_file).st_size
            file_size = ceil_4(obj_size)

            _file_len_str = f'__replaced_{op_type}_file_len'
            _file_str = f'__replaced_{op_type}_len'
            update_content = re.sub(_file_len_str, str(obj_size), update_content)
            update_content = re.sub(_file_str, str(file_size), update_content)

    return update_content


def main():
    """Main process."""
    parser = argparse.ArgumentParser()
    parser.add_argument('code_path')
    parser.add_argument('obj_path')
    parser.add_argument('soc_version')
    parser.add_argument('target_name')
    args = parser.parse_args()

    source_file = os.path.join(args.code_path, 'host_stub.cpp')
    try:
        with open(source_file, encoding='utf-8') as file:
            content = file.read()
    except Exception as err:
        print("[ERROR]: read file failed, filename is {}".format(source_file))
        raise err

    update_section = update_source_section(content, args.soc_version, args.target_name)
    
    update_content = update_source_content(args.obj_path, update_section)
    try:
        with open(source_file, 'w', encoding='utf-8') as file:
            os.chmod(source_file, stat.S_IRUSR + stat.S_IWUSR)
            file.write(update_content)
    except Exception as err:
        print("[ERROR]: write file failed, filename is {}".format(source_file))
        raise err

if __name__ == '__main__':
    main()













