#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright 2024-2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import argparse
import os.path

import dataflow.utils.log as log
from dataflow.tools.func_ws_creator import FuncWsCreator


def main():
    parser = argparse.ArgumentParser(description="flow func workspace Creator")
    parser.add_argument(
        "-f",
        "--functions",
        required=True,
        type=str,
        help="Input function info, input and output indexes. e.g. Sub:i0:i2:o0,Add:i1:i3:o1:o0",
    )
    parser.add_argument(
        "-c",
        "--clz_name",
        required=False,
        default="",
        type=str,
        help="flow func class name",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        required=False,
        default="",
        type=str,
        help="flow func workspace path",
    )
    args = parser.parse_args()

    print(
        f"args: functions={args.functions}, clz_name={args.clz_name}, workspace={args.workspace}"
    )
    path = os.path.abspath(args.workspace)
    confirmation = input(
        f"will create workspace in path '{path}', please enter Yes(y) to confirm:"
    )
    if confirmation.lower() == "yes" or confirmation.lower() == "y":
        print("create function workspace begin")
        creator = FuncWsCreator(args.functions, args.clz_name, args.workspace)
        creator.generate()
        print("create function workspace finish")
    else:
        print("exit")


if __name__ == "__main__":
    main()
