# -*- coding: UTF-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
_internal_lib.py
"""


# 'pylint: disable=too-few-public-methods
class ObjWithConst:
    """
    set const object
    """
    def __setattr__(self, name, value):
        b_has = hasattr(self, name)
        is_const = "const" in name
        if b_has and is_const:
            raise RuntimeError("{} not allow to change".format(name))
        super(ObjWithConst, self).__setattr__(name, value)


# 'pylint: disable=too-few-public-methods
class VecBufInfo:
    """
    init vector info
    """
    def __init__(self, addr, blk_stride, rpt_stride):
        self.addr = addr
        self.blk_stride = blk_stride
        self.rpt_stride = rpt_stride


# 'pylint: disable=too-few-public-methods,too-many-arguments
class VecLoopInfo:
    """
    define vector loop info
    """
    def __init__(self, num_per_cmd, max_cmd_rpt, loop, repeat, left):
        self.num_per_cmd = num_per_cmd
        self.max_cmd_rpt = max_cmd_rpt
        self.loop = loop
        self.repeat = repeat
        self.left = left
