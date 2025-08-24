#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
elewise compute
"""
import warnings
from ..api import vmuls
from ..api import vadds
from ..api import vlog
from ..api import vexp
from ..api import vabs
from ..api import vrec
from ..api import vrelu
from ..api import vnot
from ..api import vsqrt
from ..api import vrsqrt
from ..api import vdiv
from ..api import vmul
from ..api import vadd
from ..api import vsub
from ..api import vmin
from ..api import vmax
from ..api import vor
from ..api import vand
from ..api import vaxpy
from ..api import vmla
from ..api import vmadd
from ..api import vmaddrelu
from ..api import vmaxs
from ..api import vmins
from ..api import vcmp
from ..api import vlogic
from ..api import vsel
from ..api import vcmpsel
from ..api import vmod
from ..api import vlrelu
from ..api import vaddrelu
from ..api import vsubrelu

NAME_INDEX = [0]


def __binary_elewise_op(tensor_l, tensor_r, op_name, args=None):
    """
    factory method of binary elewise operations
    """
    warnings.warn("__binary_elewise_op is deprecated", DeprecationWarning)
    from tbe.dsl.compute.math import __binary_elewise_op
    return __binary_elewise_op(tensor_l, tensor_r, op_name, args)
