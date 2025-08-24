#!/usr/bin/env python
# coding: utf-8
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
tbe register
"""
from tbe.common.register.register_api import register_op_compute
from tbe.common.register.register_api import get_op_compute
from tbe.common.register.register_api import register_operator
from tbe.common.register.register_api import get_operator
from tbe.common.register.register_api import register_param_generalization
from tbe.common.register.register_api import get_param_generalization
from tbe.common.register.register_api import register_fusion_pass
from tbe.common.register.register_api import get_all_fusion_pass
from tbe.common.register.register_api import set_fusion_buildcfg
from tbe.common.register.register_api import get_fusion_buildcfg
from tbe.common.register.register_api import reset
from tbe.common.register.register_api import register_tune_space
from tbe.common.register.register_api import get_tune_space
from tbe.common.register.register_api import register_tune_param_check_supported
from tbe.common.register.register_api import get_tune_param_check_supported
from tbe.common.register.register_api import get_op_register_pattern
from tbe.common.register.register_api import register_pass_for_fusion
from tbe.common.register.register_api import register_op_param_pass

from tbe.common.register.class_manager import InvokeStage
from tbe.common.register.class_manager import Priority
from tbe.common.register.class_manager import FusionPassItem
from tbe.common.register.class_manager import OpCompute
from tbe.common.register.class_manager import Operator

from . import fusion_pass
