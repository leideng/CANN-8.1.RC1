# Copyright 2019 Huawei Technologies Co., Ltd
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
top_k_v2
"""

# 'pylint: disable=unused-argument
# 'pylint: disable=too-many-arguments
def check_supported(input_tensor,
                    indices_tensor,
                    out_tensor,
                    out_indices_tensor,
                    k,
                    sorted=True,
                    dim=-1,
                    largest=True,
                    kernel_name='top_k'):
    
    return "Unknown"