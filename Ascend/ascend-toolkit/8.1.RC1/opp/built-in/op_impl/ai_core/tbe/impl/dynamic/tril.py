# Copyright 2023 Huawei Technologies Co., Ltd
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
from impl.util.platform_adapter import para_check, register_operator

from .trilu import Trilu


@register_operator("Tril")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME
)
def tril(x, y, diagonal=0, kernel_name="tril"):
    """
    Returns the lower triangular part of the matrix (2-D tensor)
    or batch of matrices input,
    the other elements of the result tensor out are set to 0.
    See more detail in :
        https://pytorch.org/docs/stable/generated/torch.tril.html
    Parameters:
    x (dict) – the input tensor description.
    out (dict, optional) – the output tensor description.
    diagonal (int, optional) – the diagonal to consider, default=0.
    kernel_name (str): cce kernel name, default value is "tril"
    """
    op = Trilu(x, y, diagonal, upper=False, kernel_name=kernel_name)
    return op.task_schedule()
