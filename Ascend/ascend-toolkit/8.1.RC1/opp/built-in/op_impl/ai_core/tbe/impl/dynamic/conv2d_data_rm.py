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
remove dirty data in M axis.
"""

from collections import deque
from tbe import tvm
from tbe.common.utils.errormgr import error_manager_cube as err_man_cube
from impl.util.platform_adapter import register_operator_compute

INPUT_DIM_5HD = 4  # [N, C1, HOWO, C0]
INPUT_DIM_ND = 3   # [N, C, H, W]
COMPUTE_INDEX = [0]


@register_operator_compute("conv2d_data_rm", op_mode="dynamic", support_fusion=True)
def conv2d_data_rm_compute(input_tensor, res_tensor=None):
    """
    Compute for removing dirty data of tensor in M axis.

    Parameters
    ----------
    input_tensor: input tensor.

    res_tensor: res tensor set by Tefusion.

    Returns
    -------
    output_tensor: output tensor after removing pad.
    """
    def remove_pad_common(out_shape, name, tag):
        if len(out_shape) == INPUT_DIM_5HD:
            res = tvm.compute(
                out_shape,
                lambda batch_idx, c1_idx, hw_idx, c0_idx:
                input_tensor[batch_idx, c1_idx, hw_idx, c0_idx],
                name=name,
                tag=tag)
        else:
            res = tvm.compute(
                out_shape,
                lambda batch_idx, c_idx, hw_idx:
                input_tensor[batch_idx, c_idx, hw_idx],
                name=name,
                tag=tag)
        return res

    def remove_pad_row(attrs, out_shape, name, tag):
        def get_howo_res_idx(howo_idx):
            # only binary conv1d support split_w, there is bug in pass, use default remove pad to evade bug
            _ = out_width
            _ = out_width_aligned
            return howo_idx

        if "out_width" not in attrs or "out_width_aligned" not in attrs:
            err_man_cube.raise_err_message_cube("key out_width or out_width_aligned not in attrs for remove pad row")

        out_width = attrs["out_width"]
        out_width_aligned = attrs["out_width_aligned"]
        if len(out_shape) == INPUT_DIM_5HD:
            res = tvm.compute(
                out_shape,
                lambda batch_idx, c1_idx, hw_idx, c0_idx:
                input_tensor(batch_idx, c1_idx, get_howo_res_idx(hw_idx), c0_idx),
                name=name,
                tag=tag)
        else:
            res = tvm.compute(
                out_shape,
                lambda batch_idx, c_idx, hw_idx:
                input_tensor(batch_idx, c_idx, get_howo_res_idx(hw_idx)),
                name=name,
                tag=tag)
        return res

    def get_output_shape(attrs):
        if "remove_pad_M" not in attrs:
            err_man_cube.raise_err_message_cube("key remove_pad_M not in attrs")

        output_hw = attrs["remove_pad_M"]
        if len(input_tensor.shape) == INPUT_DIM_5HD:
            batch, co1, hw_mad, co0 = input_tensor.shape
            output_shape_ = (batch, co1, output_hw, co0)
        else:
            batch, c, hw_mad = input_tensor.shape
            output_shape_ = (batch, c, output_hw)

        return output_shape_

    def get_split_w_flag(attrs):
        # bool in op attrs is int type
        split_w_flag_ = 0
        if "split_w_flag" in attrs:
            split_w_flag_ = int(attrs["split_w_flag"])

        return split_w_flag_

    if type(input_tensor) != tvm.Tensor:
        err_man_cube.raise_err_one_para("E62006", "conv2d_data_rm", "The tpye of input should be tensor, "
                                        "but actually it is {}.".format(type(input_tensor)))
    if input_tensor.dtype not in ("int4", "int8", "float16"):
        err_man_cube.raise_err_one_para("E62006", "conv2d_data_rm", "The input_tensor dtype should be int4, int8 or "
                                        "float16, but actually it is {}.".format(input_tensor.dtype))

    if len(input_tensor.shape) == INPUT_DIM_ND or len(input_tensor.shape) == INPUT_DIM_5HD:
        tensor_queue = deque()
        tensor_queue.append(input_tensor)

        while tensor_queue:
            src_tensor = tensor_queue.popleft()
            if "mad1" in src_tensor.op.name:
                op_attrs = src_tensor.op.attrs
                output_shape = get_output_shape(op_attrs)
                split_w_flag = get_split_w_flag(op_attrs)
                tensor_name = "conv2d_data_rm_" + str(COMPUTE_INDEX[0])
                tensor_tag = "conv2d_data_rm"

                if split_w_flag == 1:
                    output_tensor = remove_pad_row(op_attrs, output_shape, tensor_name, tensor_tag)
                else:
                    output_tensor = remove_pad_common(output_shape, tensor_name, tensor_tag)

                COMPUTE_INDEX[0] += 1
                return output_tensor

            if src_tensor.op.input_tensors:
                tensor_queue.extend(list(i for i in src_tensor.op.input_tensors))

        err_man_cube.raise_err_specific("conv2d_data_rm", "conv2d_data_rm Cannot find remove_align_data_M information "
                                        "after traversing all input tensors!")
    else:
        err_man_cube.raise_err_specific("conv2d_data_rm", "Wrong input_tensor shape {}, \
                                         the format should be [N C1 HW C0] or [N C HW]!".format(input_tensor.shape))
