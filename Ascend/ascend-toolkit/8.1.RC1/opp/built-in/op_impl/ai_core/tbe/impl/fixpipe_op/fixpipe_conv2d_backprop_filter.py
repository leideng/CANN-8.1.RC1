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
fixpipe fusion for conv2d_bp_filter
"""

from tbe import tvm
from impl.fixpipe_op.fixpipe_base import FixpipeBase


class FixpipeConv2dBackpropFilter(FixpipeBase):
    """
    conv2d_backprop_filter Fixpipe
    """
    @staticmethod
    def _get_c0_c1_index():
        """
        get c0 c1 index according to format
        """
        return -1, 1

    def fixpipe_reform(self, res):
        """
        shape or format transform for fixpipe_op
        """
        fixpipe_op_name = "fixpipe"
        fixpipe_reform_tag = "fixpipe_reform"
        self.attrs["kernel_name"] = self.x1.op.attrs["kernel_name"]
        if self._is_nz2nd():
            _, _, cout_g, fmap_c0 = tuple(i.value for i in self.x1.shape)
            _, hk_wk, _ = self.output_shape
            res_reform = tvm.compute(self.output_shape,
                                     lambda n_idx, hw_idx, c_idx:
                                        res(n_idx // cout_g,
                                            c_idx // fmap_c0 * hk_wk + hw_idx,
                                            n_idx % cout_g,
                                            c_idx % fmap_c0),
                                     name=fixpipe_op_name + "_nz2nd",
                                     tag=fixpipe_reform_tag,
                                     attrs=self.attrs)
            return res_reform

        if self._is_channel_split():
            _, _, hk_wk, _, fmap_c0 = self.output_shape
            res_reform = tvm.compute(self.output_shape,
                                     lambda g_idx, c1_idx, kk_idx, grads_c_idx, c0_idx:
                                     res(g_idx,
                                         c1_idx // 2 * hk_wk + kk_idx,
                                         grads_c_idx,
                                         c1_idx % 2 * fmap_c0 + c0_idx
                                         ),
                                     name=fixpipe_op_name + "_channel_split",
                                     tag=fixpipe_reform_tag,
                                     attrs=self.attrs)
            return res_reform

        res_reform = tvm.compute(self.output_shape,
                                 lambda *indice: res(*indice),
                                 name=fixpipe_op_name + "_out",
                                 tag=fixpipe_reform_tag,
                                 attrs=self.attrs)
        return res_reform

    def _update_inputs(self):
        """
        skip channel_split tensor and get dw_cc as input tensor.
        """
        if self.x1.op.name == "dw_c_split":
            self.x1 = self.x1.op.input_tensors[0]

    def _get_output_shape(self):
        """
        get output shape
        """
        shape = self.output.get("shape")
        out_shape = shape
        if len(shape) == 4 and self.output.get("format") == "NHWC":
            # 1) input dtype: float16 or float32; out format: Fractal_Z -> NHWC
            out_shape = [shape[0], shape[1] * shape[2], shape[3]]
        elif len(shape) == 4 and self.output.get("format") in ["FRACTAL_Z", "FRACTAL_Z_C04"]:
            # 2) input dtype: float16; out format: FRACTAL_Z
            # (C1HW, N1, N0, C0) -> (real_g, fkk, Cout_g, fmap_c0)
            out_shape = self.x1.shape
        elif len(shape) == 5 and \
            self.output.get("format") == "FRACTAL_Z" and \
            self.output.get("need_channel_split", False):
            # 3) input dtype: float32; out format: FRACTAL_Z (with channel split)
            # (C1, HW, N1, N0, C0) -> (real_g, Cin1_g, kk, Cout_g, fmap_c0)
            out_shape = [1, shape[0], shape[1], shape[2] * shape[3], shape[4]]
        else:
            raise RuntimeError("error output shape or format")

        return out_shape

    def _is_channel_split(self):
        """
        check channel spilt scene
        """
        if self._is_nz2nd():
            return False
        # Conv2d_bp_filter only perform channel split when input dtype is float32
        if self.output.get("need_channel_split", False):
            return True
        return False
