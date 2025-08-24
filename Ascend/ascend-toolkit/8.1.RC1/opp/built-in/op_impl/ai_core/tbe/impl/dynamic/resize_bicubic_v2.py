# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
resize_d
"""

import math
from collections import namedtuple
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import platform as cce

from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # input shape indices NCHW
    N_IDX = 0
    C_IDX = 1
    H_IDX = 2
    W_IDX = 3

    # size shape indices HW
    SIZES_H_IDX = 0
    SIZES_W_IDX = 1

    # constant parameters in calculation
    VECTOR_MASK_MAX = 64
    BLOCK_NUM_FP32 = 8
    STRIDE_FP16 = 4

    # tensor shape params
    MAX_INT32 = 2 ** 31 - 1
    MAX_LINE_NUM = 10240
    TILING_ARG_NUM = 11

    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024


class ResizeBicubicV2():
    """
    Function: use to store ResizeTrilinear base parameters
    """

    def __init__(self, images, size, y, align_corners, half_pixel_centers, cubic_coeff_a=-0.75,
                    kernel_name="resizebicubicV2"):
        """
        init ResizeBicubic base parameters

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(disable_debug=False)
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers
        self.kernel_name = kernel_name
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB_SIZE
        
        # inner calculate base
        self.block_bite_size = 32
        self.inner_dtype = "float32"
        self.x_dtype = images.get("dtype").lower()
        self.inner_data_each_block = self.block_bite_size * 8 // cce.get_bit_len(self.inner_dtype)
        self.x_data_each_block = self.block_bite_size * 8 // cce.get_bit_len(self.x_dtype)

        # compile info
        self.ub_max_num = self.ub_size_bytes // 32 // 2 * self.inner_data_each_block
        self.input_channel0 = 16

        # formula related
        self.coeff = cubic_coeff_a
        self.coeff_plus_2 = cubic_coeff_a + 2
        self.coeff_plus_3 = cubic_coeff_a + 3
        self.double_coeff_plus_3 = cubic_coeff_a * 2 + 3

        self.scalar_half = 0.5
        self.scalar_negative_half = -0.5
        self.scalar_one = 1
        self.scalar_negative_one = -1

        self.init_scalar()
        self.tiling_data_init()
        self.init_gm_tensor()
        self.init_ub_tensor()

    @staticmethod
    def get_dtype_size(dtype):
        """
        :param dtype: data type

        Returns
        -------
        None
        """
        dtype_byte_size = get_bit_len(dtype) // 8
        return dtype_byte_size

    def init_scalar(self):
        self.tiling_key = self.tik_instance.Scalar("int64", "tiling_key")
        self.input_channel1 = self.tik_instance.Scalar("int64", "input_channel1")
        self.input_height = self.tik_instance.Scalar("int64", "input_height")
        self.input_width = self.tik_instance.Scalar("int64", "input_width")
        self.output_height = self.tik_instance.Scalar("int64", "output_height")
        self.output_width = self.tik_instance.Scalar("int64", "output_width")
        self.tiling_nc1_cut_num = self.tik_instance.Scalar("int64", "tiling_nc1_cut_num")
        self.tiling_height_cut_num = self.tik_instance.Scalar("int64", "tiling_height_cut_num")
        self.tiling_width_cut_num = self.tik_instance.Scalar("int64", "tiling_width_cut_num")
        self.running_core_num = self.tik_instance.Scalar("int64", "running_core_num")

        self.need_core = self.tik_instance.Scalar("int64", "need_core")
        self.max_height = self.tik_instance.Scalar("int64", "max_height")
        self.max_width = self.tik_instance.Scalar("int64", "max_width")
        self.input_line_size = self.tik_instance.Scalar("int64", "input_line_size")
        self.output_line_size = self.tik_instance.Scalar("int64", "output_line_size")
        self.scale_height = self.tik_instance.Scalar("float32", "scale_height")
        self.scale_width = self.tik_instance.Scalar("float32", "scale_width")

        # split params in huge shape cases
        self.split_input_gap = self.tik_instance.Scalar("float32", "split_input_gap")
        self.split_input_width = self.tik_instance.Scalar("int64", "split_input_width")

        self.split_output_width = self.tik_instance.Scalar("int64", "split_output_width")
        self.actually_width_cut = self.tik_instance.Scalar("int64", "actually_width_cut")
        self.tail_output_width = self.tik_instance.Scalar("int64", "tail_output_width")
        self.oper_output_width = self.tik_instance.Scalar("int64", "oper_output_width")

        self.weight_width_cut = self.tik_instance.Scalar("int64", "weight_width_cut")
        self.weight_width_tail = self.tik_instance.Scalar("int64", "weight_width_tail")
        self.oper_weight_width = self.tik_instance.Scalar("int64", "oper_weight_width")

        self.weight_height_cut = self.tik_instance.Scalar("int64", "weight_height_cut")
        self.weight_height_tail = self.tik_instance.Scalar("int64", "weight_height_tail")
        self.oper_weight_height = self.tik_instance.Scalar("int64", "oper_weight_height")

        self.images_gm_size = self.tik_instance.Scalar("int64", "images_gm_size")
        self.out_gm_size = self.tik_instance.Scalar("int64", "out_gm_size")

    def tiling_data_init(self):
        """
        init tiling gm and scalar

        Returns
        -------
        None

        """
        self.tiling_gm = self.tik_instance.Tensor("int64", [Constant.TILING_ARG_NUM],
                                                   name="tiling_gm", scope=tik.scope_gm)

        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int64", [Constant.TILING_ARG_NUM],
                                                   name="tiling_ub", scope=tik.scope_ubuf)
            self.data_move(tiling_ub, self.tiling_gm, [0, 0], Constant.TILING_ARG_NUM)
            self.tiling_key.set_as(tiling_ub[0])
            self.input_channel1.set_as(tiling_ub[2])
            self.input_height.set_as(tiling_ub[3])
            self.input_width.set_as(tiling_ub[4])
            self.output_height.set_as(tiling_ub[5])
            self.output_width.set_as(tiling_ub[6])
            self.workspace_height = (self.output_height + 7) // 8 * 8
            self.workspace_width = (self.output_width + 7) // 8 * 8
            self.tiling_nc1_cut_num.set_as(tiling_ub[7])
            self.tiling_height_cut_num.set_as(tiling_ub[8])
            self.tiling_width_cut_num.set_as(tiling_ub[9])
            self.running_core_num.set_as(self.tiling_nc1_cut_num * self.tiling_height_cut_num
                                         * self.tiling_width_cut_num)

            self.min_index = 0
            self.max_height.set_as(self.input_height - 1)
            self.max_width.set_as(self.input_width - 1)
            self.input_line_size.set_as(self.input_width * self.input_channel0)
            self.output_line_size.set_as(self.output_width * self.input_channel0)
            temp_input_height = self.tik_instance.Scalar("float32", "temp_input_height", init_value=self.input_height)
            temp_input_width = self.tik_instance.Scalar("float32", "temp_input_width", init_value=self.input_width)
            temp_output_height = self.tik_instance.Scalar("float32", "temp_output_height",
                                                            init_value=self.output_height)
            temp_output_width = self.tik_instance.Scalar("float32", "temp_output_width", init_value=self.output_width)
            if (self.align_corners):
                self.scale_height.set_as((temp_input_height - 1) / (temp_output_height - 1))
                self.scale_width.set_as((temp_input_width - 1) / (temp_output_width - 1))
            elif (self.half_pixel_centers):
                self.scale_height.set_as(temp_input_height / temp_output_height)
                self.scale_width.set_as(temp_input_width / temp_output_width)

            self.core_index = 0
            self.split_output_width.set_as(256)
            self.tik_instance.scalar_min(self.split_output_width, self.split_output_width, self.output_width)
            self.split_input_gap.set_as(self.split_output_width * self.scale_width)
            self.split_input_width.set_as(self.split_input_gap + 6)
            self.actually_width_cut.set_as((self.output_width + self.split_output_width - 1) / self.split_output_width)
            self.tail_output_width.set_as(self.output_width % self.split_output_width)
            self.need_core.set_as(self.input_channel1 * self.output_height * self.actually_width_cut)

            self.weight_cut = 256
            self.weight_width_cut.set_as((self.output_width + self.weight_cut - 1) // self.weight_cut)
            self.weight_width_tail.set_as(self.output_width % self.weight_cut)
            self.weight_height_cut.set_as((self.output_height + self.weight_cut - 1) // self.weight_cut)
            self.weight_height_tail.set_as(self.output_height % self.weight_cut)
            
            # only record element num
            self.images_gm_size.set_as(self.input_channel1 * self.input_height * self.input_width * self.input_channel0)
            self.out_gm_size.set_as(self.input_channel1 * self.output_height * self.output_width * self.input_channel0)

    def init_gm_tensor(self):
        """
        init tensor in ub

        Returns
        -------
        None
        """
        max_dim_value = list(range(Constant.MAX_LINE_NUM))
        self.images_gm = self.tik_instance.Tensor(self.x_dtype, [Constant.MAX_INT32],
                                                   name="images_gm", scope=tik.scope_gm)
        self.rois_gm = self.tik_instance.Tensor(self.x_dtype, [Constant.MAX_INT32],
                                                   name="rois_gm", scope=tik.scope_gm)
        self.scale_gm = self.tik_instance.Tensor(self.x_dtype, [Constant.MAX_INT32],
                                                   name="scale_gm", scope=tik.scope_gm)
        self.size_gm = self.tik_instance.Tensor(self.x_dtype, [Constant.MAX_INT32],
                                                   name="size_gm", scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.x_dtype, [Constant.MAX_INT32],
                                                   name="out_gm", scope=tik.scope_gm, is_atomic_add=True)
        self.dst_idx_gm = self.tik_instance.Tensor(self.inner_dtype, [Constant.MAX_LINE_NUM],
                                                    name="dst_idx_gm", scope=tik.scope_gm, init_value=max_dim_value)

        self.workspace_h_weight_1 = self.tik_instance.Tensor(self.inner_dtype, [Constant.MAX_INT32],
                                                   name="workspace_h_weight_1", scope=tik.scope_gm, is_workspace=True)
        self.workspace_h_weight_2 = self.tik_instance.Tensor(self.inner_dtype, [Constant.MAX_INT32],
                                                   name="workspace_h_weight_2", scope=tik.scope_gm, is_workspace=True)
        self.workspace_h_weight_3 = self.tik_instance.Tensor(self.inner_dtype, [Constant.MAX_INT32],
                                                   name="workspace_h_weight_3", scope=tik.scope_gm, is_workspace=True)
        self.workspace_h_weight_4 = self.tik_instance.Tensor(self.inner_dtype, [Constant.MAX_INT32],
                                                   name="workspace_h_weight_4", scope=tik.scope_gm, is_workspace=True)
        self.workspace_w_weight_1 = self.tik_instance.Tensor(self.inner_dtype, [Constant.MAX_INT32],
                                                   name="workspace_w_weight_1", scope=tik.scope_gm, is_workspace=True)
        self.workspace_w_weight_2 = self.tik_instance.Tensor(self.inner_dtype, [Constant.MAX_INT32],
                                                   name="workspace_w_weight_2", scope=tik.scope_gm, is_workspace=True)
        self.workspace_w_weight_3 = self.tik_instance.Tensor(self.inner_dtype, [Constant.MAX_INT32],
                                                   name="workspace_w_weight_3", scope=tik.scope_gm, is_workspace=True)
        self.workspace_w_weight_4 = self.tik_instance.Tensor(self.inner_dtype, [Constant.MAX_INT32],
                                                   name="workspace_w_weight_4", scope=tik.scope_gm, is_workspace=True)
        self.workspace_index_h_mapping = self.tik_instance.Tensor("int32", [Constant.MAX_INT32], scope=tik.scope_gm,
                                                   name="workspace_index_h_mapping", is_workspace=True)
        self.workspace_index_w_mapping = self.tik_instance.Tensor("int32", [Constant.MAX_INT32], scope=tik.scope_gm,
                                                   name="workspace_index_w_mapping", is_workspace=True)

    def resize_bicubic_v2_operator(self):
        """
        main entrance of the calculation process
        """
        batch_core_num = self.need_core // self.running_core_num
        batch_core_tail = self.need_core % self.running_core_num
        with self.tik_instance.for_range(0, self.running_core_num, block_num=self.running_core_num) as core_idx:
            self.core_index = core_idx
            self.calc_index_diff()
            with self.tik_instance.for_range(0, batch_core_num) as batch:
                self.compute_core(core_idx + batch * self.running_core_num)
            with self.tik_instance.if_scope(self.core_index < batch_core_tail):
                self.compute_core(batch_core_num * self.running_core_num + core_idx)

        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size_bytes,
                "core_num": self.core_num,
                "max_w_len": self.ub_max_num // self.input_channel0,
                "align_corners": int(self.align_corners),
                "half_pixel_centers": int(self.half_pixel_centers),
                "mode_name": 23
            })
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True, "save_temp_cce_file": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.images_gm, self.rois_gm, self.scale_gm, self.size_gm],
                                   outputs=[self.out_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance

    def init_ub_tensor(self):
        """
        init tensor in ub
        """
        self.temp_line_ub = self.tik_instance.Tensor(self.inner_dtype, [self.split_input_width * self.input_channel0],
                                                     name="temp_line_ub", scope=tik.scope_ubuf)
        self.line_ub = self.tik_instance.Tensor(self.inner_dtype, [self.split_output_width * self.input_channel0],
                                                name="line_ub", scope=tik.scope_ubuf)
        self.index_h_mapping_ub = self.tik_instance.Tensor("int32", [self.weight_cut],
                                                           name="index_h_mapping_ub", scope=tik.scope_ubuf)
        self.h_weight = self.tik_instance.Tensor(self.inner_dtype, [self.weight_cut],
                                                 name="h_weight", scope=tik.scope_ubuf)
        self.index_w_mapping_ub = self.tik_instance.Tensor("int32", [self.weight_cut],
                                                           name="index_w_mapping_ub", scope=tik.scope_ubuf)
        self.w_weight = self.tik_instance.Tensor(self.inner_dtype, [self.weight_cut],
                                                 name="w_weight", scope=tik.scope_ubuf)

    def calc_index_diff(self):
        """
        calc mapping index and diff to nearest left integer
        """
        self.oper_weight_height.set_as(self.weight_cut)
        with self.tik_instance.for_range(0, self.weight_height_cut) as height_idx:
            with self.tik_instance.if_scope(tik.all(height_idx == self.weight_height_cut - 1,
                                                    0 != self.weight_height_tail)):
                self.oper_weight_height.set_as(self.weight_height_tail)
            h_diff = self.tik_instance.Tensor(self.inner_dtype, [self.weight_cut],
                                              name="h_diff", scope=tik.scope_ubuf)
            h_ub_1 = self.tik_instance.Tensor(self.inner_dtype, [self.weight_cut],
                                              name="h_ub_1", scope=tik.scope_ubuf)
            h_ub_2 = self.tik_instance.Tensor(self.inner_dtype, [self.weight_cut],
                                              name="h_ub_2", scope=tik.scope_ubuf)
            self.data_move(self.h_weight, self.dst_idx_gm, [0, height_idx * self.weight_cut], self.oper_weight_height)
            if (self.align_corners):
                self.scalar_operator_template(self.tik_instance.vec_muls, h_ub_2, self.h_weight,
                                              self.scale_height, [0, 0], self.oper_weight_height)
            else:
                self.scalar_operator_template(self.tik_instance.vec_adds, h_ub_1, self.h_weight,
                                              self.scalar_half, [0, 0], self.oper_weight_height)
                self.scalar_operator_template(self.tik_instance.vec_muls, h_ub_2, h_ub_1,
                                              self.scale_height, [0, 0], self.oper_weight_height)
                self.scalar_operator_template(self.tik_instance.vec_adds, h_ub_2, h_ub_2,
                                              self.scalar_negative_half, [0, 0], self.oper_weight_height)
            self.conv_operator_template(self.tik_instance.vec_conv, "floor",
                                        self.index_h_mapping_ub, h_ub_2, [0, 0], self.oper_weight_height)
            self.data_move(self.workspace_index_h_mapping, self.index_h_mapping_ub,
                           [self.core_index * self.workspace_height + height_idx * self.weight_cut, 0],
                           self.oper_weight_height)
            self.conv_operator_template(self.tik_instance.vec_conv, "none", h_ub_1,
                                        self.index_h_mapping_ub, [0, 0], self.oper_weight_height)
            self.double_operator_template(self.tik_instance.vec_sub, h_diff,
                                          h_ub_2, h_ub_1, [0, 0, 0], self.oper_weight_height)

            
            self.scalar_operator_template(self.tik_instance.vec_adds, h_ub_1,
                                          h_diff, self.scalar_negative_one, [0, 0], self.oper_weight_height)

            # W1
            self.double_operator_template(self.tik_instance.vec_mul, self.h_weight,
                                          h_diff, h_ub_1, [0, 0, 0], self.oper_weight_height)
            self.double_operator_template(self.tik_instance.vec_mul, self.h_weight,
                                          self.h_weight, h_ub_1, [0, 0, 0], self.oper_weight_height)
            self.scalar_operator_template(self.tik_instance.vec_muls, self.h_weight,
                                          self.h_weight, self.coeff, [0, 0], self.oper_weight_height)
            self.data_move(self.workspace_h_weight_1, self.h_weight,
                           [self.core_index * self.workspace_height + height_idx * self.weight_cut, 0],
                           self.oper_weight_height)

            # W4
            self.scalar_operator_template(self.tik_instance.vec_muls, self.h_weight,
                                          h_diff, self.scalar_negative_one, [0, 0], self.oper_weight_height)
            self.double_operator_template(self.tik_instance.vec_mul, self.h_weight,
                                          self.h_weight, h_diff, [0, 0, 0], self.oper_weight_height)
            self.double_operator_template(self.tik_instance.vec_mul, self.h_weight,
                                          self.h_weight, h_ub_1, [0, 0, 0], self.oper_weight_height)
            self.scalar_operator_template(self.tik_instance.vec_muls, self.h_weight,
                                          self.h_weight, self.coeff, [0, 0], self.oper_weight_height)
            self.data_move(self.workspace_h_weight_4, self.h_weight,
                           [self.core_index * self.workspace_height + height_idx * self.weight_cut, 0],
                           self.oper_weight_height)

            # W2
            self.double_operator_template(self.tik_instance.vec_mul, h_ub_1,
                                          h_diff, h_diff, [0, 0, 0], self.oper_weight_height)
            self.scalar_operator_template(self.tik_instance.vec_muls, h_ub_2,
                                          h_ub_1, self.coeff_plus_3, [0, 0], self.oper_weight_height)
            self.scalar_operator_template(self.tik_instance.vec_muls, self.h_weight,
                                          h_ub_2, self.scalar_negative_one, [0, 0], self.oper_weight_height)

            self.double_operator_template(self.tik_instance.vec_mul, h_ub_1,
                                          h_ub_1, h_diff, [0, 0, 0], self.oper_weight_height)
            self.scalar_operator_template(self.tik_instance.vec_muls, h_ub_2,
                                          h_ub_1, self.coeff_plus_2, [0, 0], self.oper_weight_height)
            self.double_operator_template(self.tik_instance.vec_add, self.h_weight, self.h_weight, h_ub_2,
                                          [0, 0, 0], self.oper_weight_height)

            self.scalar_operator_template(self.tik_instance.vec_adds, self.h_weight,
                                          self.h_weight, self.scalar_one, [0, 0], self.oper_weight_height)
            self.data_move(self.workspace_h_weight_2, self.h_weight,
                           [self.core_index * self.workspace_height + height_idx * self.weight_cut, 0],
                           self.oper_weight_height)

            # W3
            self.scalar_operator_template(self.tik_instance.vec_muls, self.h_weight, h_diff, self.coeff,
                                          [0, 0], self.oper_weight_height)
            self.scalar_operator_template(self.tik_instance.vec_muls, self.h_weight, self.h_weight, 
                                          self.scalar_negative_one, [0, 0], self.oper_weight_height)
            
            self.scalar_operator_template(self.tik_instance.vec_muls, h_ub_2, h_ub_1, self.coeff_plus_2,
                                          [0, 0], self.oper_weight_height)
            self.double_operator_template(self.tik_instance.vec_sub, self.h_weight, self.h_weight, h_ub_2,
                                          [0, 0, 0], self.oper_weight_height)

            self.double_operator_template(self.tik_instance.vec_mul, h_ub_1, h_diff, h_diff,
                                          [0, 0, 0], self.oper_weight_height)
            self.scalar_operator_template(self.tik_instance.vec_muls, h_ub_2, h_ub_1, self.double_coeff_plus_3,
                                          [0, 0], self.oper_weight_height)
            self.double_operator_template(self.tik_instance.vec_add, self.h_weight, self.h_weight, h_ub_2,
                                          [0, 0, 0], self.oper_weight_height)

            self.data_move(self.workspace_h_weight_3, self.h_weight,
                           [self.core_index * self.workspace_height + height_idx * self.weight_cut, 0],
                           self.oper_weight_height)

        
        self.oper_weight_width.set_as(self.weight_cut)
        with self.tik_instance.for_range(0, self.weight_width_cut) as width_idx:
            with self.tik_instance.if_scope(tik.all(width_idx == self.weight_width_cut - 1,
                                                    0 != self.weight_width_tail)):
                self.oper_weight_width.set_as(self.weight_width_tail)
            w_diff = self.tik_instance.Tensor(self.inner_dtype, [self.weight_cut],
                                                    name="w_diff", scope=tik.scope_ubuf)
            w_ub_1 = self.tik_instance.Tensor(self.inner_dtype, [self.weight_cut],
                                                    name="w_ub_1", scope=tik.scope_ubuf)
            w_ub_2 = self.tik_instance.Tensor(self.inner_dtype, [self.weight_cut],
                                                    name="w_ub_2", scope=tik.scope_ubuf)

            self.data_move(self.w_weight, self.dst_idx_gm, [0, width_idx * self.weight_cut], self.oper_weight_width)
            if (self.align_corners):
                self.scalar_operator_template(self.tik_instance.vec_muls, w_ub_2, self.w_weight,
                                                self.scale_width, [0, 0], self.oper_weight_width)
            else:
                self.scalar_operator_template(self.tik_instance.vec_adds, w_ub_1,
                                                self.w_weight, self.scalar_half, [0, 0], self.oper_weight_width)
                self.scalar_operator_template(self.tik_instance.vec_muls, w_ub_2, w_ub_1,
                                                self.scale_width, [0, 0], self.oper_weight_width)
                self.scalar_operator_template(self.tik_instance.vec_adds, w_ub_2, w_ub_2,
                                                self.scalar_negative_half, [0, 0], self.oper_weight_width)
            self.conv_operator_template(self.tik_instance.vec_conv, "floor",
                                        self.index_w_mapping_ub, w_ub_2, [0, 0], self.oper_weight_width)
            self.data_move(self.workspace_index_w_mapping, self.index_w_mapping_ub,
                           [self.core_index * self.workspace_width + width_idx * self.weight_cut, 0],
                           self.oper_weight_width)
            self.conv_operator_template(self.tik_instance.vec_conv, "none",
                                        w_ub_1, self.index_w_mapping_ub, [0, 0], self.oper_weight_width)
            self.double_operator_template(self.tik_instance.vec_sub,
                                        w_diff, w_ub_2, w_ub_1, [0, 0, 0], self.oper_weight_width)
            """
            Calculate weight
            W1 = Ar(r-1)^2
            W2 = (A+2)r^3 - (A+3)r^2 + 1
            W3 = -(A+2)r^3 + (2A+3)r^2 -Ar
            W4 = -Ar^2(r-1)
            """
            self.scalar_operator_template(self.tik_instance.vec_adds, w_ub_1, w_diff, self.scalar_negative_one,
                                            [0, 0], self.oper_weight_width)

            # W1
            self.double_operator_template(self.tik_instance.vec_mul, self.w_weight, w_diff, w_ub_1,
                                            [0, 0, 0], self.oper_weight_width)
            self.double_operator_template(self.tik_instance.vec_mul, self.w_weight, self.w_weight, w_ub_1,
                                            [0, 0, 0], self.oper_weight_width)
            self.scalar_operator_template(self.tik_instance.vec_muls, self.w_weight, self.w_weight, self.coeff,
                                            [0, 0], self.oper_weight_width)
            self.data_move(self.workspace_w_weight_1, self.w_weight,
                           [self.core_index * self.workspace_width + width_idx * self.weight_cut, 0],
                           self.oper_weight_width)

            # W4
            self.scalar_operator_template(self.tik_instance.vec_muls, self.w_weight,
                                            w_diff, self.scalar_negative_one, [0, 0], self.oper_weight_width)
            self.double_operator_template(self.tik_instance.vec_mul, self.w_weight, self.w_weight, w_diff,
                                            [0, 0, 0], self.oper_weight_width)
            self.double_operator_template(self.tik_instance.vec_mul, self.w_weight, self.w_weight, w_ub_1,
                                            [0, 0, 0], self.oper_weight_width)
            self.scalar_operator_template(self.tik_instance.vec_muls, self.w_weight, self.w_weight, self.coeff,
                                            [0, 0], self.oper_weight_width)
            self.data_move(self.workspace_w_weight_4, self.w_weight,
                           [self.core_index * self.workspace_width + width_idx * self.weight_cut, 0],
                           self.oper_weight_width)

            # W2
            self.double_operator_template(self.tik_instance.vec_mul, w_ub_1,
                                            w_diff, w_diff, [0, 0, 0], self.oper_weight_width)
            self.scalar_operator_template(self.tik_instance.vec_muls, w_ub_2,
                                            w_ub_1, self.coeff_plus_3, [0, 0], self.oper_weight_width)
            self.scalar_operator_template(self.tik_instance.vec_muls, self.w_weight,
                                            w_ub_2, self.scalar_negative_one, [0, 0], self.oper_weight_width)

            self.double_operator_template(self.tik_instance.vec_mul, w_ub_1,
                                            w_ub_1, w_diff, [0, 0, 0], self.oper_weight_width)
            self.scalar_operator_template(self.tik_instance.vec_muls, w_ub_2,
                                            w_ub_1, self.coeff_plus_2, [0, 0], self.oper_weight_width)
            self.double_operator_template(self.tik_instance.vec_add, self.w_weight, self.w_weight, w_ub_2,
                                            [0, 0, 0], self.oper_weight_width)

            self.scalar_operator_template(self.tik_instance.vec_adds, self.w_weight,
                                            self.w_weight, self.scalar_one, [0, 0], self.oper_weight_width)
            self.data_move(self.workspace_w_weight_2, self.w_weight,
                           [self.core_index * self.workspace_width + width_idx * self.weight_cut, 0],
                           self.oper_weight_width)

            # W3
            self.scalar_operator_template(self.tik_instance.vec_muls, self.w_weight, w_diff, self.coeff,
                                            [0, 0], self.oper_weight_width)
            self.scalar_operator_template(self.tik_instance.vec_muls, self.w_weight, self.w_weight, 
                                            self.scalar_negative_one, [0, 0], self.oper_weight_width)
            
            self.scalar_operator_template(self.tik_instance.vec_muls, w_ub_2, w_ub_1, self.coeff_plus_2,
                                            [0, 0], self.oper_weight_width)
            self.double_operator_template(self.tik_instance.vec_sub, self.w_weight, self.w_weight, w_ub_2,
                                            [0, 0, 0], self.oper_weight_width)

            self.double_operator_template(self.tik_instance.vec_mul, w_ub_1, w_diff, w_diff,
                                            [0, 0, 0], self.oper_weight_width)
            self.scalar_operator_template(self.tik_instance.vec_muls, w_ub_2, w_ub_1, self.double_coeff_plus_3,
                                            [0, 0], self.oper_weight_width)
            self.double_operator_template(self.tik_instance.vec_add, self.w_weight, self.w_weight, w_ub_2,
                                            [0, 0, 0], self.oper_weight_width)

            self.data_move(self.workspace_w_weight_3, self.w_weight,
                           [self.core_index * self.workspace_width + width_idx * self.weight_cut, 0],
                           self.oper_weight_width)

    def compute_core(self, core_idx):
        """
        core function of computer
        Parameters:
        ----------
        core_idx : int64.
            The index of the core
        """
        core_height_idx = core_idx // self.actually_width_cut
        core_width_idx = core_idx % self.actually_width_cut

        self.get_weighted_src_line(core_height_idx, core_width_idx)

        self.oper_output_width.set_as(self.split_output_width)
        with self.tik_instance.if_scope(tik.all(core_width_idx == self.actually_width_cut - 1,
                                                0 != self.tail_output_width)):
            self.oper_output_width.set_as(self.tail_output_width)

        self.calc_output_line(core_width_idx)

        if self.x_dtype == "float16":
            output_cast_line = self.tik_instance.Tensor(self.x_dtype, [self.oper_output_width * self.input_channel0],
                                                    name="output_cast_line", scope=tik.scope_ubuf)
            self.conv_operator_template(self.tik_instance.vec_conv, "none", output_cast_line, self.line_ub,
                                        [0, 0], self.oper_output_width * self.input_channel0, 4, 8)
            self.data_move_pad(self.out_gm, output_cast_line,
                           [core_height_idx * self.output_line_size +
                           core_width_idx * self.split_output_width * self.input_channel0, 0],
                           self.oper_output_width * self.input_channel0, self.out_gm_size, "ub2gm")
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                self.tik_instance.set_flag("PIPE_MTE3", "PIPE_MTE2", 0)
                self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_MTE2", 0)
        else:
            self.data_move_pad(self.out_gm, self.line_ub,
                           [core_height_idx * self.output_line_size +
                           core_width_idx * self.split_output_width * self.input_channel0, 0],
                           self.oper_output_width * self.input_channel0, self.out_gm_size, "ub2gm")

    def get_weighted_src_line(self, core_height_idx, core_width_idx):
        """
        obtain related four line and calc weighted average
        Parameters:
        ----------
        core_height_idx : int64.
            The height index of the operating slice
        core_width_idx : int64.
            The width splited index of the operating slice
        """
        with self.tik_instance.new_stmt_scope():
            # calculate split line start index
            input_index_start = self.tik_instance.Scalar("int64", "input_index_start")
            input_index_start.set_as(self.split_input_gap * core_width_idx - 2)
            self.tik_instance.scalar_max(input_index_start, self.min_index, input_index_start)
            nc_idx = core_height_idx // self.output_height
            h_out_index = core_height_idx % self.output_height

            src_line = self.tik_instance.Tensor(self.inner_dtype, [self.split_input_width * self.input_channel0],
                                                name="src_line", scope=tik.scope_ubuf)
            ub_cast_line = self.tik_instance.Tensor(self.x_dtype, [self.split_input_width * self.input_channel0],
                                                    name="ub_cast_line", scope=tik.scope_ubuf)

            # calculate horizontal indies
            h_in_index = self.tik_instance.Scalar("int64", "h_in_index")
            h_in_index.set_as(self.workspace_index_h_mapping[self.core_index * self.workspace_height + h_out_index])
            index_1 = self.tik_instance.Scalar("int64", "index_1", init_value=h_in_index - 1)
            index_2 = self.tik_instance.Scalar("int64", "index_2", init_value=h_in_index)
            index_3 = self.tik_instance.Scalar("int64", "index_3", init_value=h_in_index + 1)
            index_4 = self.tik_instance.Scalar("int64", "index_4", init_value=h_in_index + 2)
            temp_h_weight = self.tik_instance.Scalar("float32", "temp_h_weight")

            self.tik_instance.scalar_max(index_1, self.min_index, index_1)
            self.tik_instance.scalar_max(index_2, self.min_index, index_2)
            self.tik_instance.scalar_min(index_3, index_3, self.max_height)
            self.tik_instance.scalar_min(index_4, index_4, self.max_height)

            # move four line snippets from gm to ub
            if self.x_dtype == "float16":
                self.data_move_pad(ub_cast_line, self.images_gm,
                               [0, (nc_idx * self.input_height + index_1) * self.input_line_size +
                               input_index_start * self.input_channel0],
                               self.split_input_width * self.input_channel0, 
                               self.images_gm_size, "gm2ub")
                self.conv_operator_template(self.tik_instance.vec_conv, "none", src_line, ub_cast_line,
                                            [0, 0], self.split_input_width * self.input_channel0, 8, 4)
            else:
                self.data_move_pad(src_line, self.images_gm,
                               [0, (nc_idx * self.input_height + index_1) * self.input_line_size +
                               input_index_start * self.input_channel0],
                               self.split_input_width * self.input_channel0, 
                               self.images_gm_size, "gm2ub")
            temp_h_weight.set_as(self.workspace_h_weight_1[self.core_index * self.workspace_height + h_out_index])
            self.scalar_operator_template(self.tik_instance.vec_muls, self.temp_line_ub, src_line, temp_h_weight,
                                          [0, 0], self.split_input_width * self.input_channel0)

            if self.x_dtype == "float16":
                self.data_move_pad(ub_cast_line, self.images_gm,
                               [0, (nc_idx * self.input_height + index_2) * self.input_line_size +
                               input_index_start * self.input_channel0],
                               self.split_input_width * self.input_channel0, 
                               self.images_gm_size, "gm2ub")
                self.conv_operator_template(self.tik_instance.vec_conv, "none", src_line, ub_cast_line, [0, 0],
                                            self.split_input_width * self.input_channel0, 8, 4)
            else:
                self.data_move_pad(src_line, self.images_gm,
                               [0, (nc_idx * self.input_height + index_2) * self.input_line_size +
                               input_index_start * self.input_channel0],
                               self.split_input_width * self.input_channel0, 
                               self.images_gm_size, "gm2ub")
            temp_h_weight.set_as(self.workspace_h_weight_2[self.core_index * self.workspace_height + h_out_index])
            self.scalar_operator_template(self.tik_instance.vec_muls, src_line, src_line, temp_h_weight, [0, 0],
                                          self.split_input_width * self.input_channel0)
            self.double_operator_template(self.tik_instance.vec_add, self.temp_line_ub, self.temp_line_ub, src_line,
                                          [0, 0, 0], self.split_input_width * self.input_channel0)

            if self.x_dtype == "float16":
                self.data_move_pad(ub_cast_line, self.images_gm,
                               [0, (nc_idx * self.input_height + index_3) * self.input_line_size +
                               input_index_start * self.input_channel0],
                               self.split_input_width * self.input_channel0, 
                               self.images_gm_size, "gm2ub")
                self.conv_operator_template(self.tik_instance.vec_conv, "none", src_line, ub_cast_line,
                                            [0, 0], self.split_input_width * self.input_channel0, 8, 4)
            else:
                self.data_move_pad(src_line, self.images_gm,
                               [0, (nc_idx * self.input_height + index_3) * self.input_line_size +
                               input_index_start * self.input_channel0],
                               self.split_input_width * self.input_channel0, 
                               self.images_gm_size, "gm2ub")
            temp_h_weight.set_as(self.workspace_h_weight_3[self.core_index * self.workspace_height + h_out_index])
            self.scalar_operator_template(self.tik_instance.vec_muls, src_line, src_line, temp_h_weight,
                                          [0, 0], self.split_input_width * self.input_channel0)
            self.double_operator_template(self.tik_instance.vec_add, self.temp_line_ub, self.temp_line_ub, src_line,
                                          [0, 0, 0], self.split_input_width * self.input_channel0)

            if self.x_dtype == "float16":
                self.data_move_pad(ub_cast_line, self.images_gm,
                               [0, (nc_idx * self.input_height + index_4) * self.input_line_size +
                               input_index_start * self.input_channel0],
                               self.split_input_width * self.input_channel0, 
                               self.images_gm_size, 
                               "gm2ub")
                self.conv_operator_template(self.tik_instance.vec_conv, "none", src_line, ub_cast_line,
                                            [0, 0], self.split_input_width * self.input_channel0, 8, 4)
            else:
                self.data_move_pad(src_line, self.images_gm,
                               [0, (nc_idx * self.input_height + index_4) * self.input_line_size +
                               input_index_start * self.input_channel0],
                               self.split_input_width * self.input_channel0, 
                               self.images_gm_size, 
                               "gm2ub")
            temp_h_weight.set_as(self.workspace_h_weight_4[self.core_index * self.workspace_height + h_out_index])
            self.scalar_operator_template(self.tik_instance.vec_muls, src_line, src_line, temp_h_weight,
                                          [0, 0], self.split_input_width * self.input_channel0)
            self.double_operator_template(self.tik_instance.vec_add, self.temp_line_ub, self.temp_line_ub, src_line,
                                          [0, 0, 0], self.split_input_width * self.input_channel0)

    def calc_output_line(self, core_width_idx):
        """
        calc weighted average value in one line
        Parameters:
        ----------
        core_width_idx : int64.
            The splited width index of the operating slice
        """
        with self.tik_instance.new_stmt_scope():
            output_idx_start = self.tik_instance.Scalar("int64", "output_idx_start", init_value=256 * core_width_idx)
            input_idx_start = self.tik_instance.Scalar("int64", "input_idx_start",
                                                       init_value=self.split_input_gap * core_width_idx - 2)
            self.tik_instance.scalar_max(input_idx_start, self.min_index, input_idx_start)

            w_operator_ub = self.tik_instance.Tensor(self.inner_dtype, [self.split_output_width * self.input_channel0],
                                                    name="w_operator_ub", scope=tik.scope_ubuf)
            w_line_weight = self.tik_instance.Tensor(self.inner_dtype, [self.split_output_width * self.input_channel0],
                                                    name="w_line_weight", scope=tik.scope_ubuf)

            # obtain ll, l, r, rr item of whole line seperately
            w_in_index_ll = self.tik_instance.Scalar("int64", "w_in_index_ll")
            w_in_index_l = self.tik_instance.Scalar("int64", "w_in_index_l")
            w_in_index_r = self.tik_instance.Scalar("int64", "w_in_index_r")
            w_in_index_rr = self.tik_instance.Scalar("int64", "w_in_index_rr")
            scalar = self.tik_instance.Scalar(self.inner_dtype, "scalar")

            with self.tik_instance.for_range(0, self.oper_output_width) as w_out_index:
                w_in_index = self.tik_instance.Scalar("int64", "w_in_index")
                w_in_index.set_as(self.workspace_index_w_mapping[self.core_index * self.workspace_width
                                                                 + output_idx_start + w_out_index])
                w_in_index_ll.set_as(w_in_index - 1)
                self.tik_instance.scalar_max(w_in_index_ll, self.min_index, w_in_index_ll)
                self.data_move(w_operator_ub, self.temp_line_ub,
                               [w_out_index * self.input_channel0,
                               (w_in_index_ll - input_idx_start) * self.input_channel0],
                               self.input_channel0)
                scalar.set_as(self.workspace_w_weight_1[self.core_index * self.workspace_width
                                                        + output_idx_start + w_out_index])
                self.dup_value(w_line_weight, self.input_channel0, scalar, offset=w_out_index * self.input_channel0)

            self.double_operator_template(self.tik_instance.vec_mul, self.line_ub, w_operator_ub, w_line_weight,
                                            [0, 0, 0], self.oper_output_width * self.input_channel0)

            with self.tik_instance.for_range(0, self.oper_output_width) as w_out_index:
                w_in_index = self.tik_instance.Scalar("int64", "w_in_index")
                w_in_index.set_as(self.workspace_index_w_mapping[self.core_index * self.workspace_width
                                                                 + output_idx_start + w_out_index])
                w_in_index_l.set_as(w_in_index)
                self.tik_instance.scalar_max(w_in_index_l, self.min_index, w_in_index_l)
                self.data_move(w_operator_ub, self.temp_line_ub,
                               [w_out_index * self.input_channel0,
                               (w_in_index_l - input_idx_start) * self.input_channel0],
                               self.input_channel0)
                scalar.set_as(self.workspace_w_weight_2[self.core_index * self.workspace_width
                                                        + output_idx_start + w_out_index])
                self.dup_value(w_line_weight, self.input_channel0, scalar, offset=w_out_index * self.input_channel0)

            self.double_operator_template(self.tik_instance.vec_mul, w_operator_ub, w_operator_ub, w_line_weight,
                                            [0, 0, 0], self.oper_output_width * self.input_channel0)
            self.double_operator_template(self.tik_instance.vec_add, self.line_ub, self.line_ub, w_operator_ub,
                                            [0, 0, 0], self.oper_output_width * self.input_channel0)

            with self.tik_instance.for_range(0, self.oper_output_width) as w_out_index:
                w_in_index = self.tik_instance.Scalar("int64", "w_in_index")
                w_in_index.set_as(self.workspace_index_w_mapping[self.core_index * self.workspace_width
                                                                 + output_idx_start + w_out_index])
                w_in_index_r.set_as(w_in_index + 1)
                self.tik_instance.scalar_min(w_in_index_r, w_in_index_r, self.max_width)
                self.data_move(w_operator_ub, self.temp_line_ub,
                               [w_out_index * self.input_channel0,
                               (w_in_index_r - input_idx_start) * self.input_channel0],
                               self.input_channel0)
                scalar.set_as(self.workspace_w_weight_3[self.core_index * self.workspace_width
                                                        + output_idx_start + w_out_index])
                self.dup_value(w_line_weight, self.input_channel0, scalar, offset=w_out_index * self.input_channel0)

            self.double_operator_template(self.tik_instance.vec_mul, w_operator_ub, w_operator_ub, w_line_weight,
                                            [0, 0, 0], self.oper_output_width * self.input_channel0)
            self.double_operator_template(self.tik_instance.vec_add, self.line_ub, self.line_ub, w_operator_ub,
                                            [0, 0, 0], self.oper_output_width * self.input_channel0)

            with self.tik_instance.for_range(0, self.oper_output_width) as w_out_index:
                w_in_index = self.tik_instance.Scalar("int64", "w_in_index")
                w_in_index.set_as(self.workspace_index_w_mapping[self.core_index * self.workspace_width
                                                                 + output_idx_start + w_out_index])
                w_in_index_rr.set_as(w_in_index + 2)
                self.tik_instance.scalar_min(w_in_index_rr, w_in_index_rr, self.max_width)
                self.data_move(w_operator_ub, self.temp_line_ub,
                               [w_out_index * self.input_channel0,
                               (w_in_index_rr - input_idx_start) * self.input_channel0],
                               self.input_channel0)
                scalar.set_as(self.workspace_w_weight_4[self.core_index * self.workspace_width
                                                        + output_idx_start + w_out_index])
                self.dup_value(w_line_weight, self.input_channel0, scalar, offset=w_out_index * self.input_channel0)

            self.double_operator_template(self.tik_instance.vec_mul, w_operator_ub, w_operator_ub, w_line_weight,
                                            [0, 0, 0], self.oper_output_width * self.input_channel0)
            self.double_operator_template(self.tik_instance.vec_add, self.line_ub, self.line_ub, w_operator_ub,
                                            [0, 0, 0], self.oper_output_width * self.input_channel0)

    def conv_operator_template(self, op_obj, mode, dst, src, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template used to call vec_conv api functions
        Parameters:
        ----------
        op_obj : Function.
            The name of vec_conv
        mode : String.
            The mode of convet mode, such as floor, none
        dst : Tensor.
            The dest tensor used to store result
        src : Tensor.
            The source tensor used to load raw data
        offsets : Dict.
            The offsets of dest tensor and source tensor
        num : int64.
            The num of data attempted to conv
        """
        vector_mask_max = Constant.VECTOR_MASK_MAX
        dst_offset, src_offset = offsets

        tensor_size = src.size
        with self.tik_instance.if_scope(num > 0):
            tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, mode, dst[tmp_dst_offset], src[tmp_src_offset], 255,
                       dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, mode, dst[dst_offset], src[src_offset], repeat_time, dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)

    def scalar_operator_template(self, op_obj, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template used to call scalar binoculars api functions
        Parameters:
        ----------
        op_obj : Function.
            The name of vector api function
        dst : Tensor.
            The dest tensor used to store result
        src : Tensor.
            The source tensor used to load raw data
        scalar : Scalar.
            The Scalar item used to calculate
        offsets : Dict.
            The offsets of dest tensor and source tensor
        num : int64.
            The num of data attempted to operate
        """
        vector_mask_max = Constant.VECTOR_MASK_MAX
        dst_offset, src_offset = offsets

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)
        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset], src[src_offset], scalar, repeat_time, dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset], src[src_offset], scalar, 1, dst_stride, src_stride)

    def double_operator_template(self, op_obj, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8,
                                 src1_stride=8):
        """
        tik api template used to call trimembrane api functions
        Parameters:
        ----------
        op_obj : Function.
            The name of vector api function
        dst : Tensor.
            The dest tensor used to store result
        src0 : Tensor.
            The source tensor used to load left raw data
        src1 : Scalar.
            The source tensor used to load right raw data
        offsets : Dict.
            The offsets of dest tensor, source0 tensor and source1 tensor
        num : int64.
            The num of data attempted to operate
        """
        vector_mask_max = Constant.VECTOR_MASK_MAX
        dst_offset, src0_offset, src1_offset = offsets

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)
        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_time, dst_stride,
                   src0_stride, src1_stride)
            dst_offset += repeat_time * vector_mask_max
            src0_offset += repeat_time * vector_mask_max
            src1_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1, dst_stride, src0_stride,
                   src1_stride)

    def data_move_pad(self, dst, src, offsets, num, limit_size, trans):
        """
        pad move data from ub or gm to ub or gm
        Parameters:
        ----------
        dst : Tensor.
            The dest tensor used to store result data.
        src : Tensor.
            The source tensor used to load raw data
        offsets : Dict.
            The offsets of dest tensor and source tensor
        num : int64.
            The num of data attempted to operate
        limit_size: int64
            The operation element upper limit
        trans: str
            The data trans method
        """
        dst_offset, src_offset = offsets
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        if trans == "gm2ub":
            if tbe_platform.api_check_support("tik.data_move_pad"):
                with self.tik_instance.if_scope(src_offset + num - 1 > limit_size):
                    pad_length = limit_size - src_offset + 1
                    with self.tik_instance.if_scope(pad_length > 0):
                        self.tik_instance.data_move_pad(dst[dst_offset], src[src_offset], 
                                                        1, pad_length * dtype_byte_size, 0, 0)
                with self.tik_instance.else_scope():
                    self.data_move(dst, src, offsets, num)
            else:
                self.data_move(dst, src, offsets, num)
        else:
            if tbe_platform.api_check_support("tik.data_move_pad"):
                with self.tik_instance.if_scope(dst_offset + num - 1 > limit_size):
                    pad_length = limit_size - dst_offset + 1
                    with self.tik_instance.if_scope(pad_length > 0):
                        self.tik_instance.data_move_pad(dst[dst_offset], src[src_offset], 
                                                        1, pad_length * dtype_byte_size, 0, 0)
                with self.tik_instance.else_scope():
                    self.data_move(dst, src, offsets, num)
            else:    
                self.data_move(dst, src, offsets, num)

    def data_move(self, dst, src, offsets, num, nburst=1, src_stride=0, dst_stride=0):
        """
        move data from ub or gm to ub or gm
        Parameters:
        ----------
        dst : Tensor.
            The dest tensor used to store result data.
        src : Tensor.
            The source tensor used to load raw data
        src1 : Scalar.
            The source tensor used to load right raw data
        offsets : Dict.
            The offsets of dest tensor and source tensor
        num : int64.
            The num of data attempted to operate
        """
        dst_offset, src_offset = offsets
        sid = 0
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_bite_size // dtype_byte_size
        burst_len = (num + data_each_block - 1) // data_each_block
        self.tik_instance.data_move(dst[dst_offset],
                                    src[src_offset],
                                    sid,
                                    nburst,
                                    burst_len,
                                    src_stride=src_stride,
                                    dst_stride=dst_stride)

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        tik api template used to dup value in ub
        Parameters:
        ----------
        dst : Tensor.
            The dest tensor used to store result
        num : int64.
            The num of data attempted to dup
        dup_value : Scalar.
            The expected data attempted to fill dest tensor
        offset : int64.
            The offset of dest tensor
        """
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        stride = 8

        loop = num // (mask * 255)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
            offset += loop * mask * 255

        repeat_time = (num % (mask * 255)) // mask
        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
            offset += repeat_time * mask

        last_num = num % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)