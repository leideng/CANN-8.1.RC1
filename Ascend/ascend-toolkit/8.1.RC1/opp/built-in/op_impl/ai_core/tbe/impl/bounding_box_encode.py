#!/usr/bin/python
# -*- coding: utf-8 -*-
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
bounding_box_encode
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# ub size for slect max dim value
SELECT_UB_SIZE = 196608


# 'pylint: disable = unused-argument
# 'pylint: disable=too-many-arguments,singleton-comparison
def get_op_support_info(anchorbox_in_dict,
                        ground_truth_in_dict,
                        delta_out_dict,
                        means_attrs=(0, 0, 0, 0),
                        stds_attrs=(1, 1, 1, 1),
                        kernel_name_val="bounding_box_encode"):
    """
    get_op_support_info
    """
    axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]], [1, [0], [-1], [-1]]), SplitOutput([0, [0]])]]
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=too-many-instance-attributes
class BoundingBoxEncode():
    """
    Funtion: use to store BoundingBoxEncode base parameters
    """
    # 'pylint: disable=too-many-arguments
    def __init__(self, anchorbox, ground_truth_box, delta, means, stds,
                 kernel_name):
        self.init_tik_instance()
        self.anchor_box_shape = anchorbox.get("shape")
        self.anchor_box_dtype = anchorbox.get("dtype").lower()
        self.ground_truth_shape = ground_truth_box.get("shape")
        self.ground_truth_dtype = ground_truth_box.get("dtype").lower()
        self.delta_shape = delta.get("shape")
        self.delta_shape = delta.get("dtype").lower()
        self.means = means
        self.stds = stds
        self.kernel_name = kernel_name
        self.core_num = 32
        if tbe_platform.api_check_support("tik.vcopy"):
            self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
            if self.core_num == 50:
                self.core_num = 48
        self.data_dtype_bytes_size = tbe_platform.get_bit_len(self.anchor_box_dtype) // 8
        block_size = 32
        self.data_num_in_each_block = block_size // self.data_dtype_bytes_size
        self.each_core_start_addr, self.each_core_calcul_num = self.get_core_param()
        ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        max_ub_element_number_fp16 = 4096 if ub_size == SELECT_UB_SIZE else 5120
        self.ub_max_size = max_ub_element_number_fp16
        self.init_gm_tensor()
        max_ub_element_number_fp32 = 2560
        if self.anchor_box_dtype == "float32":
            self.ub_max_size = max_ub_element_number_fp32
        self.loop_cycle = self.get_loop_cycle()
        self.start_block_addr, self.block_number = self.get_loop_param()
        self.repeat_times = self.get_repeat_cycle()

    def init_tik_instance(self):
        """
        init_tik_instance

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.support_div = tbe_platform.api_check_support(
            "tik.vdiv", "float32")

    def data_move_mte2_function(self, loop_input, block_number):
        """
        data_move_mte2_function

        Parameters
        ----------
        loop_input : int
            loop index
        block_number: int
            block_number

        Returns
        -------
        result : list
            [anchor_box_ub, ground_truth_in_ub]
        """
        default_nburst = 1
        anchor_box_ub = self.tik_instance.Tensor(
            self.anchor_box_dtype, (self.ub_max_size // 4, 4),
            name="anchor_box_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.data_move(anchor_box_ub,
                                    self.anchorbox_in[loop_input], 0,
                                    default_nburst, block_number, 1, 1, 1)
        ground_truth_in_ub = self.tik_instance.Tensor(
            self.ground_truth_dtype, (self.ub_max_size // 4, 4),
            name="ground_truth_in_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.data_move(ground_truth_in_ub,
                                    self.ground_truth_in[loop_input], 0,
                                    default_nburst, block_number, 1, 1, 1)
        return anchor_box_ub, ground_truth_in_ub

    def data_move_mte3_function(self, loop_input, block_num, delta_dst_ub):
        """
        data_move_mte3_function

        Parameters
        ----------
        loop_input : int
            loop index
        block_num: int
            block_number
        delta_dst_ub : addr
            delta_dst_ub

        Returns
        -------
        None
        """
        default_nburst = 1
        self.tik_instance.data_move(self.delta_out[loop_input], delta_dst_ub,
                                    0, default_nburst, block_num, 0, 0)

    def get_repeat_cycle(self):
        """
        data_move_mte2_function

        Parameters
        ----------
        None

        Returns
        -------
        result : int
            repeat_times
        """
        block_number_fp16 = 32
        block_number_fp32 = 64
        each_repeat_block_number = block_number_fp16
        if self.anchor_box_dtype == "float32":
            each_repeat_block_number = block_number_fp32
        if self.block_number < each_repeat_block_number:
            repeat_times = 1
        elif self.block_number % each_repeat_block_number == 0:
            repeat_times = self.block_number // each_repeat_block_number
        else:
            repeat_times = self.block_number // each_repeat_block_number + 1
        return repeat_times

    def get_core_param(self):
        """
        calculate data in number, each core start address
        """
        data_in_number = self.anchor_box_shape[0] * self.anchor_box_shape[1]
        each_core_start_addr = (data_in_number // (self.core_num * 4)) * 4

        # check input data number can equal divivde to (32 core * 4 point)
        if data_in_number % (self.core_num * 4) == 0:
            # check input data number is equal to block
            if each_core_start_addr % self.data_num_in_each_block == 0:
                each_core_calcul_num = each_core_start_addr
            else:
                each_core_calcul_num = (
                    each_core_start_addr // self.data_num_in_each_block + 1
                ) * self.data_num_in_each_block
        else:
            each_core_calcul_num = data_in_number - each_core_start_addr * (
                self.core_num - 1)
            if each_core_calcul_num % self.data_num_in_each_block != 0:
                each_core_calcul_num = (
                    each_core_calcul_num // self.data_num_in_each_block + 1
                ) * self.data_num_in_each_block
        return each_core_start_addr, each_core_calcul_num

    def set_means_stds_scalar(self, means, stds):
        """
        set_means_stds_scalar
        """
        dtype = "float16"
        # set means value [0, 0, 0, 0]
        means_0_scalar = self.tik_instance.Scalar(dtype, name="means_0_scalar")
        means_0_scalar.set_as(-means[0])
        means_1_scalar = self.tik_instance.Scalar(dtype, name="means_1_scalar")
        means_1_scalar.set_as(-means[1])
        means_2_scalar = self.tik_instance.Scalar(dtype, name="means_2_scalar")
        means_2_scalar.set_as(-means[2])
        means_3_scalar = self.tik_instance.Scalar(dtype, name="means_3_scalar")
        means_3_scalar.set_as(-means[3])

        # set stds value [1, 1, 1, 1]
        stds_0_scalar = self.tik_instance.Scalar(dtype, name="stds_0_scalar")
        stds_0_scalar.set_as(1.0 / stds[0])
        stds_1_scalar = self.tik_instance.Scalar(dtype, name="stds_1_scalar")
        stds_1_scalar.set_as(1.0 / stds[1])
        stds_2_scalar = self.tik_instance.Scalar(dtype, name="stds_2_scalar")
        stds_2_scalar.set_as(1.0 / stds[2])
        stds_3_scalar = self.tik_instance.Scalar(dtype, name="stds_3_scalar")
        stds_3_scalar.set_as(1.0 / stds[3])

        scalar_list = [means_0_scalar, means_1_scalar, means_2_scalar, means_3_scalar,
                       stds_0_scalar, stds_1_scalar, stds_2_scalar, stds_3_scalar]
        return scalar_list

    def tik_instance_function(self):
        """
        tik_instance_function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            self.calculation_process(block_id)
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.anchorbox_in, self.ground_truth_in],
            outputs=[self.delta_out])

    def init_gm_tensor(self):
        """
        init_gm_tensor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        gm_shape_size = self.each_core_start_addr * (
            self.core_num - 1) + self.each_core_calcul_num
        self.anchorbox_in = self.tik_instance.Tensor(
            self.anchor_box_dtype, (gm_shape_size // 4, 4),
            name="anchorbox_in",
            scope=tik.scope_gm)
        self.ground_truth_in = self.tik_instance.Tensor(
            self.anchor_box_dtype, (gm_shape_size // 4, 4),
            name="ground_truth_in",
            scope=tik.scope_gm)
        self.delta_out = self.tik_instance.Tensor(
            self.anchor_box_dtype, (gm_shape_size // 4, 4),
            name="delta_out",
            scope=tik.scope_gm)

    def get_loop_cycle(self):
        """
        get_loop_cycle

        Parameters
        ----------
        None

        Returns
        -------
        result : int
            loop_cycle
        """
        if self.each_core_calcul_num % self.ub_max_size == 0:
            loop_cycle = int(self.each_core_calcul_num // self.ub_max_size)
        else:
            loop_cycle = int(self.each_core_calcul_num // self.ub_max_size + 1)

        return loop_cycle

    def get_loop_param(self):
        """
        get_loop_param

        Parameters
        ----------
        None

        Returns
        -------
        result : list
            [start_block_addr, block_number]
        """
        block_number = self.each_core_calcul_num // self.data_num_in_each_block
        if block_number == 0:
            block_number = 1
        start_block_addr = block_number // self.loop_cycle
        max_ub_element_number_fp16 = 5120
        if self.loop_cycle > 1:
            if block_number % self.loop_cycle != 0:
                block_number_loop = block_number - start_block_addr * (
                    self.loop_cycle - 1)
                while block_number * self.loop_cycle < block_number_loop or \
                      block_number_loop * 16 > max_ub_element_number_fp16:
                    self.loop_cycle += 1
                    start_block_addr = block_number // self.loop_cycle
                    block_number_loop = block_number - start_block_addr * (
                        self.loop_cycle - 1)
                block_number = block_number_loop
            else:
                block_number = start_block_addr
        return start_block_addr, block_number

    def calculation_process(self, block_id):
        """
        get_loop_param

        Parameters
        ----------
        block_id : int
            block_id

        Returns
        -------
        None
        """
        scalar_list = self.set_means_stds_scalar(self.means, self.stds)
        if self.loop_cycle == 1:
            loop_input = block_id * self.each_core_start_addr
            anchorbox_src_ub, groundtruthbox_src_ub = \
                self.data_move_mte2_function(loop_input, self.block_number)
            delta_dst_ub = self.bounding_box_encode_compute(
                scalar_list, self.repeat_times, anchorbox_src_ub,
                groundtruthbox_src_ub)
            self.data_move_mte3_function(loop_input, self.block_number,
                                         delta_dst_ub)
        else:
            loop_input = block_id * self.each_core_start_addr
            thread_num_value = 2
            with self.tik_instance.for_range(0, self.loop_cycle, thread_num=thread_num_value) as cycle:
                loop_input = loop_input + cycle * self.start_block_addr * self.data_num_in_each_block
                anchorbox_src_ub, groundtruthbox_src_ub = \
                    self.data_move_mte2_function(loop_input, self.block_number)
                delta_dst_ub = self.bounding_box_encode_compute(
                    scalar_list, self.repeat_times, anchorbox_src_ub,
                    groundtruthbox_src_ub)
                self.data_move_mte3_function(loop_input, self.block_number,
                                             delta_dst_ub)

    # 'pylint: disable=too-many-locals,too-many-statements,too-many-branches
    def bounding_box_encode_compute(self, scalar_list, repeat_times,
                                    anchorbox_src_ub, groundtruthbox_src_ub):
        """
        use tik instruction to calculate result bounding_box_encode_compute

        Parameters
        ----------
        scalar_list : list
            block_id
        repeat_times : int
            repeat_times
        anchorbox_src_ub : TVM tensor
            anchorbox_src_ub
        groundtruthbox_src_ub : TVM tensor
            groundtruthbox_src_ub

        Returns
        -------
        delta_out_ub : TVM tensor
        """
        anchorbox_dst_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="anchorbox_dst_ub",
            scope=tik.scope_ubuf)
        groundtruthbox_dst_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="groundtruthbox_dst_ub",
            scope=tik.scope_ubuf)

        # convert float32 to float16
        anchorbox_vconv_src_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="anchorbox_vconv_src_ub",
            scope=tik.scope_ubuf)
        groundtruthbox_vconv_src_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="groundtruthbox_vconv_src_ub",
            scope=tik.scope_ubuf)

        if self.anchor_box_dtype == "float32":
            self.tik_instance.vconv(64, 'none', anchorbox_vconv_src_ub,
                                    anchorbox_src_ub, repeat_times * 8,
                                    1, 1, 4, 8)
            self.tik_instance.vconv(64, 'none', groundtruthbox_vconv_src_ub,
                                    groundtruthbox_src_ub, repeat_times * 8, 1,
                                    1, 4, 8)
        else:
            anchorbox_vconv_src_ub = anchorbox_src_ub
            groundtruthbox_vconv_src_ub = groundtruthbox_src_ub

        # transverse input data use vnchwconv instruction
        anchorbox_src_list = [anchorbox_vconv_src_ub[16 * i]
                              for i in range(16)]
        anchorbox_dst_list = [anchorbox_dst_ub[16 * i] for i in range(16)]

        groundtruthbox_src_list = [groundtruthbox_vconv_src_ub[16 * i] for i in range(16)]
        groundtruthbox_dst_list = [groundtruthbox_dst_ub[16 * i] for i in range(16)]

        anchorbox_ptmp_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="anchorbox_ptmp_ub",
            scope=tik.scope_ubuf)
        groundtruthbox_ptmp_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="groundtruthbox_ptmp_ub",
            scope=tik.scope_ubuf)

        # transform anchorbox and groundtruth box
        self.tik_instance.vnchwconv(True, True, anchorbox_dst_list,
                                    anchorbox_src_list, repeat_times * 2, 16,
                                    16)
        self.tik_instance.vnchwconv(True, True, groundtruthbox_dst_list,
                                    groundtruthbox_src_list, repeat_times * 2,
                                    16, 16)

        # Calculate px, py, pw, ph
        anchorbox_dst_ub16 = anchorbox_dst_ub[16]
        anchorbox_dst_ub32 = anchorbox_dst_ub[32]
        anchorbox_dst_ub48 = anchorbox_dst_ub[48]
        anchorbox_ptmp_ub16 = anchorbox_ptmp_ub[16]
        anchorbox_ptmp_ub32 = anchorbox_ptmp_ub[32]
        anchorbox_ptmp_ub48 = anchorbox_ptmp_ub[48]
        groundtruthbox_ptmp_ub16 = groundtruthbox_ptmp_ub[16]
        groundtruthbox_ptmp_ub32 = groundtruthbox_ptmp_ub[32]
        groundtruthbox_ptmp_ub48 = groundtruthbox_ptmp_ub[48]
        groundtruthbox_dst_ub16 = groundtruthbox_dst_ub[16]
        groundtruthbox_dst_ub32 = groundtruthbox_dst_ub[32]
        groundtruthbox_dst_ub48 = groundtruthbox_dst_ub[48]
        self.tik_instance.vadd(128, anchorbox_ptmp_ub,
                               anchorbox_dst_ub,
                               anchorbox_dst_ub32, repeat_times, 4, 4, 4, 32,
                               32, 32)
        self.tik_instance.vmuls(128, anchorbox_ptmp_ub,
                                anchorbox_ptmp_ub,
                                0.5, repeat_times, 4, 4, 32, 32)

        self.tik_instance.vadd(128, anchorbox_ptmp_ub16,
                               anchorbox_dst_ub16,
                               anchorbox_dst_ub48, repeat_times, 4, 4, 4, 32,
                               32, 32)
        self.tik_instance.vmuls(128, anchorbox_ptmp_ub16,
                                anchorbox_ptmp_ub16, 0.5, repeat_times, 4, 4,
                                32, 32)

        self.tik_instance.vsub(128, anchorbox_ptmp_ub32,
                               anchorbox_dst_ub32,
                               anchorbox_dst_ub, repeat_times, 4, 4, 4, 32,
                               32, 32)
        self.tik_instance.vadds(128, anchorbox_ptmp_ub32,
                                anchorbox_ptmp_ub32, 1, repeat_times, 4, 4,
                                32, 32)

        if self.support_div == False:
            rec_1 = groundtruthbox_ptmp_ub32
            rec_2 = groundtruthbox_ptmp_ub48
            self.tik_instance.vrec(128, rec_1, anchorbox_ptmp_ub32,
                                   repeat_times,
                                   4, 4, 32, 32)
            self.tik_instance.vmul(128, rec_2,
                                   rec_1,
                                   anchorbox_ptmp_ub32, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmuls(128, rec_2,
                                    rec_2, -1, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vadds(128, rec_2,
                                    rec_2, 2, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vmul(128, rec_2,
                                   rec_2,
                                   rec_1, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmul(128, rec_1,
                                   rec_2,
                                   anchorbox_ptmp_ub32, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmuls(128, rec_1,
                                    rec_1, -1, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vadds(128, rec_1,
                                    rec_1, 2, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vmul(128, anchorbox_ptmp_ub32,
                                   rec_1,
                                   rec_2, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)

        self.tik_instance.vsub(128, anchorbox_ptmp_ub48,
                               anchorbox_dst_ub48,
                               anchorbox_dst_ub16, repeat_times, 4, 4, 4, 32,
                               32, 32)
        self.tik_instance.vadds(128, anchorbox_ptmp_ub48,
                                anchorbox_ptmp_ub48, 1, repeat_times, 4, 4,
                                32, 32)
        if self.support_div == False:
            rec_1 = groundtruthbox_ptmp_ub32
            rec_2 = groundtruthbox_ptmp_ub48
            self.tik_instance.vrec(128, rec_1, anchorbox_ptmp_ub48,
                                   repeat_times,
                                   4, 4, 32, 32)
            self.tik_instance.vmul(128, rec_2,
                                   rec_1,
                                   anchorbox_ptmp_ub48, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmuls(128, rec_2,
                                    rec_2, -1, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vadds(128, rec_2,
                                    rec_2, 2, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vmul(128, rec_2,
                                   rec_2,
                                   rec_1, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmul(128, rec_1,
                                   rec_2,
                                   anchorbox_ptmp_ub48, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)
            self.tik_instance.vmuls(128, rec_1,
                                    rec_1, -1, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vadds(128, rec_1,
                                    rec_1, 2, repeat_times, 4, 4,
                                    32, 32)
            self.tik_instance.vmul(128, anchorbox_ptmp_ub48,
                                   rec_1,
                                   rec_2, repeat_times,
                                   4, 4, 4, 32,
                                   32, 32)

        # Calculate gx, gy, gw, gh
        self.tik_instance.vadd(
            128, groundtruthbox_ptmp_ub, groundtruthbox_dst_ub,
            groundtruthbox_dst_ub32, repeat_times, 4, 4, 4, 32, 32, 32)
        self.tik_instance.vmuls(128, groundtruthbox_ptmp_ub,
                                groundtruthbox_ptmp_ub, 0.5, repeat_times,
                                4, 4, 32, 32)

        self.tik_instance.vadd(
            128, groundtruthbox_ptmp_ub16, groundtruthbox_dst_ub16,
            groundtruthbox_dst_ub48, repeat_times, 4, 4, 4, 32, 32, 32)
        self.tik_instance.vmuls(128, groundtruthbox_ptmp_ub16,
                                groundtruthbox_ptmp_ub16, 0.5, repeat_times,
                                4, 4, 32, 32)

        self.tik_instance.vsub(
            128, groundtruthbox_ptmp_ub32, groundtruthbox_dst_ub32,
            groundtruthbox_dst_ub, repeat_times, 4, 4, 4, 32, 32, 32)
        self.tik_instance.vadds(128, groundtruthbox_ptmp_ub32,
                                groundtruthbox_ptmp_ub32, 1, repeat_times, 4,
                                4, 32, 32)

        self.tik_instance.vsub(
            128, groundtruthbox_ptmp_ub48, groundtruthbox_dst_ub48,
            groundtruthbox_dst_ub16, repeat_times, 4, 4, 4, 32, 32, 32)
        self.tik_instance.vadds(128, groundtruthbox_ptmp_ub48,
                                groundtruthbox_ptmp_ub48, 1, repeat_times, 4,
                                4, 32, 32)

        # Calculate dx, dy, dw, dh
        delta_tmp_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="delta_tmp_ub",
            scope=tik.scope_ubuf)

        self.tik_instance.vsub(128, delta_tmp_ub, groundtruthbox_ptmp_ub,
                               anchorbox_ptmp_ub, repeat_times, 4, 4, 4, 32,
                               32, 32)

        if self.support_div == True:
            self.tik_instance.vdiv(128, delta_tmp_ub, delta_tmp_ub,
                                   anchorbox_ptmp_ub32, repeat_times,
                                   4, 4, 4, 32, 32, 32)
        else:
            self.tik_instance.vmul(128, delta_tmp_ub, delta_tmp_ub,
                                   anchorbox_ptmp_ub32, repeat_times,
                                   4, 4, 4, 32, 32, 32)
        self.tik_instance.vadds(128, delta_tmp_ub, delta_tmp_ub,
                                scalar_list[0], repeat_times, 4, 4, 32, 32)
        self.tik_instance.vmuls(128, delta_tmp_ub, delta_tmp_ub,
                                scalar_list[4], repeat_times, 4, 4, 32, 32)

        # 'dy = ( (gy - py)/ph + (-means[1]) * (1/stds[1])
        delta_tmp_ub16 = delta_tmp_ub[16]
        self.tik_instance.vsub(
            128, delta_tmp_ub16, groundtruthbox_ptmp_ub16,
            anchorbox_ptmp_ub16, repeat_times, 4, 4, 4, 32, 32, 32)

        if self.support_div == True:
            self.tik_instance.vdiv(128, delta_tmp_ub16, delta_tmp_ub16,
                                   anchorbox_ptmp_ub48, repeat_times,
                                   4, 4, 4, 32, 32, 32)
        else:
            self.tik_instance.vmul(128, delta_tmp_ub16, delta_tmp_ub16,
                                   anchorbox_ptmp_ub48, repeat_times,
                                   4, 4, 4, 32, 32, 32)

        self.tik_instance.vadds(128, delta_tmp_ub16, delta_tmp_ub16,
                                scalar_list[1], repeat_times, 4, 4, 32, 32)
        self.tik_instance.vmuls(128, delta_tmp_ub16, delta_tmp_ub16,
                                scalar_list[5], repeat_times, 4, 4, 32, 32)

        # 'dw = ( log(gw/pw) + (-means[2]) * (1/stds[2])
        delta_tmp_ub32 = delta_tmp_ub[32]
        if self.support_div == True:
            self.tik_instance.vdiv(
                128, delta_tmp_ub32, groundtruthbox_ptmp_ub32,
                anchorbox_ptmp_ub32, repeat_times, 4, 4, 4, 32, 32, 32)
        else:
            self.tik_instance.vmul(
                128, delta_tmp_ub32, groundtruthbox_ptmp_ub32,
                anchorbox_ptmp_ub32, repeat_times, 4, 4, 4, 32, 32, 32)

        self.tik_instance.vln(128, delta_tmp_ub32, delta_tmp_ub32,
                              repeat_times, 4, 4, 32, 32)
        self.tik_instance.vadds(128, delta_tmp_ub32, delta_tmp_ub32,
                                scalar_list[2], repeat_times, 4, 4, 32, 32)
        self.tik_instance.vmuls(128, delta_tmp_ub32, delta_tmp_ub32,
                                scalar_list[6], repeat_times, 4, 4, 32, 32)

        # 'dy = ( log(gh/ph) + (-means[3]) * (1/stds[3])
        delta_tmp_ub48 = delta_tmp_ub[48]
        if self.support_div == True:
            self.tik_instance.vdiv(
                128, delta_tmp_ub48, groundtruthbox_ptmp_ub48,
                anchorbox_ptmp_ub48, repeat_times, 4, 4, 4, 32, 32, 32)
        else:
            self.tik_instance.vmul(
                128, delta_tmp_ub48, groundtruthbox_ptmp_ub48,
                anchorbox_ptmp_ub48, repeat_times, 4, 4, 4, 32, 32, 32)

        self.tik_instance.vln(128, delta_tmp_ub48, delta_tmp_ub48,
                              repeat_times, 4, 4, 32, 32)
        self.tik_instance.vadds(128, delta_tmp_ub48, delta_tmp_ub48,
                                scalar_list[3], repeat_times, 4, 4, 32, 32)
        self.tik_instance.vmuls(128, delta_tmp_ub48, delta_tmp_ub48,
                                scalar_list[7], repeat_times, 4, 4, 32, 32)

        # transverse output data back
        delta_out_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="delta_out_ub",
            scope=tik.scope_ubuf)
        delta_out_fp32_ub = self.tik_instance.Tensor(
            "float32", (self.ub_max_size,),
            name="delta_out_fp32_ub",
            scope=tik.scope_ubuf)

        delta_tmp_list = [delta_tmp_ub[16 * i] for i in range(16)]
        delta_out_list = [delta_out_ub[16 * i] for i in range(16)]

        self.tik_instance.vnchwconv(True, True, delta_out_list, delta_tmp_list,
                                    repeat_times * 2, 16, 16)

        if self.anchor_box_dtype == "float32":
            self.tik_instance.vconv(64, 'none', delta_out_fp32_ub,
                                    delta_out_ub, repeat_times * 8, 1, 1, 8, 4)
            delta_out_ub = delta_out_fp32_ub
        return delta_out_ub


# 'pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_FLOAT, para_check.OPTION_ATTR_LIST_FLOAT,
                            para_check.KERNEL_NAME)
def bounding_box_encode(anchorbox_in_dict,
                        ground_truth_in_dict,
                        delta_out_dict,
                        means_attrs=(0, 0, 0, 0),
                        stds_attrs=(1, 1, 1, 1),
                        kernel_name="bounding_box_encode"):
    """
    algorithm: bounding_box_encode

    Parameters
    ----------
    anchorbox_in_dict : dict
        shape and dtype of input
    ground_truth_in_dict : dict
        shape and dtype of input
    delta_out_dict : dict
        shape and dtype of output, should be same shape and type as input
    means_attrs : list
        shape and dtype of output, should be same shape and type as input
    stds_attrs : list
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "bounding_box_encode"

    Returns
    -------
    None
    """
    anchor_box_shape = anchorbox_in_dict.get("shape")
    ground_truth_box_shape = ground_truth_in_dict.get("shape")

    para_check.check_shape(anchor_box_shape, param_name="anchorbox_in_dict")

    para_check.check_shape(ground_truth_box_shape, param_name="ground_truth_in_dict")

    bounding_box_encode_ = BoundingBoxEncode(
        anchorbox_in_dict, ground_truth_in_dict, delta_out_dict, means_attrs,
        stds_attrs, kernel_name)

    bounding_box_encode_.tik_instance_function()

    return bounding_box_encode_.tik_instance
