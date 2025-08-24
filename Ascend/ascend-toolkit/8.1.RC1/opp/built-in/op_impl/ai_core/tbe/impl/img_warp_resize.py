#!/usr/bin/python
# -*- coding: utf-8 -*-
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
img_warp_resize.py
"""
from impl.util.util_common import div_align_scalar as div_align
from impl.util.util_common import ceil_div_scalar as ceil_div
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util import util_tik_comm_func


# 'pylint: disable=too-many-instance-attributes,too-many-locals
class Resization:
    """
    Function: store Resization params and compute resize
    """

    def __init__(self, img, warp_index, warp_img, kernel_name):
        self.tik_instance = tik.Tik()
        self.x_dtype = img.get("dtype")
        self.x_shape = list(img.get("shape"))
        self.y_dtype = warp_index.get("dtype")
        self.y_shape = list(warp_index.get("shape"))
        self.z_dtype = warp_img.get("dtype")
        self.z_shape = list(warp_img.get("shape"))
        para_check.check_dtype(self.x_dtype, {"float16", "float32"}, param_name="img")
        para_check.check_dtype(self.y_dtype, {"float32"}, param_name="warp_index")
        para_check.check_shape_rule(self.x_shape, min_dim=5, max_dim=5)
        para_check.check_shape_rule(self.y_shape, min_dim=4, max_dim=4)
        self.image_batch, _, self.image_channel, self.image_height, self.image_width = self.x_shape
        self.image_area = self.image_height * self.image_width
        self.concat_x_shape = self.x_shape[0:3] + [self.image_area]
        self.concat_y_shape = self.y_shape[0:2] + [self.image_area]
        self.concat_z_shape = self.z_shape[0:2] + [self.image_area]
        self.x_gm = self.tik_instance.Tensor(self.x_dtype, self.concat_x_shape, name="img_gm", scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.y_dtype, self.concat_y_shape,
                                             name="warp_index_gm", scope=tik.scope_gm)
        self.z_gm = self.tik_instance.Tensor(self.z_dtype, self.concat_z_shape,
                                             name="warp_img_gm", scope=tik.scope_gm)
        self.kernel_name = kernel_name
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        # calcu with float32 inner
        self.inner_block_num = 8
        self.outer_block_num = 16 if self.x_dtype in ("float16",) else 8

    def do_resization(self):
        """
        do_resization
        """
        cut_process_num_minest = 256
        cut_core_num = ceil_div(self.image_area, self.ai_core_num)
        cut_core_num = max(cut_core_num, cut_process_num_minest)
        cut_core_num = div_align(cut_core_num, self.outer_block_num)
        used_core_num = ceil_div(self.image_area, cut_core_num)
        cut_last_core_num = self.image_area - cut_core_num * (used_core_num - 1)

        def _run_one_core(core_area_start, core_area_num):
            """
            core_area_start
            core_area_num
            core_batch_start
            """
            with self.tik_instance.for_range(0, self.image_batch) as batch_idx:
                process_num = 4096
                process_loop = core_area_num // process_num
                process_tail = core_area_num % process_num
                with self.tik_instance.for_range(0, process_loop) as area_loop_idx:
                    new_core_area_start = core_area_start + area_loop_idx * process_num
                    self.compute_bilinear(batch_idx, new_core_area_start, process_num)

                if process_tail != 0:
                    new_core_area_start = core_area_start + process_loop * process_num
                    self.compute_bilinear(batch_idx, new_core_area_start, process_tail)

        with self.tik_instance.for_range(0, used_core_num, block_num=used_core_num) as core_index:
            with self.tik_instance.if_scope(core_index < used_core_num - 1):
                area_start = core_index * cut_core_num
                area_num = cut_core_num
                _run_one_core(area_start, area_num)
            with self.tik_instance.else_scope():
                last_offset = 0
                if cut_last_core_num % self.outer_block_num != 0 and used_core_num > 1:
                    last_offset = self.outer_block_num - (cut_last_core_num % self.outer_block_num)
                    cut_last_core_num = \
                        (cut_last_core_num + self.outer_block_num - 1) // self.outer_block_num * self.outer_block_num
                area_start = core_index * cut_core_num - last_offset
                area_num = cut_last_core_num
                _run_one_core(area_start, area_num)

    # 'pylint: disable=too-many-statements
    def compute_bilinear(self, batch_idx, core_area_start, core_area_num):
        """
        compute_bilinear
        """
        apply_ub_mem = div_align(core_area_num, self.inner_block_num)
        copy_y_burst_len = ceil_div(core_area_num, self.inner_block_num)
        copy_x_burst_len = ceil_div(core_area_num, self.outer_block_num)
        # calcu x_lerp
        x_ub = self.tik_instance.Tensor(self.y_dtype, (apply_ub_mem,), name="x_ub", scope=tik.scope_ubuf)
        tmp_x_ub = self.tik_instance.Tensor("int32", (apply_ub_mem,), name="tmp_x_ub", scope=tik.scope_ubuf)
        conv_x_ub = self.tik_instance.Tensor(self.y_dtype, (apply_ub_mem,), name="conv_x_ub", scope=tik.scope_ubuf)
        y_ub = self.tik_instance.Tensor(self.y_dtype, (apply_ub_mem,), name="y_ub", scope=tik.scope_ubuf)
        x_lerp_ub = x_ub
        self.tik_instance.data_move(x_ub, self.y_gm[batch_idx, 0, core_area_start:], 0, 1, copy_y_burst_len, 0, 0)
        util_tik_comm_func.tik_func_vconv(self.tik_instance, tmp_x_ub, x_ub, core_area_num, mode="floor")
        util_tik_comm_func.tik_func_vconv(self.tik_instance, conv_x_ub, tmp_x_ub, core_area_num)
        util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vsub", x_lerp_ub, x_ub, conv_x_ub, core_area_num)

        # calcu y_lerp
        self.tik_instance.data_move(y_ub, self.y_gm[batch_idx, 1, core_area_start:], 0, 1, copy_y_burst_len, 0, 0)
        tmp_y_ub = tmp_x_ub
        conv_y_ub = conv_x_ub
        y_lerp_ub = y_ub
        util_tik_comm_func.tik_func_vconv(self.tik_instance, tmp_y_ub, y_ub, core_area_num, mode="floor")
        util_tik_comm_func.tik_func_vconv(self.tik_instance, conv_y_ub, tmp_y_ub, core_area_num)
        util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vsub", y_lerp_ub, y_ub, conv_y_ub, core_area_num)

        # add loop for c
        with self.tik_instance.for_range(0, self.image_channel) as idx:
            # top_left
            top_left_ub = self.tik_instance.Tensor(self.y_dtype, (apply_ub_mem,),
                                                   name="top_left_ub",
                                                   scope=tik.scope_ubuf)
            top_right_ub = self.tik_instance.Tensor(self.y_dtype, (apply_ub_mem,),
                                                    name="top_right_ub",
                                                    scope=tik.scope_ubuf)
            bottom_left_ub = self.tik_instance.Tensor(self.y_dtype, (apply_ub_mem,),
                                                      name="bottom_left_ub",
                                                      scope=tik.scope_ubuf)
            bottom_right_ub = self.tik_instance.Tensor(self.y_dtype, (apply_ub_mem,),
                                                       name="bottom_right_ub",
                                                       scope=tik.scope_ubuf)
            top_ub = self.tik_instance.Tensor(self.y_dtype, (apply_ub_mem,), name="top_ub", scope=tik.scope_ubuf)
            bottom_ub = self.tik_instance.Tensor(self.y_dtype, (apply_ub_mem,),
                                                 name="bottom_ub", scope=tik.scope_ubuf)
            if self.x_dtype != self.y_dtype:
                data_move_ping_ub = self.tik_instance.Tensor(self.x_dtype, (apply_ub_mem,),
                                                             name="data_move_ping_ub",
                                                             scope=tik.scope_ubuf)
                data_move_pang_ub = self.tik_instance.Tensor(self.x_dtype, (apply_ub_mem,),
                                                             name="data_move_pang_ub",
                                                             scope=tik.scope_ubuf)
                # top_left
                self.tik_instance.data_move(data_move_ping_ub, self.x_gm[batch_idx, 0, idx, core_area_start:], 0, 1,
                                            copy_x_burst_len, 0, 0)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, top_left_ub,
                                                  data_move_ping_ub, core_area_num)
                # top_right
                self.tik_instance.data_move(data_move_pang_ub, self.x_gm[batch_idx, 1, idx, core_area_start:], 0, 1,
                                            copy_x_burst_len, 0, 0)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, top_right_ub,
                                                  data_move_pang_ub, core_area_num)
                # bottom_left
                self.tik_instance.data_move(data_move_ping_ub, self.x_gm[batch_idx, 2, idx, core_area_start:], 0, 1,
                                            copy_x_burst_len, 0, 0)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, bottom_left_ub,
                                                  data_move_ping_ub, core_area_num)
                # bottom_right
                self.tik_instance.data_move(data_move_pang_ub, self.x_gm[batch_idx, 3, idx, core_area_start:], 0, 1,
                                            copy_x_burst_len, 0, 0)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, bottom_right_ub,
                                                  data_move_pang_ub, core_area_num)

            else:
                # top_left
                self.tik_instance.data_move(top_left_ub, self.x_gm[batch_idx, 0, idx, core_area_start:], 0, 1,
                                            copy_x_burst_len, 0, 0)
                # top_right
                self.tik_instance.data_move(top_right_ub, self.x_gm[batch_idx, 1, idx, core_area_start:], 0, 1,
                                            copy_x_burst_len, 0, 0)
                # bottom_left
                self.tik_instance.data_move(bottom_left_ub, self.x_gm[batch_idx, 2, idx, core_area_start:], 0, 1,
                                            copy_x_burst_len, 0, 0)
                # bottom_right
                self.tik_instance.data_move(bottom_right_ub, self.x_gm[batch_idx, 3, idx, core_area_start:], 0, 1,
                                            copy_x_burst_len, 0, 0)
            # calcu top
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vsub",
                                                top_right_ub, top_right_ub, top_left_ub, core_area_num)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul",
                                                top_right_ub, top_right_ub, x_lerp_ub, core_area_num)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd",
                                                top_ub, top_left_ub, top_right_ub, core_area_num)

            # calcu bottom
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vsub",
                                                bottom_right_ub, bottom_right_ub, bottom_left_ub, core_area_num)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul",
                                                bottom_right_ub, bottom_right_ub, x_lerp_ub, core_area_num)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd",
                                                bottom_ub, bottom_left_ub, bottom_right_ub, core_area_num)

            # calcu out
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vsub",
                                                bottom_ub, bottom_ub, top_ub, core_area_num)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul",
                                                bottom_ub, bottom_ub, y_lerp_ub, core_area_num)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd",
                                                bottom_ub, bottom_ub, top_ub, core_area_num)

            if self.x_dtype != self.y_dtype:
                data_move_out_ub = self.tik_instance.Tensor(self.x_dtype, (apply_ub_mem,),
                                                            name="data_move_out_ub",
                                                            scope=tik.scope_ubuf)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, data_move_out_ub, bottom_ub, core_area_num)
                self.tik_instance.data_move(self.z_gm[batch_idx, idx, core_area_start:], data_move_out_ub, 0, 1,
                                            copy_x_burst_len, 0, 0)
            else:
                self.tik_instance.data_move(self.z_gm[batch_idx, idx, core_area_start:], bottom_ub, 0, 1,
                                            copy_x_burst_len, 0, 0)

    def resization_compute(self):
        """
        The tik implementation of operator resization_compute
        """
        self.do_resization()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=(
            self.x_gm,
            self.y_gm,
        ), outputs=(self.z_gm,))
        return self.tik_instance


def img_warp_resize(img, warp_index, warp_img, kernel_name="img_warp_resize"):
    """
    algorithm: img_warp_resize
    calculating: img_warp_resize

    Parameters
    ----------
    img: dict
        dict of input_x, include keys(shape and dtype[float16/float32/uint8])
        5D tensor [N, 4, C, H, W], 4 mean [top_left, top_right, bottom_left, bottom_right]
    warp_index: dict
        dict of input_y, include keys(shape and dtype[float32])
        4D tensor [N, 2, H, W], 2 mean [h_float, w_float]
    warp_img: dict
        dict of  output
        4D tensor [N, C, H, W], resize_bilinear
    kernel_name : str
        cce kernel name, default value is "img_warp_resize"

    Returns
    -------
    None
    """

    gc_object = Resization(img, warp_index, warp_img, kernel_name)

    return gc_object.resization_compute()
