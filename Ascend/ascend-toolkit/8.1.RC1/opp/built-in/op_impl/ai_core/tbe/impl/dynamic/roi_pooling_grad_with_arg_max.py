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
"""
dynamic roi_pool_grad_with_arg_max
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from impl import constant_util as constant


MAX_INT32 = 2 ** 31 - 1
TILING_NUM = 16


# 'pylint: disable=unused-argument,too-many-locals
class RoiPoolingGrad(object):
    # 'pylint: disable=too-many-arguments,too-many-locals
    def __init__(self, grad, x, rois, argmax, y, pooled_h, pooled_w,
                 spatial_scale, kernel_name="roi_pooling_grad_with_arg_max"):
        self.tik_instance = tik.Tik()
        self.x_dtype = x.get("dtype")     
        self.argmax_dtype = argmax.get("dtype")   
        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        self.kernel_name = kernel_name
        
        self.fp16 = "float16"
        self.fp32 = "float32"
        self.int32 = "int32"
        self.is_fp16 = self.x_dtype == self.fp16

        self.block_byte_size = 32
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.block_num = None
        self.total_rois = None
        self.avg_rois = None
        self.last_rois = None
        self.batch_size = None
        self.c = None
        self.fm_h = None
        self.fm_w = None
        self.total_x = None
        self.total_grad = None
        self.support_b = tbe_platform.api_check_support("tik.vcmpv_eq", "int32")
        self.atmic_add_num = 2 if self.is_fp16 else 1
        self.buffer_size = 256 * 32 * 4
        self.data_each_buffer = self.buffer_size // self.get_dtype_size(self.x_dtype)
        self.data_each_group_hw = 1
        self.data_each_mask = 128 if self.is_fp16 else 64
        self.blocks_per_numc = 1
        self.spatial_scale = spatial_scale

        self.grad_gm = self.tik_instance.Tensor(self.x_dtype, [MAX_INT32], scope=tik.scope_gm, name="grad_gm")
        self.feature_map_gm = self.tik_instance.Tensor(self.x_dtype, [MAX_INT32], scope=tik.scope_gm, 
                                                       name="feature_map_gm")
        self.rois_gm = self.tik_instance.Tensor(self.x_dtype, [MAX_INT32], scope=tik.scope_gm, name="rois_gm")
        self.roi_actual_num_gm = self.tik_instance.Tensor(self.argmax_dtype, [MAX_INT32], scope=tik.scope_gm, 
                                                          name="roi_actual_num_gm")
        self.argmax_gm = self.tik_instance.Tensor(self.argmax_dtype, [MAX_INT32], scope=tik.scope_gm, 
                                                  name="argmax_gm")
        self.assist_gm = self.tik_instance.Tensor(self.argmax_dtype, [MAX_INT32], name="assist_gm",
                                            scope=tik.scope_gm, is_workspace=True)
        self.y_gm = self.tik_instance.Tensor(self.x_dtype, [MAX_INT32], scope=tik.scope_gm, name="y_gm",
                                             is_atomic_add=True)
        self.tiling_gm = self.tik_instance.Tensor("int32", [TILING_NUM], scope=tik.scope_gm, name="tiling_gm")

    def get_dtype_size(self, dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2, "int64": 8}
        return dtype_dict.get(dtype)

    def get_tiling_params(self):
        """
        get runtime params from tiling
        :return: None
        """
        self.block_num = self.tik_instance.Scalar("int32")
        self.total_rois = self.tik_instance.Scalar("int32")
        self.avg_rois = self.tik_instance.Scalar("int32")
        self.last_rois = self.tik_instance.Scalar("int32")
        self.batch_size = self.tik_instance.Scalar("int32")    
        self.c = self.tik_instance.Scalar("int32")    
        self.fm_h = self.tik_instance.Scalar("int32")
        self.fm_w = self.tik_instance.Scalar("int32")   
        self.total_x = self.tik_instance.Scalar("int32")
        self.total_grad = self.tik_instance.Scalar("int32")

        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32", shape=(TILING_NUM,), scope=tik.scope_ubuf, name="tiling_ub")
            self.data_move(tiling_ub, self.tiling_gm, num=TILING_NUM)
            self.block_num.set_as(tiling_ub[0])
            self.total_rois.set_as(tiling_ub[1])
            self.avg_rois.set_as(tiling_ub[2])
            self.last_rois.set_as(tiling_ub[3])
            self.batch_size.set_as(tiling_ub[4])
            self.c.set_as(tiling_ub[5])
            self.fm_h.set_as(tiling_ub[6])
            self.fm_w.set_as(tiling_ub[7])
            self.total_x.set_as(tiling_ub[8])
            self.total_grad.set_as(tiling_ub[9])

    def compute(self):
        self.get_tiling_params()
        self.data_each_group_hw = (self.data_each_buffer + self.c - 1) // self.c
        self.blocks_per_numc = self.get_dtype_size(self.x_dtype) * self.c // self.block_byte_size
        with self.tik_instance.new_stmt_scope():
            assist_ub = self.tik_instance.Tensor(self.argmax_dtype, [self.c], scope=tik.scope_ubuf, name="assist_ub")
            with self.tik_instance.for_range(0, self.fm_h * self.fm_w) as i:
                self.dup_value(assist_ub, num=self.c, dup_value=i)
                self.data_move(self.assist_gm[i * self.c], assist_ub, num=self.c)
        if self.support_b:
            self.tik_instance.set_atomic_add(self.atmic_add_num)
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_idx:
            roi_num = self.tik_instance.Scalar(self.int32)
            with self.tik_instance.if_scope(block_idx < self.block_num):
                with self.tik_instance.if_scope(block_idx < self.block_num - 1):
                    roi_num.set_as(self.avg_rois)
                with self.tik_instance.if_scope(block_idx == self.block_num - 1):
                    roi_num.set_as(self.last_rois)
                self.compute_per_core(block_idx, roi_num)
        if self.support_b:
            self.tik_instance.set_atomic_add(0)

        opt_config = {"enable_const_fold": True}
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num})

        inputs = [self.grad_gm, self.feature_map_gm, self.rois_gm, self.roi_actual_num_gm, self.argmax_gm]
        outputs = [self.y_gm]

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=inputs, outputs=outputs,
                                   flowtable=[self.tiling_gm], config=opt_config)
        return self.tik_instance    
    
    def compute_per_core(self, block_idx, roi_num):
        base_idx = self.tik_instance.Scalar(self.int32, init_value=0)

        grad_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_group_hw, self.c], scope=tik.scope_ubuf, 
                                    name="grad_ub")
        rois_ub = self.tik_instance.Tensor(self.x_dtype, 
                                    [(roi_num * 5 + self.block_byte_size - 1) 
                                    // self.block_byte_size * self.block_byte_size], 
                                    scope=tik.scope_ubuf, name="rois_ub")
        argmax_ub = self.tik_instance.Tensor(self.argmax_dtype, [self.data_each_group_hw, self.c], scope=tik.scope_ubuf, 
                                    name="argmax_ub")
        matrix_ub = self.tik_instance.Tensor(self.argmax_dtype, [self.data_each_group_hw, self.c], scope=tik.scope_ubuf, 
                                    name="matrix_ub")
        rois_int_ub = self.tik_instance.Tensor(self.int32, 
                                    [(roi_num * 5 + self.block_byte_size - 1) 
                                    // self.block_byte_size * self.block_byte_size], 
                                    scope=tik.scope_ubuf, name="rois_int_ub")
        y_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_group_hw, self.c], scope=tik.scope_ubuf, 
                                    name="y_ub")

        base_idx.set_as(block_idx * self.avg_rois)
        self.data_move(rois_ub, self.rois_gm[base_idx * 5], num=roi_num * 5)
        self.tik_instance.h_cast(rois_int_ub, rois_ub, "round")
        with self.tik_instance.for_range(0, roi_num) as n_idx:
            self.loop_compute(grad_ub, rois_ub, rois_int_ub, argmax_ub, matrix_ub, y_ub, n_idx, block_idx)
    
    # 'pylint: disable=too-many-arguments,too-many-locals
    def loop_compute(self, grad_ub, rois_ub, rois_int_ub, argmax_ub, matrix_ub, y_ub, n_idx, block_idx):
        in_offset = self.tik_instance.Scalar(self.int32, init_value=0)
        batch_id = self.tik_instance.Scalar(self.int32, init_value=0)
        h_min = self.tik_instance.Scalar(self.int32, init_value=0)
        h_max = self.tik_instance.Scalar(self.int32, init_value=0)
        w_min = self.tik_instance.Scalar(self.int32, init_value=0)
        w_max = self.tik_instance.Scalar(self.int32, init_value=0)
        fh_min = self.tik_instance.Scalar(self.fp32, init_value=0)
        fh_max = self.tik_instance.Scalar(self.fp32, init_value=0)
        fw_min = self.tik_instance.Scalar(self.fp32, init_value=0)
        fw_max = self.tik_instance.Scalar(self.fp32, init_value=0)
        fpooledh = self.tik_instance.Scalar(self.fp32, init_value=0)
        fpooledw = self.tik_instance.Scalar(self.fp32, init_value=0)
        matrix_h1 = self.tik_instance.Scalar(self.fp32, init_value=0)
        matrix_w1 = self.tik_instance.Scalar(self.fp32, init_value=0)
        ftmp = self.tik_instance.Scalar(self.fp32, init_value=0)
        x0 = self.tik_instance.Scalar(self.int32, init_value=0)
        x1 = self.tik_instance.Scalar(self.int32, init_value=0)
        y0 = self.tik_instance.Scalar(self.int32, init_value=0)
        y1 = self.tik_instance.Scalar(self.int32, init_value=0)
        matrix_h = self.tik_instance.Scalar(self.int32, init_value=0)
        matrix_w = self.tik_instance.Scalar(self.int32, init_value=0)
        matrix_offset = self.tik_instance.Scalar(self.int32, init_value=0)
        y_offset = self.tik_instance.Scalar(self.int32, init_value=0)
        matrix_offset_loop = self.tik_instance.Scalar(self.int32, init_value=0)
        matrix_stride = self.tik_instance.Scalar(self.int32, init_value=0)
        h_offset = self.tik_instance.Scalar(self.int32, init_value=0)
        w_offset = self.tik_instance.Scalar(self.int32, init_value=0)
        ffmh = self.tik_instance.Scalar(self.fp32, init_value=0)
        ffmw = self.tik_instance.Scalar(self.fp32, init_value=0)

        mask_ub = self.tik_instance.Tensor("uint8", [self.data_each_group_hw, self.c], scope=tik.scope_ubuf, 
                                                name="mask_ub")                

        batch_id.set_as(rois_int_ub[n_idx * 5])
        w_min.set_as(rois_int_ub[n_idx * 5 + 1])
        h_min.set_as(rois_int_ub[n_idx * 5 + 2])
        w_max.set_as(rois_int_ub[n_idx * 5 + 3])
        h_max.set_as(rois_int_ub[n_idx * 5 + 4])
        fw_min.set_as(rois_ub[n_idx * 5 + 1])
        fw_min.set_as(fw_min * self.spatial_scale)
        fh_min.set_as(rois_ub[n_idx * 5 + 2])
        fh_min.set_as(fh_min * self.spatial_scale)
        fw_max.set_as(rois_ub[n_idx * 5 + 3])
        fw_max.set_as((fw_max + 1) * self.spatial_scale)
        fh_max.set_as(rois_ub[n_idx * 5 + 4])
        fh_max.set_as((fh_max + 1) * self.spatial_scale)
        fpooledh.set_as(self.pooled_h)
        fpooledw.set_as(self.pooled_w)
        matrix_h1.set_as((fh_max - fh_min) / self.pooled_h)
        matrix_w1.set_as((fw_max - fw_min) / self.pooled_w)
        matrix_offset.set_as((h_min * self.fm_w + w_min) * self.c)
        ffmh.set_as(self.fm_h)
        ffmw.set_as(self.fm_w)
        y_offset.set_as(batch_id * self.fm_h * self.fm_w * self.c)
        with self.tik_instance.for_range(0, self.pooled_h) as ph:
            ftmp.set_as(ph * matrix_h1 + fh_min)
            if self.support_b:   
                self.tik_instance.scalar_min(ftmp, ftmp, ffmh)                 
            else:
                with self.tik_instance.if_scope(ftmp > ffmh):
                    ftmp.set_as(ffmh)
            self.tik_instance.scalar_conv('floor', y0, ftmp)
            ftmp.set_as((ph + 1) * matrix_h1 + fh_min)
            if self.support_b:
                self.tik_instance.scalar_min(ftmp, ftmp, ffmh)
            else:
                with self.tik_instance.if_scope(ftmp > ffmh):
                    ftmp.set_as(ffmh)
            self.tik_instance.scalar_conv('ceil', y1, ftmp)
            matrix_h.set_as(y1 - y0)
            with self.tik_instance.for_range(0, self.pooled_w) as pw:
                ftmp.set_as(pw * matrix_w1 + fw_min)
                if self.support_b:
                    self.tik_instance.scalar_min(ftmp, ftmp, ffmw)
                else:
                    with self.tik_instance.if_scope(ftmp > ffmw):
                        ftmp.set_as(ffmw)
                self.tik_instance.scalar_conv('floor', x0, ftmp)
                ftmp.set_as((pw + 1) * matrix_w1 + fw_min)
                if self.support_b:
                    self.tik_instance.scalar_min(ftmp, ftmp, ffmw)
                else:
                    with self.tik_instance.if_scope(ftmp > ffmw):
                        ftmp.set_as(ffmw)
                self.tik_instance.scalar_conv('ceil', x1, ftmp)
                matrix_w.set_as(x1 - x0)
                matrix_stride.set_as(self.fm_w - matrix_w)
                w_offset.set_as(x0 - w_min)
                h_offset.set_as(y0 - h_min)
                matrix_offset_loop.set_as(matrix_offset + (w_offset + self.fm_w * h_offset) * self.c)
                in_offset.set_as(((block_idx * self.avg_rois + n_idx) * self.pooled_h 
                                * self.pooled_w + ph * self.pooled_w + pw) * self.c)

                with self.tik_instance.if_scope(self.data_each_group_hw >= matrix_w):
                    self.compute_bin_smaller(grad_ub, argmax_ub, matrix_ub, mask_ub, y_ub,
                        matrix_h, matrix_w, matrix_offset_loop, in_offset, matrix_stride, y_offset)

                with self.tik_instance.else_scope():
                    self.compute_bin_larger(grad_ub, argmax_ub, matrix_ub, mask_ub, y_ub,
                        matrix_h, matrix_w, matrix_offset_loop, in_offset, matrix_stride, y_offset)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def compute_bin_smaller(self, grad_ub, argmax_ub, matrix_ub, mask_ub, y_ub,
            matrix_h, matrix_w, matrix_offset_loop, in_offset, matrix_stride, y_offset):
        h_each_time = self.tik_instance.Scalar(self.int32, init_value=0)
        databyte_each_loop = self.tik_instance.Scalar(self.int32, init_value=0)
        bin_loop = self.tik_instance.Scalar(self.int32, init_value=0)
        bin_left = self.tik_instance.Scalar(self.int32, init_value=0)
        zeros = self.tik_instance.Scalar(self.x_dtype, init_value=0)
        data_left = self.tik_instance.Scalar(self.int32, init_value=0)
        matrix_index = self.tik_instance.Scalar(self.int32, init_value=0)

        h_each_time.set_as(self.data_each_group_hw // matrix_w)
        databyte_each_loop.set_as(h_each_time * matrix_w * self.c * self.get_dtype_size(self.x_dtype))
        bin_loop.set_as(matrix_h // h_each_time)
        bin_left.set_as(matrix_h % h_each_time)

        self.grad_argmax_copy(grad_ub, argmax_ub, h_each_time * matrix_w, in_offset)
        
        matrix_index.set_as(matrix_offset_loop)
        with self.tik_instance.for_range(0, bin_loop):
            if self.support_b:
                self.tik_instance.data_move(matrix_ub, self.assist_gm[matrix_index], 0, h_each_time,
                    matrix_w * self.blocks_per_numc, matrix_stride * self.blocks_per_numc, 0)
                self.tik_instance.vcmpv_eq(mask_ub, matrix_ub, argmax_ub, 
                    (databyte_each_loop + (self.block_byte_size * 8) - 1) // (self.block_byte_size * 8), 1, 1, 8, 8)
                self.tik_instance.vec_sel(self.data_each_mask, 1, y_ub, mask_ub, grad_ub, zeros, 
                    (databyte_each_loop + (self.block_byte_size * 8) - 1) // (self.block_byte_size * 8), 8, 8, 8)
                self.tik_instance.data_move(self.y_gm[matrix_index + y_offset], y_ub, 0, h_each_time,
                    matrix_w * self.blocks_per_numc, 0, matrix_stride * self.blocks_per_numc)
            matrix_index.set_as(matrix_index + self.c * h_each_time * self.fm_w)

        with self.tik_instance.if_scope(bin_left > 0):
            data_left.set_as(bin_left * self.c * matrix_w * self.get_dtype_size(self.x_dtype))
            if self.support_b:
                self.tik_instance.data_move(matrix_ub, self.assist_gm[matrix_index], 0, bin_left,
                    matrix_w * self.blocks_per_numc, matrix_stride * self.blocks_per_numc, 0)
                self.tik_instance.vcmpv_eq(mask_ub, matrix_ub, argmax_ub, 
                    (databyte_each_loop + (self.block_byte_size * 8) - 1) // (self.block_byte_size * 8), 1, 1, 8, 8)
                self.tik_instance.vec_sel(self.data_each_mask, 1, y_ub, mask_ub, grad_ub, zeros, 
                    (databyte_each_loop + (self.block_byte_size * 8) - 1) // (self.block_byte_size * 8), 8, 8, 8)  
                self.tik_instance.data_move(self.y_gm[matrix_index + y_offset], y_ub, 0, bin_left,
                    matrix_w * self.blocks_per_numc, 0, matrix_stride * self.blocks_per_numc)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def compute_bin_larger(self, grad_ub, argmax_ub, matrix_ub, mask_ub, y_ub,
            matrix_h, matrix_w, matrix_offset_loop, in_offset, matrix_stride, y_offset):
        h_each_time = self.tik_instance.Scalar(self.int32, init_value=0)
        bin_left = self.tik_instance.Scalar(self.int32, init_value=0)
        zeros = self.tik_instance.Scalar(self.x_dtype, init_value=0)
        data_left = self.tik_instance.Scalar(self.int32, init_value=0)
        matrix_index = self.tik_instance.Scalar(self.int32, init_value=0)

        h_each_time.set_as(matrix_w // self.data_each_group_hw)
        bin_left.set_as(matrix_w % self.data_each_group_hw)

        self.grad_argmax_copy(grad_ub, argmax_ub, self.data_each_group_hw, in_offset)

        matrix_index.set_as(matrix_offset_loop)
        with self.tik_instance.for_range(0, matrix_h) as i:
            with self.tik_instance.for_range(0, h_each_time):
                if self.support_b:
                    self.data_move(matrix_ub, self.assist_gm[matrix_index], num=self.data_each_buffer)
                    self.tik_instance.vcmpv_eq(mask_ub, matrix_ub, argmax_ub, 
                        (self.buffer_size + (self.block_byte_size * 8) - 1) // (self.block_byte_size * 8), 1, 1, 8, 8)
                    self.tik_instance.vec_sel(self.data_each_mask, 1, y_ub, mask_ub, grad_ub, zeros,
                        (self.buffer_size + (self.block_byte_size * 8) - 1) // (self.block_byte_size * 8), 8, 8, 8)
                    self.data_move(self.y_gm[matrix_index + y_offset], y_ub, num=self.data_each_buffer)
                matrix_index.set_as(matrix_index + self.data_each_buffer)
            with self.tik_instance.if_scope(bin_left > 0):
                data_left.set_as(bin_left * self.c * self.get_dtype_size(self.x_dtype))
                if self.support_b:
                    self.data_move(matrix_ub, self.assist_gm[matrix_index], num=bin_left * self.c)
                    self.tik_instance.vcmpv_eq(mask_ub, matrix_ub, argmax_ub,
                        (data_left + (self.block_byte_size * 8) - 1) // (self.block_byte_size * 8), 1, 1, 8, 8)
                    self.tik_instance.vec_sel(self.data_each_mask, 1, y_ub, mask_ub, grad_ub, zeros, 
                        (data_left + (self.block_byte_size * 8) - 1) // (self.block_byte_size * 8), 8, 8, 8)
                    self.data_move(self.y_gm[matrix_index + y_offset], y_ub, num=bin_left * self.c)
            matrix_index.set_as(matrix_offset_loop + self.fm_w * self.c * (i + 1))

    # 'pylint: disable=too-many-arguments,too-many-locals
    def grad_argmax_copy(self, grad_ub, argmax_ub, loop_time, in_offset):
        copy_offset = self.tik_instance.Scalar(self.int32, init_value=0)
        copy_offset.set_as(0)
        if self.support_b:
            with self.tik_instance.for_range(0, loop_time):
                self.data_move(grad_ub[copy_offset], self.grad_gm[in_offset], num=self.c)
                self.data_move(argmax_ub[copy_offset], self.argmax_gm[in_offset], num=self.c)
                copy_offset.set_as(copy_offset + self.c)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def data_move(self, dst, src, num, src_stride=0, dst_stride=0):
        """
        move data
        """
        sid = 0
        nburst = 1
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        burst_len = (num + data_each_block - 1) // data_each_block
        self.tik_instance.data_move(dst, src, sid, nburst, burst_len, src_stride=src_stride, dst_stride=dst_stride)

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        dup value to ub
        """
        offset = self.tik_instance.Scalar("int32", init_value=offset)
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        mask = 256 // dtype_byte_size
        stride = mask // data_each_block

        loop = num // (mask * 255)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
            offset.set_as(offset + loop * mask * 255)

        repeat_time = (num % (mask * 255)) // mask
        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
            offset.set_as(offset + repeat_time * mask)

        last_num = num % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)   

# 'pylint: disable=unused-argument, too-many-locals, too-many-lines


@register_operator("RoiPoolingGradWithArgMax")


@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
# 'pylint: disable=too-many-arguments,too-many-locals
def roi_pooling_grad_with_arg_max(grad, x, rois, roi_actual_num, argmax, y, pooled_h, pooled_w, spatial_scale_h,
                             spatial_scale_w, pool_channel, kernel_name="roi_pooling_grad_with_arg_max"):
    """
    calculating data

    Parameters
    ----------
    grad: dict
        shape and dtype of backward grad
    x : dict
        shape and dtype of feature_map after roi pooling
    rois : dict
        shape and dtype of input rois (N,5)
    rois_actual_num: int
        nums of rois (N), defined according to IR but not really used
    y : dict
        shape and dtype of output (N, C, pooled_h, pooled_w)
    pooled_h : int
        the height of pooling
    pooled_w : int
        the width of pooling
    spatial_scale_h : float
        the spatial_scale of rois height
    spatial_scale_w : float
        the spatial_scale of rois width
    pool_channel:  int
        the pooling channel
    kernel_name : str
        kernel name, default value is "roi_pooling_grad_with_arg_max"

    Returns
    -------
    None
    """
    grad_dtype = grad.get("dtype").lower()
    x_dtype = x.get("dtype").lower()
    rois_dtype = rois.get("dtype").lower()
    argmax_dtype = argmax.get("dtype").lower()
    para_check.check_dtype(grad_dtype, ["float16", "float32"])
    para_check.check_dtype(x_dtype, ["float16", "float32"])
    para_check.check_dtype(rois_dtype, ["float16", "float32"])
    para_check.check_dtype(argmax_dtype, ["int32"])

    instance = RoiPoolingGrad(grad, x, rois, argmax, y, pooled_h, pooled_w, spatial_scale_h, kernel_name)
    return instance.compute()
