#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2023-2023 Huawei Technologies Co., Ltd
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
dynamic roi_extractor
"""
from impl import common_util
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.dynamic.roi_align_compute import RoiAlignCompute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for Constant
    """
    C0 = 16
    ZERO = 0.0
    ROI_NUM = 128
    B32_MASK = 64
    REPEAT_FP32 = 2
    ALIGN_LEN = 128
    B32_PER_BLOCK = 8
    BLOCK_BIT_SIZE = 32
    TILING_ARG_NUM = 72
    PARAMS_SIZE = 2 ** 30 - 1
    RESERVED_UB_BYTES = 32 * 1024


# 'pylint: disable=unused-argument,too-many-arguments
class RoiExtractor:
    """
    RoiExtractor op.
    """

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def __init__(self, feats, rois, index, roi_feats, finest_scale, roi_scale_factor, spatial_scale, pooled_h,
                 pooled_w, sample_num, pool_mode, aligned, kernel_name):
        """init."""
        # This operator currently only support average pooling
        if pool_mode != 'avg':
            raise RuntimeError("pool_mode only support avg now!")

        # Init parameters
        self.num_levels = len(feats)
        self.rois_dtype = rois.get("dtype")
        self.x_dtype = feats[0].get("dtype")
        self.index = index
        self.kernel_name = kernel_name
        self.aligned = aligned
        self.get_shape(feats, rois, index, roi_feats)

        # Prepare TIk
        self.tik = tik.Tik(tik.Dprofile())
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB_BYTES
        if not tbe_platform.intrinsic_check_support("Intrinsic_data_move_l12ub"):
            self.l1_size_bytes = 0
        else:
            self.l1_size_bytes = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
        self.support_vbi = tbe_platform.api_check_support("tik.vbi", self.x_dtype) and not \
            tbe_platform.api_check_support("tik.vbi", "float32")
        self.init_scalar(index)

        # Get Tiling data
        self.tiling_gm = self.tik.Tensor("int32", [Constant.TILING_ARG_NUM], name="tiling_gm", scope=tik.scope_gm)
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num,
                                                            "ub_size": self.ub_size,
                                                            "dynamic_tiling": 0,
                                                            "use_vbi": int(self.support_vbi)})
        self.get_tiling_args()
        self.convert_scale()

        # Calculate some attributes
        self.pow_table = [2 ** i * self.finest_scale for i in range(self.num_levels + 1)]
        self.index_arr = list(range(self.num_levels))
        self.data_size = common_util.get_data_size(self.x_dtype)
        self.block_element = common_util.get_block_element(self.x_dtype)
        self.max_ub_element = self.ub_size // self.data_size
        self.c0_blocks = Constant.C0 // self.block_element

        # Init gm tensors
        self.feats_gm_list = [self.tik.Tensor(self.x_dtype, self.feats_shape[i]["shape"],
                                              name="feats_gm_%d" % i, scope=tik.scope_gm,) for i in self.index_arr]
        self.rois_gm = self.tik.Tensor(self.rois_dtype, self.rois_shape["shape"], name="rois_gm", scope=tik.scope_gm)
        self.roi_feats_gm = self.tik.Tensor(self.x_dtype, self.roi_feats_shape["shape"],
                                            name="roi_feats_gm", scope=tik.scope_gm)

        # Init input shape
        self.y_shape_last_4_dim = self.tik.Scalar("int32", init_value=self.feats_c1_dim * self.pooled_h *
                                                  self.pooled_w * self.feats_c0_dim)

    def get_shape(self, feats, rois, index, roi_feats):
        """
        Get shape of input and output and judge if the shape is dynamic.
        """
        self.feats_shape = []
        for i in range(self.num_levels):
            if -1 in feats[i].get("shape") or -2 in feats[i].get("shape"):
                self.feats_shape.append({"shape": [Constant.PARAMS_SIZE], "dynamic": True})
            else:
                self.feats_shape.append({"shape": feats[i].get("shape"), "dynamic": False})  
        if -1 in rois.get("shape") or -2 in rois.get("shape"):
            self.rois_shape = {"shape": [Constant.PARAMS_SIZE], "dynamic": True}
        else:
            self.rois_shape = {"shape": rois.get("shape"), "dynamic": False}
        if index is not None:
            if -1 in index.get("shape") or -2 in index.get("shape"):
                self.index_shape = {"shape": [Constant.PARAMS_SIZE], "dynamic": True}
            else:
                self.index_shape = {"shape": index.get("shape"), "dynamic": False}
        if -1 in roi_feats.get("shape") or -2 in roi_feats.get("shape"):
            self.roi_feats_shape = {"shape": [Constant.PARAMS_SIZE], "dynamic": True}
        else:
            self.roi_feats_shape = {"shape": roi_feats.get("shape"), "dynamic": False}

    
    def init_scalar(self, index):
        """
        Init Scalar parameters and attributes.
        """
        self.core_num_var = self.tik.Scalar("int32")
        self.feats_n_dim = self.tik.Scalar("int32")
        self.feats_c1_dim = self.tik.Scalar("int32")
        self.feats_c0_dim = self.tik.Scalar("int32")
        self.rois_total_num = self.tik.Scalar("int32")
        self.rois_last_dim = self.tik.Scalar("int32")
        self.feats_h_dims = [self.tik.Scalar("int32") for _ in range(self.num_levels)]
        self.feats_w_dims = [self.tik.Scalar("int32") for _ in range(self.num_levels)]
        if index is not None:
            self.index_shape = self.tik.Scalar("int32")
        self.finest_scale = self.tik.Scalar("int32")
        self.roi_scale_factor = self.tik.Scalar("float32")
        self.pooled_h = self.tik.Scalar("int32")
        self.pooled_w = self.tik.Scalar("int32")
        self.spatial_scale_temp = [self.tik.Scalar("float32") for _ in range(self.num_levels)]
        self.sample_num = self.tik.Scalar("int32")
        self.pooled_h_reciprocal = self.tik.Scalar("float32")
        self.pooled_w_reciprocal = self.tik.Scalar("float32")
        self.sample_num_reciprocal = self.tik.Scalar("float32")
        self.c1_per_core = self.tik.Scalar("int32")
        self.tiliing_key = [self.tik.Scalar("int32") for _ in range(self.num_levels)]
        self.out_c1_per_loop = [self.tik.Scalar("int32") for _ in range(self.num_levels)]
        self.out_h_per_loop = self.tik.Scalar("int32")
        self.out_w_per_loop = [self.tik.Scalar("int32") for _ in range(self.num_levels)]

    def get_tiling_args(self):
        """Get tiling data."""
        tiling_ub = self.tik.Tensor("int32", [Constant.TILING_ARG_NUM], name='tiling_ub', scope=tik.scope_ubuf)
        burst_len = ceil_div(Constant.TILING_ARG_NUM, Constant.B32_PER_BLOCK)
        self.tik.data_move(tiling_ub, self.tiling_gm, 0, 1, burst_len, 0, 0)

        self.core_num_var.set_as(tiling_ub[0])
        self.feats_n_dim.set_as(tiling_ub[1])
        self.feats_c1_dim.set_as(tiling_ub[2])
        self.feats_c0_dim.set_as(tiling_ub[3])
        for i in range(self.num_levels):
            self.feats_h_dims[i].set_as(tiling_ub[4 + i])
        for i in range(self.num_levels):
            self.feats_w_dims[i].set_as(tiling_ub[12 + i])
        self.rois_total_num.set_as(tiling_ub[20])
        self.rois_last_dim.set_as(tiling_ub[21])
        if self.index is not None:
            self.index_shape.set_as(tiling_ub[22])
        self.finest_scale.set_as(tiling_ub[23])
        self.pooled_h.set_as(tiling_ub[24])
        self.pooled_w.set_as(tiling_ub[25])
        self.sample_num.set_as(tiling_ub[26])
        for i in range(self.num_levels):
            self.tiliing_key[i].set_as(tiling_ub[27 + i])
        self.c1_per_core.set_as(tiling_ub[35])
        self.out_h_per_loop.set_as(tiling_ub[36])
        for i in range(self.num_levels):
            self.out_c1_per_loop[i].set_as(tiling_ub[37 + i])
        for i in range(self.num_levels):
            self.out_w_per_loop[i].set_as(tiling_ub[45 + i])

        tiling_ub_fp32 = tiling_ub.reinterpret_cast_to("float32")
        self.roi_scale_factor.set_as(tiling_ub_fp32[53])
        for i in range(self.num_levels):
            self.spatial_scale_temp[i].set_as(tiling_ub_fp32[54 + i])
        self.pooled_h_reciprocal.set_as(tiling_ub_fp32[62])
        self.pooled_w_reciprocal.set_as(tiling_ub_fp32[63])
        self.sample_num_reciprocal.set_as(tiling_ub_fp32[64])

    def convert_scale(self):
        """
        Convert dtype of spatial_scale.
        """
        with self.tik.if_scope(self.x_dtype == "float16"):
            self.spatial_scale = []
            for sc in self.spatial_scale_temp:
                sc_fp16 = self.tik.Scalar("float16")
                self.tik.scalar_conv("", sc_fp16, sc)
                self.spatial_scale.append(sc_fp16)
        with self.tik.else_scope():
            self.spatial_scale = self.spatial_scale_temp

    def compute(self, index):
        """
        The main computing function of RoiExtractor.
        """
        support_balance = tbe_platform.api_check_support("tik.v4dtrans")

        if support_balance:
            return self.balance_compute(index)
        else:
            return self.feature_per_core_compute()

    def feature_per_core_compute(self):
        """
        Calculation with one feature map per core.
        """
        data_each_block = Constant.BLOCK_BIT_SIZE // self.data_size

        if self.rois_shape["dynamic"]:
            roi_buf = self.tik.Tensor(self.rois_dtype, self.rois_shape["shape"], name="roi_buf",
                                    scope=tik.scope_gm, is_workspace=True)
        else:
            shape = [self.num_levels * self.rois_shape["shape"][0], 5]
            roi_buf = self.tik.Tensor(self.rois_dtype, shape, name="roi_buf",
                                    scope=tik.scope_gm, is_workspace=True)
        if self.roi_feats_shape["dynamic"]:
            roi_feats_buf = self.tik.Tensor(self.x_dtype, self.roi_feats_shape["shape"], name="roi_feats_buf",
                                            scope=tik.scope_gm, is_workspace=True)
        else:
            shape = [self.num_levels * self.roi_feats_shape["shape"][0]] + list(self.roi_feats_shape["shape"][1:])
            roi_feats_buf = self.tik.Tensor(self.x_dtype, shape, name="roi_feats_buf",
                                            scope=tik.scope_gm, is_workspace=True)

        with self.tik.for_range(0, self.num_levels, block_num=self.num_levels) as block_idx:
            inds_ub = self.tik.Tensor("int32", (ceil_div(self.rois_total_num, 128) * 128, ),
                                      name="inds_ub", scope=tik.scope_ubuf)

            # Valid rois number in current level
            index_reg = self.tik.Scalar("int32", name="index_reg")
            idx = self.tik.Scalar("int32", name="roi_idx")

            output_offset_start = self._get_output_offset(block_idx * self.rois_total_num, 0, 0, 0)
            output_offset_end = self._get_output_offset((block_idx + 1) * self.rois_total_num, 0, 0, 0)

            with self.tik.new_stmt_scope():
                target_lvls = self.tik.Tensor("float16", (ceil_div(self.rois_total_num, 128) * 128,),
                                              name="target_lvls_ub", scope=tik.scope_ubuf)
                self.map_roi_levels(target_lvls, self.rois_total_num)
                self.where_and_nonzero(target_lvls, index_reg, inds_ub, block_idx, self.rois_total_num)

                rois_ub = self.tik.Tensor(self.rois_dtype, (data_each_block, ),
                                          name="rois_ub", scope=tik.scope_ubuf)
                with self.tik.for_range(0, index_reg) as roi_i:
                    idx.set_as(inds_ub[roi_i])
                    self.tik.data_move(rois_ub, self.rois_gm[idx * self.rois_last_dim], 0, 1, 1, 0, 0)
                    self.tik.data_move(roi_buf[(block_idx * self.rois_total_num + roi_i) * 5],
                                       rois_ub, 0, 1, 1, 0, 0)

            with self.tik.if_scope(index_reg > 0):
                for i in self.index_arr:
                    with self.tik.if_scope(block_idx == i):
                        rac = RoiAlignCompute(self,
                                              self.feats_gm_list[i],
                                              roi_buf[block_idx * self.rois_total_num * 5:
                                                      (block_idx + 1) * self.rois_total_num * 5],
                                              roi_feats_buf[output_offset_start: output_offset_end],
                                              0,
                                              index_reg,
                                              i)
                        rac.roi_align_compute()

            with self.tik.new_stmt_scope():
                out_ub = self.tik.Tensor(self.x_dtype, (self.y_shape_last_4_dim, ),
                                         name="out_ub", scope=tik.scope_ubuf)
                with self.tik.for_range(0, index_reg) as roi_i:
                    idx.set_as(inds_ub[roi_i])
                    self.tik.data_move(out_ub,
                                       roi_feats_buf[(block_idx * self.rois_total_num + roi_i) *
                                                     self.y_shape_last_4_dim],
                                       0, 1, ceil_div(self.y_shape_last_4_dim, data_each_block), 0, 0)
                    self.tik.data_move(self.roi_feats_gm[idx * self.y_shape_last_4_dim],
                                       out_ub, 0, 1, ceil_div(self.y_shape_last_4_dim, data_each_block), 0, 0)

        # Build CCE
        sch_list = self.feats_gm_list + [self.rois_gm]
        opt_config = {"enable_const_fold": True, "out_of_bound_sync_check": False}
        self.tik.BuildCCE(kernel_name=self.kernel_name,
                          inputs=sch_list,
                          outputs=[self.roi_feats_gm],
                          flowtable=(self.tiling_gm,),
                          config=opt_config)
        return self.tik

    def balance_compute(self, index):
        """
        Calculation with balanced tiling.
        """
        index_ub, data_index = None, None
        if index is not None:
            index_dtype = index.get("dtype")
            data_index = self.tik.Tensor(index_dtype, [self.index_shape], name="index", scope=tik.scope_gm)

        data_each_block = Constant.BLOCK_BIT_SIZE // self.data_size
        
        if self.rois_shape["dynamic"]:
            roi_buf = self.tik.Tensor(self.rois_dtype, self.rois_shape["shape"], name="roi_buf",
                                    scope=tik.scope_gm, is_workspace=True)
            roi_buf_last = self.tik.Tensor(self.rois_dtype, self.rois_shape["shape"], name="roi_buf_last",
                                       scope=tik.scope_gm, is_workspace=True)
        else:
            shape = [self.num_levels * self.rois_shape["shape"][0], 5]
            roi_buf = self.tik.Tensor(self.rois_dtype, shape, name="roi_buf",
                                    scope=tik.scope_gm, is_workspace=True)
            roi_buf_last = self.tik.Tensor(self.rois_dtype, shape, name="roi_buf_last",
                                       scope=tik.scope_gm, is_workspace=True)
        if self.roi_feats_shape["dynamic"]:
            roi_feats_buf = self.tik.Tensor(self.x_dtype, self.roi_feats_shape["shape"], name="roi_feats_buf",
                                            scope=tik.scope_gm, is_workspace=True)
        else:
            shape = [self.num_levels * self.roi_feats_shape["shape"][0]] + list(self.roi_feats_shape["shape"][1:])
            roi_feats_buf = self.tik.Tensor(self.x_dtype, shape, name="roi_feats_buf",
                                            scope=tik.scope_gm, is_workspace=True)

        roi_per_core = self.tik.Scalar("int32", init_value=ceil_div(self.rois_total_num, self.core_num_var))
        block_num = self.tik.Scalar("int32", init_value=ceil_div(self.rois_total_num, roi_per_core))
        last_roi = self.tik.Scalar("int32", init_value=self.rois_total_num - roi_per_core * (block_num - 1))

        with self.tik.for_range(0, block_num, block_num=block_num) as block_idx:
            offset = self.tik.Scalar("int32", init_value=block_idx * roi_per_core)

            # Valid rois number in current level
            index_reg = self.tik.Scalar("int32", name="index_reg")
            idx_tmp = self.tik.Scalar("int32", name="roi_idx_tmp")
            idx = self.tik.Scalar("int32", name="roi_idx")

            with self.tik.if_scope(block_idx != block_num - 1):
                inds_ub = self.tik.Tensor("int32", (ceil_div(roi_per_core, 128) * 128,),
                                          name="inds_ub", scope=tik.scope_ubuf)
                if index is not None:
                    index_ub = self.tik.Tensor("int32", (roi_per_core,), name="index_ub", scope=tik.scope_ubuf)
                    self.tik.data_move(index_ub, data_index[offset], 0, 1, (roi_per_core + 7) // 8, 0, 0)
                self.compute_per_core(roi_per_core, offset, index_reg, inds_ub, data_each_block,
                                      idx_tmp, idx, roi_buf, roi_feats_buf, index_ub)

            with self.tik.else_scope():
                inds_ub = self.tik.Tensor("int32", (ceil_div(last_roi, 128) * 128,),
                                          name="inds_ub", scope=tik.scope_ubuf)
                if index is not None:
                    index_ub = self.tik.Tensor("int32", (last_roi,), name="index_ub", scope=tik.scope_ubuf)
                    self.tik.data_move(index_ub, data_index[offset], 0, 1, (last_roi + 7) // 8, 0, 0)
                self.compute_per_core(last_roi, offset, index_reg, inds_ub, data_each_block,
                                      idx_tmp, idx, roi_buf_last, roi_feats_buf, index_ub)

        # Build CCE
        if index is not None:
            sch_list = self.feats_gm_list + [self.rois_gm, data_index]
        else:
            sch_list = self.feats_gm_list + [self.rois_gm]
        opt_config = {
            "enable_const_fold": True,
            "out_of_bound_sync_check": False
        }
        self.tik.BuildCCE(kernel_name=self.kernel_name,
                          inputs=sch_list,
                          outputs=[self.roi_feats_gm],
                          flowtable=(self.tiling_gm,),
                          config=opt_config)
        return self.tik

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def compute_per_core(self, roi_per_core, offset, index_reg, inds_ub, data_each_block,
                         idx_tmp, idx, roi_buf, roi_feats_buf, index_ub):
        """
        Roi align compute process.
        """
        target_lvls = self.tik.Tensor("float16", (ceil_div(roi_per_core, 128) * 128,),
                                      name="target_lvls_ub", scope=tik.scope_ubuf)
        self.map_roi_levels(target_lvls, roi_per_core, offset)

        for fm_idx in range(self.num_levels):
            self.where_and_nonzero(target_lvls, index_reg, inds_ub, fm_idx, roi_per_core)
            with self.tik.new_stmt_scope():
                rois_ub = self.tik.Tensor(self.rois_dtype, (data_each_block,),
                                          name="rois_ub", scope=tik.scope_ubuf)
                with self.tik.for_range(0, index_reg) as roi_i:
                    idx_tmp.set_as(inds_ub[roi_i])
                    idx.set_as(idx_tmp + offset)
                    self.tik.data_move(rois_ub, self.rois_gm[idx * self.rois_last_dim], 0, 1, 1, 0, 0)
                    self.tik.data_move(roi_buf[(fm_idx * self.rois_total_num + roi_i + offset) * 5],
                                       rois_ub, 0, 1, 1, 0, 0)

            with self.tik.new_stmt_scope():
                with self.tik.if_scope(index_reg > 0):
                    output_offset_start = self._get_output_offset(fm_idx * self.rois_total_num + offset, 0, 0, 0)
                    output_offset_end = self._get_output_offset(fm_idx * self.rois_total_num +
                                                                offset + roi_per_core, 0, 0, 0)
                    rac = RoiAlignCompute(self,
                                          self.feats_gm_list[fm_idx],
                                          roi_buf[(fm_idx * self.rois_total_num + offset) * 5:
                                                  (fm_idx * self.rois_total_num + offset + roi_per_core) * 5],
                                          roi_feats_buf[output_offset_start: output_offset_end],
                                          0,
                                          index_reg,
                                          fm_idx)
                    rac.roi_align_compute()

            with self.tik.new_stmt_scope():
                out_ub = self.tik.Tensor(self.rois_dtype, (self.y_shape_last_4_dim,),
                                         name="out_ub", scope=tik.scope_ubuf)
                with self.tik.for_range(0, index_reg) as roi_i:
                    idx_tmp.set_as(inds_ub[roi_i])
                    if index_ub is not None:
                        idx.set_as(index_ub[idx_tmp])
                    else:
                        idx.set_as(idx_tmp + offset)

                    output_buf_offset = self._get_output_offset(fm_idx * self.rois_total_num + roi_i + offset, 0, 0, 0)
                    output_offset = self._get_output_offset(idx, 0, 0, 0)
                    self.tik.data_move(out_ub, roi_feats_buf[output_buf_offset], 0, 1,
                                       ceil_div(self.y_shape_last_4_dim, data_each_block), 0, 0)
                    self.tik.data_move(self.roi_feats_gm[output_offset], out_ub, 0, 1,
                                       ceil_div(self.y_shape_last_4_dim, data_each_block), 0, 0)

    def map_roi_levels(self, target_lvls, proc_num, roi_offset=0):
        """
        Calculate `scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))`
        Return: target_lvls
        """
        x1_fp16 = self.tik.Tensor("float16", (Constant.ALIGN_LEN,), name="rois_x1_fp16", scope=tik.scope_ubuf)
        y1_fp16 = self.tik.Tensor("float16", (Constant.ALIGN_LEN,), name="rois_y1_fp16", scope=tik.scope_ubuf)
        x2_fp16 = self.tik.Tensor("float16", (Constant.ALIGN_LEN,), name="rois_x2_fp16", scope=tik.scope_ubuf)
        y2_fp16 = self.tik.Tensor("float16", (Constant.ALIGN_LEN,), name="rois_y2_fp16", scope=tik.scope_ubuf)
        x1_fp32 = self.tik.Tensor("float32", (Constant.ALIGN_LEN,),
                                  name="rois_x1_fp32_ub", scope=tik.scope_ubuf)
        y1_fp32 = self.tik.Tensor("float32", (Constant.ALIGN_LEN,),
                                  name="rois_y1_fp32_ub", scope=tik.scope_ubuf)
        x2_fp32 = self.tik.Tensor("float32", (Constant.ALIGN_LEN,),
                                  name="rois_x2_fp32_ub", scope=tik.scope_ubuf)
        y2_fp32 = self.tik.Tensor("float32", (Constant.ALIGN_LEN,),
                                  name="rois_y2_fp32_ub", scope=tik.scope_ubuf)
        rois_ub = self.tik.Tensor(self.rois_dtype, (Constant.ALIGN_LEN, 8),
                                  name="rois_ub", scope=tik.scope_ubuf)
        rois_ub_n5 = self.tik.Tensor(self.rois_dtype, (Constant.ALIGN_LEN, 5),
                                     name="rois_ub_n5", scope=tik.scope_ubuf)
        work_tensor0 = self.tik.Tensor("float32", (4 * Constant.ALIGN_LEN,),
                                       name="work_tensor0_ub", scope=tik.scope_ubuf)

        loop = self.tik.Scalar("int32", init_value=proc_num // Constant.ALIGN_LEN)
        tail = self.tik.Scalar("int32", init_value=proc_num % Constant.ALIGN_LEN)

        if self.rois_dtype == "float32":
            n_burst = 2
        else:
            n_burst = 1

        def inner_compute(offset):
            """
            Calculate the areas of rois.
            Convert dtype of inputs to fp32 when dtype of inputs is fp16 before calculating.
            """
            if self.rois_dtype == "float16":
                support_vextract = tbe_platform.api_check_support("tik.vextract", "float16")
                self.extract_roi(rois_ub, x1_fp16, y1_fp16, x2_fp16, y2_fp16, 128, support_vextract)
                self.tik.vec_conv(Constant.B32_MASK, "none", x1_fp32, x1_fp16, Constant.REPEAT_FP32, 8, 4)
                self.tik.vec_conv(Constant.B32_MASK, "none", y1_fp32, y1_fp16, Constant.REPEAT_FP32, 8, 4)
                self.tik.vec_conv(Constant.B32_MASK, "none", x2_fp32, x2_fp16, Constant.REPEAT_FP32, 8, 4)
                self.tik.vec_conv(Constant.B32_MASK, "none", y2_fp32, y2_fp16, Constant.REPEAT_FP32, 8, 4)
            else:
                support_vextract = tbe_platform.api_check_support("tik.vextract", "float32")
                self.extract_roi(rois_ub, x1_fp32, y1_fp32, x2_fp32, y2_fp32, 128, support_vextract)

            # Calc levels
            self.tik.vec_sub(Constant.B32_MASK, x1_fp32, x2_fp32, x1_fp32, Constant.REPEAT_FP32, 8, 8, 8)
            self.tik.vec_sub(Constant.B32_MASK, y1_fp32, y2_fp32, y1_fp32, Constant.REPEAT_FP32, 8, 8, 8)
            self.tik.vec_mul(Constant.B32_MASK, y1_fp32, x1_fp32, y1_fp32, Constant.REPEAT_FP32, 8, 8, 8)
            self.tik.vec_rsqrt_high_preci(Constant.B32_MASK, x1_fp32, y1_fp32,
                                          work_tensor0[0:], Constant.REPEAT_FP32, 8, 8)
            self.tik.vec_rec_high_preci(Constant.B32_MASK, y1_fp32, x1_fp32,
                                        work_tensor0[0:], Constant.REPEAT_FP32, 8, 8)
            self.tik.vec_conv(Constant.B32_MASK, "none", target_lvls[offset:offset + Constant.ALIGN_LEN],
                              y1_fp32, Constant.REPEAT_FP32, 4, 8)

        with self.tik.if_scope(loop > 0):
            with self.tik.for_range(0, loop) as loop_i:
                self.tik.data_move(rois_ub_n5[0, 0], self.rois_gm[(loop_i * Constant.ALIGN_LEN + roi_offset) *
                                                                  self.rois_last_dim],
                                   0, 1, 40 * n_burst, 0, 0)
                self._tf_n52n8(rois_ub, rois_ub_n5, Constant.ALIGN_LEN)
                inner_compute(loop_i * Constant.ALIGN_LEN)
        with self.tik.if_scope(tail > 0):
            self.tik.data_move(rois_ub_n5[0, 0], self.rois_gm[(loop * Constant.ALIGN_LEN + roi_offset) *
                                                              self.rois_last_dim], 0, 1, 40 * n_burst, 0, 0)
            self._tf_n52n8(rois_ub, rois_ub_n5, tail)
            inner_compute(loop * Constant.ALIGN_LEN)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def extract_roi(self, rois_ub, x0_ub, y0_ub, x1_ub, y1_ub, roi_num, support_vextract):
        """
        Extract rois.
        """
        if not support_vextract:
            with self.tik.for_range(0, roi_num) as j:
                x0_ub[j].set_as(rois_ub[j, 1])
                y0_ub[j].set_as(rois_ub[j, 2])
                x1_ub[j].set_as(rois_ub[j, 3])
                y1_ub[j].set_as(rois_ub[j, 4])
        else:
            self.tik.vextract(x0_ub[0], rois_ub[0], 8, 1)
            self.tik.vextract(y0_ub[0], rois_ub[0], 8, 2)
            self.tik.vextract(x1_ub[0], rois_ub[0], 8, 3)
            self.tik.vextract(y1_ub[0], rois_ub[0], 8, 4)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def where_and_nonzero(self, target_lvls, index_reg, inds_ub, fm_idx, proc_num):
        """
        Select roi.
        """
        align_len = self.tik.Scalar("int32", init_value=ceil_div(self.rois_total_num, 128) * 128)
        inds_buf = self.tik.Tensor("int32", (align_len, ), name="inds_buf", scope=tik.scope_ubuf)
        i_ub = self.tik.Tensor("float16", (128,), name="i_ub", scope=tik.scope_ubuf)
        i1_ub = self.tik.Tensor("float16", (128,), name="i1_ub", scope=tik.scope_ubuf)
        cmp_bit_size = 16
        cmp_ub = self.tik.Tensor("uint16", (align_len // cmp_bit_size,), name="cmp_ub", scope=tik.scope_ubuf)
        cmp1_ub = self.tik.Tensor("uint16", (align_len // cmp_bit_size,), name="cmp1_ub", scope=tik.scope_ubuf)
        one_ub = self.tik.Tensor("float16", (128,), name="one_ub", scope=tik.scope_ubuf)
        zero_ub = self.tik.Tensor("float16", (128,), name="zero_ub", scope=tik.scope_ubuf)
        self.tik.vector_dup(128, one_ub, 1, 1, 1, 8)
        self.tik.vector_dup(128, zero_ub, 0, 1, 1, 8)
        for i in self.index_arr:
            with self.tik.if_scope(fm_idx == i):
                self.tik.vector_dup(128, i_ub, self.pow_table[i], 1, 1, 8)
                self.tik.vector_dup(128, i1_ub, self.pow_table[i + 1], 1, 1, 8)

        repeat = self.tik.Scalar("int32", init_value=align_len // 128)
        with self.tik.if_scope(fm_idx == 0):
            self.tik.vec_cmpv_le(cmp_ub, target_lvls, i1_ub, repeat, 8, 0)
        with self.tik.else_scope():
            with self.tik.if_scope(fm_idx == self.num_levels - 1):
                self.tik.vec_cmpv_gt(cmp_ub, target_lvls, i_ub, repeat, 8, 0)
            with self.tik.else_scope():
                self.tik.vec_cmpv_gt(cmp_ub, target_lvls, i_ub, repeat, 8, 0)
                self.tik.vec_cmpv_le(cmp1_ub, target_lvls, i1_ub, repeat, 8, 0)
                self.tik_vand(cmp_ub, cmp_ub, cmp1_ub, align_len // cmp_bit_size)

        with self.tik.for_range(0, repeat) as i:
            self.tik.vec_sel(128, 0, i_ub, cmp_ub[128 // cmp_bit_size * i], one_ub, zero_ub, 1, 1, 8, 8)
            self.tik.vec_conv(Constant.B32_MASK, "floor", inds_buf[128 * i], i_ub, 2, 8, 4)
        idx = self.tik.Scalar("int32", name="roi_idx")
        index_reg.set_as(0)
        with self.tik.for_range(0, proc_num) as roi_i:
            idx.set_as(inds_buf[roi_i])
            with self.tik.if_scope(idx != 0):
                inds_ub[index_reg].set_as(roi_i)
                index_reg.set_as(index_reg + 1)

    def tik_vand(self, dst, src0, src1, data_len):
        """
        Tik command vand.
        """
        offset = self.tik.Scalar("int32", init_value=0)
        loop = self.tik.Scalar("int32", init_value=data_len // (128 * 255))
        tail = data_len % 128
        with self.tik.if_scope(loop > 0):
            with self.tik.for_range(0, loop) as loop_i:
                offset.set_as(loop_i * 128 * 255)
                self.tik.vec_and(128, dst[offset], src0[offset], src1[offset], 255, 8, 8, 8)
        repeat_t = self.tik.Scalar("int32", init_value=data_len // 128)
        with self.tik.if_scope(repeat_t > 0):
            self.tik.vec_and(128, dst[offset], src0[offset], src1[offset], repeat_t, 8, 8, 8)
        offset.set_as(offset + repeat_t * 128)
        with self.tik.if_scope(tail > 0):
            self.tik.vec_and(tail, dst[offset], src0[offset], src1[offset], 1, 8, 8, 8)

    def _tf_n52n8(self, rois_ub, rois_n5, block_num):
        """
        Transform Rois form N5 to N8.
        """
        with self.tik.for_range(0, block_num) as rois_num:
            rois_ub[rois_num, 1].set_as(rois_n5[rois_num, 1])
            rois_ub[rois_num, 2].set_as(rois_n5[rois_num, 2])
            rois_ub[rois_num, 3].set_as(rois_n5[rois_num, 3])
            rois_ub[rois_num, 4].set_as(rois_n5[rois_num, 4])

    # 'pylint: disable=invalid-name
    def _get_output_offset(self, n, c1, h, w):
        """
        Calculate output offset.
        """
        n_offset = n * self.feats_c1_dim * self.pooled_h * self.pooled_w * self.feats_c0_dim
        c1_offset = c1 * self.pooled_h * self.pooled_w * self.feats_c0_dim
        h_offset = h * self.pooled_w * self.feats_c0_dim
        w_offset = w * self.feats_c0_dim
        return n_offset + c1_offset + h_offset + w_offset


def ceil_div(value, factor):
    """
    Ceil div.
    """
    return (value + factor - 1) // factor


# 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_LIST_FLOAT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def roi_extractor(feats, rois, index=None, roi_feats=None, finest_scale=56, roi_scale_factor=0, spatial_scale=None,
                  pooled_h=7, pooled_w=7, sample_num=0, pool_mode='avg', aligned=True, kernel_name="roi_extractor"):
    """
    Interface of RoiExtractor.

    Parameters
    ----------
    feats : dict
        Shape and dtype of dynamic input feats.
    rois : dict
        Shape and dtype of input rois.
    index : dict
        Shape and dtype of optional input index.
    roi_feats : dict
        Shape and dtype of output roi_feats
    finest_scale : int
        specifying the scale of calculate levels of "rois"., default is 56.
    roi_scale_factor : float
        Specifying the rescaling of "rois" coordinates, default value is 0.
    spatial_scale : list
        Specifying the scaling ratio of "features" to the original image, default value is None.
    pooled_h : int
        specifying the num of H dimension, default value  is 7.
    pooled_w : int
        specifying the num of W dimension, default value  is 7.
    sample_num : int
        specifying the horizontal and vertical sampling frequency  of each output. 
        If this attribute is set to "0", the sampling frequency is equal to the 
        rounded up value of "rois", which is a floating point number, default value is 0.
    pool_mode : str
        Pooling mode, default value is "avg".
    aligned : bool
        Specifying the align to corner, default value is True.
    kernel_name : str
        Kernel name, default value is "roi_extractor".

    Returns
    -------
    Tik.instance
    """
    rpn_instance = RoiExtractor(feats, rois, index, roi_feats, finest_scale, roi_scale_factor,
                                spatial_scale, pooled_h, pooled_w, sample_num, pool_mode, aligned, kernel_name)
    return rpn_instance.compute(index)
