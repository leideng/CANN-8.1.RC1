#!/usr/bin/env python
# coding: utf-8
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
PSROIPoolingV2
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik

# data type of fp16
FP16 = "float16"
# data type of fp32
FP32 = "float32"
# data type of int32
INT32 = "int32"
# data type of int16
INT16 = "int16"
# one block size takes up 32b
BLOCK_SIZE = 32
# instruction's default sid is 0
SID = 0
# C0 is 16 in davinci
C0 = 16
# instruction mask 64
MASK64 = 64
# one burst
BURST_1 = 1
# default repeat time
REPEAT_1 = 1
# repeat 2 time
REPEAT_2 = 2
# repeat 4 time
REPEAT_4 = 4
# stride zero
STRIDE_ZERO = 0
# stride one
STRIDE_ONE = 1
# default repeat stride length
REP_STRIDE_EIGHT = 8
REP_STRIDE_FOUR = 4
# default deqscale in vconv instruction
DEQSCALE = 1.0
# the max stride of data move instruction
MAX_GAP_SIZE = 65536
# length of fp16 and fp32 data type
TYPE_LEN_DICT = {FP16: 2, FP32: 4}
# number of element of fp16 and fp32 data type in one block
BLOCK_ELEM_NUM = {FP16: 16, FP32: 8}
# number of element of fp16 and fp32 data type in one vector
VEC_ELEM_NUM = {FP16: 128, FP32: 64}
# repeat times of fp16 and fp32 data type in vconv instruction
REP_TIMES = {FP16: 2, FP32: 1}
# repeat stride of fp16 and fp32 data type in vconv instruction
REP_STRIDE = {FP16: 4, FP32: 8}

# digit 256
DIGIT_256 = 256
# digit 255
DIGIT_255 = 255
# digit 128
DIGIT_128 = 128
# digit 64
DIGIT_64 = 64
# digit 4
DIGIT_4 = 4
# digit 5
DIGIT_5 = 5
# digit 2
DIGIT_2 = 2
# digit 8
DIGIT_8 = 8
# 0.1
POINT_1 = 0.1
# 0.5
POINT_5 = 0.5
# 1.0
ONE_POINT = 1.0
# neg two
NEG_TWO = -2
# neg one
NEG_ONE = -1


def _ceil_value(value, factor):
    """
    if not divide exactly then plus 1

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    ceil value
    """
    return (value + factor - 1) // factor


# 'pylint: disable=invalid-name, too-many-locals, too-many-arguments
# 'pylint: disable=too-many-instance-attributes, too-many-lines
class PSROIPoolingV2Class(object):
    """
    Function: class that execute PSROIPoolingV2
    """

    def _input_param_check(self):
        """
        check if the inputs are valid

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        para_check.check_dtype(self.dtype, (FP16, FP32), param_name="x")
        para_check.check_dtype(self.roi_dtype, (FP16, FP32), param_name="rois")

        if self.dtype != self.roi_dtype or self.dtype != self.y_dtype:
            error_info = {'errCode': 'E80017'}
            raise RuntimeError(error_info, "In op[%s], the dtype of input x[%s], "
                                           "rois[%s] and y[%s] must be equal."
                               % ('PSROIPoolingV2', self.dtype, self.roi_dtype, self.y_dtype))

        para_check.check_shape(self.x_shape, param_name="x")
        para_check.check_shape(self.roi_shape, param_name="rois")
        para_check.check_shape(self.y_shape, param_name="y")
        # x and y must be 5HD
        para_check.check_shape(self.x_shape, min_rank=DIGIT_5, max_rank=DIGIT_5, param_name="x")
        para_check.check_shape(self.y_shape, min_rank=DIGIT_5, max_rank=DIGIT_5, param_name="y")

        if self.spatial_scale <= 0:
            raise RuntimeError('spatial_scale must be larger than zero.')

        if self.roi_shape[0] != self.x_shape[0]:
            raise RuntimeError("In op PSROIPoolingV2, the batch of input x[%s] and rois[%s] must be equal."
                               % (self.x_shape[0], self.roi_shape[0]))

        if self.roi_shape[1] != DIGIT_5:
            error_info = {'errCode': 'E80000'}
            raise RuntimeError(error_info, "In op[%s], the parameter [%s] must be equal 5, but actually is [%s]."
                               % ('PSROIPoolingV2', "roi_shape[1]", self.roi_shape[1]))

        if self.roi_shape[0] * self.roi_shape[2] != self.y_shape[0]:
            error_info = {'errCode': 'E81009'}
            raise RuntimeError(error_info, "In op[%s], all num of rois must be equal to "
                                           "y_shape[0][%s],but actually is [%s]."
                               % ('PSROIPoolingV2', self.y_shape[0], self.roi_shape[0] * self.roi_shape[2]))

        if self.group_size >= DIGIT_128:
            error_info = {'errCode': 'E80002'}
            raise RuntimeError(error_info, "In op[%s], the parameter[%s] must be "
                                           "less than [%s],but actually is [%s]."
                               % ('PSROIPoolingV2', 'group_size', DIGIT_128, self.group_size))

        if self.ori_x_shape[1] // self.ori_y_shape[1] != self.ori_y_shape[2] * self.ori_y_shape[3]:
            error_info = {'errCode': 'E81010'}
            raise RuntimeError(error_info, "In op PSROIPoolingV2, the parameter ori_x_shape[1] is invalid, "
                                           "it should follow the rule: ori_x_shape[1](%s)//ori_y_shape[1](%s) "
                                           "== ori_y_shape[2](%s) * ori_y_shape[3](%s)."
                               % (self.ori_x_shape[1], self.ori_y_shape[1], self.ori_y_shape[2],
                                  self.ori_y_shape[3]))

        if self.group_size != self.y_shape[2] or self.group_size != self.y_shape[3]:
            error_info = {'errCode': 'E80017'}
            raise RuntimeError(error_info, "In op[%s], the shape of y_shape[2] and y_shape[3] "
                                           "must be equal to group_size[%s],but actually is %s and %s."
                               % ('PSROIPoolingV2', self.group_size, self.y_shape[2],
                                  self.y_shape[3]))

        if _ceil_value(self.output_dim, C0) != self.y_shape[1]:
            error_info = {'errCode': 'E81011'}
            raise RuntimeError(error_info, "In op PSROIPoolingV2, the parameter output_dim is invalid,it should "
                                           "follow the rule:(output_dim + C0 -1) // C0 == y_shape[1]")

    def __init__(self, x_dict, rois_dict, y_dict, params, kernel_name):
        """
        constructor of PSROIPoolingV2Class

        Parameters
        ----------
        x_dict: dict describes input fm, NC1HWC0
        rois_dict: dict describes input rois
        params: a tuple, contain output_dim, group_size, spatial_scale
        kernel_name: name of kernel

        Returns
        -------
        None
        """
        self.x_shape = x_dict["shape"]
        self.dtype = x_dict["dtype"].lower()
        self.roi_dtype = rois_dict["dtype"].lower()
        self.roi_shape = rois_dict["shape"]
        self.y_dtype = y_dict["dtype"].lower()
        self.y_shape = y_dict["shape"]
        self.ori_x_shape = x_dict["ori_shape"]
        self.ori_y_shape = y_dict["ori_shape"]
        self.output_dim = params[0]
        self.group_size = params[1]
        self.spatial_scale = params[2]
        self.kernel_name = kernel_name

        profile = tik.Dprofile()

        product_name = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
        self._input_param_check()

        self.dsize = TYPE_LEN_DICT.get(self.dtype)
        self.fm_batch = self.x_shape[0]
        self.fm_c1 = self.x_shape[1]
        self.fm_h = self.x_shape[2]
        self.fm_w = self.x_shape[3]
        self.fm_c0 = self.x_shape[4]
        self.fm_c = self.fm_c1 * self.fm_c0
        self.hw = self.fm_h * self.fm_w
        self.x_data_size = (self.fm_batch * self.fm_c * self.hw) * self.dsize
        # roi num of one batch, roi_shape is (batch, 5, rois_num)
        self.roi_num_b = self.roi_shape[2]

        self.k2 = self.group_size * self.group_size
        self.vec_elem_num = VEC_ELEM_NUM.get(self.dtype)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        # divide the available UB space into four parts
        self.ub_one_buf = self.ub_size // 4
        self.ub_one_buf_elem = self.ub_one_buf // self.dsize
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.is_hisi_cs = False
        if self.dtype == FP16 and product_name in ("Hi3796CV300CS", "SD3403"):
            self.is_hisi_cs = True

        self.roi_num_step = self.vec_elem_num
        self.mask = self.roi_num_step

        # set parameters
        self.inner_c_size = (self.fm_c1 // self.k2) * C0 * self.dsize

        self.x = None
        self.rois = None
        self.y = None
        self.const_0_127_ub = None
        self.const_1_128_ub = None

        self.output_dim_align = _ceil_value(self.output_dim, self.mask) * self.mask
        self.output_dim_align_c0 = _ceil_value(self.output_dim, C0) * C0

    def _load_rois_to_ub(self, rois_ub, rois_offset, roi_step):
        """
        load rois data to ub from gm.

        Parameters
        ----------
        rois_ub: a tensor, which store rois data
        rois_offset: the roi offset of current loop in block_id aicore
        roi_step: the number of rois per loop in process, 64(fp32) or 128(fp16)

        Returns
        -------
        None
        """
        burst_len = roi_step * self.dsize // BLOCK_SIZE
        for i in range(DIGIT_5):
            self.tik_instance.data_move(rois_ub[i, 0], \
                                        self.rois[rois_offset + i * self.roi_num_b], SID, \
                                        BURST_1, burst_len, STRIDE_ZERO, STRIDE_ZERO)

    def _spatial_scale_rois(self, rois_ub, rois_floor_ub, rois_spatial_ub,
                            roi_step):
        """
        compute the width and height of rois and bin.

        Parameters
        ----------
        rois_ub: input rois data in ub, (5, roi_step).
            batch_id,batch_id,batch_id...
            x1,x1,x1...
            y1,y1,y1...
            x2,x2,x2...
            y2,y2,y2...
        rois_floor_ub: store rois data of convert to s32
        rois_spatial_ub: store the width and height of rois and bin in ub
        roi_step: the number of rois per loop in process, 64(fp32) or 128(fp16)

        Returns
        -------
        None
        """
        point_one_ub = self.tik_instance.Tensor(self.dtype, (roi_step,),
                                                name="point_one_ub",
                                                scope=tbe_platform.scope_ubuf)
        self.tik_instance.vadds(self.mask, rois_ub[1, 0], rois_ub[1, 0],
                                POINT_5, REPEAT_4, STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        # rois_floor_ub[0]: batch id; rois_floor_ub[1-4]: roi coordinates
        # vconv.floor: f162s32r or f322s32r
        if (tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in
            (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B)) and self.dtype == FP32:
            rois_fp16_ub = self.tik_instance.Tensor(FP16, (DIGIT_5, roi_step),
                                                    name="rois_fp16_ub",
                                                    scope=tbe_platform.scope_ubuf)
            self.tik_instance.vconv(MASK64, '', rois_fp16_ub, rois_ub,
                                    REP_TIMES.get(self.dtype) * DIGIT_5,
                                    STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE.get(FP16), REP_STRIDE.get(self.dtype))
            self.tik_instance.vconv(MASK64, 'floor', rois_floor_ub, rois_fp16_ub,
                                    REP_TIMES.get(self.dtype) * DIGIT_5,
                                    STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE.get(self.dtype), REP_STRIDE.get(FP16))
            self._bias_f32_f16_int32(rois_ub, rois_floor_ub, rois_floor_ub, "floor",
                                     REP_TIMES.get(self.dtype) * DIGIT_5)
        else:
            self.tik_instance.vconv(MASK64, 'floor', rois_floor_ub, rois_ub,
                                    REP_TIMES.get(self.dtype) * DIGIT_5,
                                    STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_EIGHT, REP_STRIDE.get(self.dtype))
        # s322f16: vconv.deq, or s322f32: vconv
        if self.dtype == FP16:
            if self.is_hisi_cs:
                # s322s16:vcbd, and s162f16:vconv
                rois_floor_ub_int16 = self.tik_instance.Tensor(INT16, (DIGIT_4, roi_step), \
                                                               name="rois_floor_ub_int16",
                                                               scope=tbe_platform.scope_ubuf)
                self.tik_instance.vcbd(MASK64, rois_floor_ub_int16, rois_floor_ub[1, 0],
                                       REPEAT_2 * DIGIT_4, STRIDE_ONE, STRIDE_ONE,
                                       REP_STRIDE.get(self.dtype), REP_STRIDE_EIGHT)
                self.tik_instance.vconv(DIGIT_128, '', rois_spatial_ub,
                                        rois_floor_ub_int16, REPEAT_1 * DIGIT_4, STRIDE_ONE,
                                        STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            else:
                self.tik_instance.vconv(MASK64, '', rois_spatial_ub,
                                        rois_floor_ub[1, 0],
                                        REP_TIMES.get(self.dtype) * DIGIT_4,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE.get(self.dtype), REP_STRIDE_EIGHT,
                                        deqscale=DEQSCALE)
        else:
            if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in \
                    (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                rois_floor_fp16_ub = self.tik_instance.Tensor(FP16, (DIGIT_5, roi_step),
                                                              name="rois_floor_fp16_ub",
                                                              scope=tbe_platform.scope_ubuf)
                self.tik_instance.vconv(MASK64, '', rois_floor_fp16_ub, rois_floor_ub,
                                        REP_TIMES.get(self.dtype) * DIGIT_5,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE.get(FP16), REP_STRIDE.get(self.dtype), deqscale=1.0)
                self.tik_instance.vconv(MASK64, '', rois_spatial_ub,
                                        rois_floor_fp16_ub[1, 0],
                                        REP_TIMES.get(self.dtype) * DIGIT_4,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE.get(self.dtype), REP_STRIDE.get(FP16))
            else:
                self.tik_instance.vconv(MASK64, '', rois_spatial_ub,
                                        rois_floor_ub[1, 0],
                                        REP_TIMES.get(self.dtype) * DIGIT_4,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE.get(self.dtype), REP_STRIDE_EIGHT)
        self.tik_instance.vadds(self.mask, rois_spatial_ub[2, 0],
                                rois_spatial_ub[2, 0], ONE_POINT, REPEAT_2,
                                STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        # multiply spatial
        self.tik_instance.vmuls(self.mask, rois_spatial_ub, rois_spatial_ub,
                                self.spatial_scale, REPEAT_4, STRIDE_ONE,
                                STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)

        # roi width and height: roi_end_w-roi_start_w, roi_end_h-roi_start_h
        self.tik_instance.vsub(self.mask, rois_spatial_ub[4, 0],
                               rois_spatial_ub[2, 0], rois_spatial_ub, REPEAT_2,
                               STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                               REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                               REP_STRIDE_EIGHT)
        self.tik_instance.vector_dup(self.mask, point_one_ub, POINT_1, REPEAT_1,
                                     STRIDE_ONE, REP_STRIDE_EIGHT)
        self.tik_instance.vmax(self.mask, rois_spatial_ub[4, 0],
                               rois_spatial_ub[4, 0], point_one_ub, REPEAT_2,
                               STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                               REP_STRIDE_EIGHT, REP_STRIDE_EIGHT, STRIDE_ZERO)

        pooled_k_recip = self.tik_instance.Scalar(self.dtype,
                                                  name="pooled_k_recip", init_value=self.group_size)
        self.tik_instance.vector_dup(self.mask, point_one_ub, pooled_k_recip, REPEAT_1,
                                     STRIDE_ONE, REP_STRIDE_EIGHT)
        # bin width and height
        self._newton_div(rois_spatial_ub[6, :], rois_spatial_ub[4, :], point_one_ub, REPEAT_1)
        self._newton_div(rois_spatial_ub[7, :], rois_spatial_ub[5, :], point_one_ub, REPEAT_1)

    def _newton_div(self, dst, divisor, dividend, repeat):
        """
        use newton_div to improve performance

        Parameters
        ----------
        dst: vdiv's dest tensor
        divisor: vdiv's src0 tensor
        dividend: vdiv's src1 tensor
        repeat: vdiv's needs repeat times

        Returns
        -------
        None
        """
        if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in (tbe_platform.ASCEND_910, "Ascend910B",
                                                                   tbe_platform.ASCEND_310P, tbe_platform.ASCEND_610):
            self.tik_instance.vdiv(self.mask, dst, divisor, dividend, repeat,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
        else:
            with self.tik_instance.new_stmt_scope():
                if self.dtype == FP16:
                    divisor_f32 = self.tik_instance.Tensor(FP32, divisor.shape, name="divisor_f32",
                                                           scope=tbe_platform.scope_ubuf)
                    dividend_f32 = self.tik_instance.Tensor(FP32, dividend.shape, name="dividend_f32",
                                                            scope=tbe_platform.scope_ubuf)
                    rec_f32 = self.tik_instance.Tensor(FP32, dividend.shape, name="rec_f32",
                                                       scope=tbe_platform.scope_ubuf)
                    self.tik_instance.vconv(MASK64, '', divisor_f32, divisor,
                                            DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE_EIGHT, REP_STRIDE_FOUR)
                    self.tik_instance.vconv(MASK64, '', dividend_f32, dividend,
                                            DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE_EIGHT, REP_STRIDE_FOUR)
                    self.tik_instance.vrec(MASK64, rec_f32, dividend_f32,
                                           DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                           REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)

                    # Newton start
                    self.tik_instance.vmul(MASK64, dividend_f32, dividend_f32, rec_f32,
                                           DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                           STRIDE_ONE, REP_STRIDE_EIGHT,
                                           REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                    self.tik_instance.vadds(MASK64, dividend_f32, dividend_f32, NEG_TWO,
                                            DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                    self.tik_instance.vmul(MASK64, dividend_f32, dividend_f32, rec_f32,
                                           DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                           STRIDE_ONE, REP_STRIDE_EIGHT,
                                           REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                    self.tik_instance.vmuls(MASK64, dividend_f32, dividend_f32, NEG_ONE,
                                            DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                    # Newton end
                    self.tik_instance.vmul(MASK64, divisor_f32, divisor_f32, dividend_f32,
                                           DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                           STRIDE_ONE, REP_STRIDE_EIGHT,
                                           REP_STRIDE_EIGHT, STRIDE_ZERO)
                    self.tik_instance.vconv(MASK64, '', dst, divisor_f32,
                                            DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE_FOUR, REP_STRIDE_EIGHT)
                else:
                    rec_f32 = self.tik_instance.Tensor(FP32, dividend.shape, name="rec_f32",
                                                       scope=tbe_platform.scope_ubuf)
                    dividend_copy = self.tik_instance.Tensor(self.dtype, dividend.shape,
                                                             name="dividend_copy",
                                                             scope=tbe_platform.scope_ubuf)
                    self.tik_instance.vrec(self.mask, rec_f32, dividend, repeat,
                                           STRIDE_ONE, STRIDE_ONE,
                                           REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                    self.tik_instance.data_move(dividend_copy, dividend, SID, BURST_1,
                                                dividend.shape[0] // BLOCK_ELEM_NUM.get(self.dtype),
                                                STRIDE_ZERO, STRIDE_ZERO)

                    # Newton start
                    self.tik_instance.vmul(self.mask, dividend_copy, dividend_copy, rec_f32,
                                           repeat, STRIDE_ONE, STRIDE_ONE,
                                           STRIDE_ONE, REP_STRIDE_EIGHT,
                                           REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                    self.tik_instance.vadds(self.mask, dividend_copy, dividend_copy, NEG_TWO,
                                            repeat, STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                    self.tik_instance.vmul(self.mask, dividend_copy, dividend_copy, rec_f32,
                                           repeat, STRIDE_ONE, STRIDE_ONE,
                                           STRIDE_ONE, REP_STRIDE_EIGHT,
                                           REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                    self.tik_instance.vmuls(self.mask, dividend_copy, dividend_copy, NEG_ONE,
                                            repeat, STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                    # Newton end

                    self.tik_instance.vmul(self.mask, dst, divisor, dividend_copy,
                                           repeat, STRIDE_ONE, STRIDE_ONE,
                                           STRIDE_ONE, REP_STRIDE_EIGHT,
                                           REP_STRIDE_EIGHT, STRIDE_ZERO)

    def _process_one_bin_2(self, params):
        """
        process one bin of roi: inner_c1 == 1, or
                                (bin_all_dsize > self.ub_one_buf, or
                                 bin_load_stride > MAX_GAP_SIZE)
        Parameters
        ----------
        params: param is a dict, contains multiple keys

        Returns
        -------
        None
        """
        bursts_s = self.tik_instance.Scalar(INT32, name="bursts_s")
        bursts_s.set_as(params["h_width"])
        burst_len_s = self.tik_instance.Scalar(INT32, name="burst_len_s")
        burst_len_s.set_as(params["w_width"] * C0 * self.dsize // BLOCK_SIZE)
        src_stride_s = self.tik_instance.Scalar(INT32, name="src_stride_s")
        src_stride_s.set_as(
            (self.fm_w - params["w_width"]) * C0 * self.dsize // BLOCK_SIZE)
        ub_output_dim = params["ub_output_dim"]
        output_dim_index = params["output_dim_bias"] + params["bin_i_offset"]
        bin_i_offset_c1 = output_dim_index // C0
        bin_i_offset_c0 = output_dim_index - bin_i_offset_c1 * C0

        x_src = self.x
        load_count_ceil = _ceil_value(C0, self.vec_elem_num)
        load_count_align = load_count_ceil * self.vec_elem_num
        ub_bin_input_buf = self.tik_instance.Tensor(self.dtype, \
                                                    (self.ub_one_buf_elem,), name="ub_bin_input_buf", \
                                                    scope=tbe_platform.scope_ubuf)
        ub_bin_output_buf = self.tik_instance.Tensor(self.dtype, \
                                                     (load_count_align,), name="ub_bin_output_buf", \
                                                     scope=tbe_platform.scope_ubuf)
        self.tik_instance.vector_dup(self.mask, ub_bin_output_buf, 0,
                                     load_count_ceil, STRIDE_ONE,
                                     REP_STRIDE_EIGHT)
        # move feature map from gm to ub
        self.tik_instance.data_move(ub_bin_input_buf, \
                                    x_src[params["scalar_roi_batch_id"],
                                          bin_i_offset_c1,
                                          params["h_start"], params["w_start"], bin_i_offset_c0], \
                                    SID, bursts_s, burst_len_s, src_stride_s, STRIDE_ZERO)

        # avg pooling
        src0_rep_stride = C0 * self.dsize // BLOCK_SIZE
        if (tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in
            (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B)) and self.dtype == FP32:
            with self.tik_instance.for_range(0, params["bin_area"]) as rep_i:
                self.tik_instance.vadd([0, 0x1], ub_bin_output_buf, \
                                       ub_bin_input_buf[src0_rep_stride * BLOCK_ELEM_NUM.get(self.dtype) * rep_i],
                                       ub_bin_output_buf, \
                                       1, STRIDE_ONE, STRIDE_ONE, \
                                       STRIDE_ONE, STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)
        else:
            with self.tik_instance.if_scope(params["bin_area"] < DIGIT_256):
                self.tik_instance.vadd([0, 0x1], ub_bin_output_buf, \
                                       ub_bin_input_buf, ub_bin_output_buf, \
                                       params["bin_area"], STRIDE_ONE, STRIDE_ONE, \
                                       STRIDE_ONE, STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)
            with self.tik_instance.else_scope():
                tail = params["bin_area"] % DIGIT_255
                times = params["bin_area"] // DIGIT_255
                with self.tik_instance.for_range(0, times) as times_i:
                    self.tik_instance.vadd([0, 0x1], ub_bin_output_buf, \
                                           ub_bin_input_buf[C0 * DIGIT_255 * times_i], \
                                           ub_bin_output_buf, DIGIT_255, \
                                           STRIDE_ONE, STRIDE_ONE, STRIDE_ONE, \
                                           STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)
                with self.tik_instance.if_scope(tail != 0):
                    self.tik_instance.vadd([0, 0x1], ub_bin_output_buf, \
                                           ub_bin_input_buf[C0 * DIGIT_255 * times], \
                                           ub_bin_output_buf, tail, \
                                           STRIDE_ONE, STRIDE_ONE, STRIDE_ONE, \
                                           STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)

        bin_area_fp_ub = params["bin_area_fp_ub"]
        self._newton_div(ub_bin_output_buf, ub_bin_output_buf, bin_area_fp_ub, load_count_ceil)

        output_val = self.tik_instance.Scalar(self.dtype, name="output_val")
        output_val.set_as(ub_bin_output_buf[0])
        mask = params["output_dim_mask"]
        ub_output_dim[mask] = output_val

    def _process_one_bin_area_positive(self, params):
        """
        process one bin of roi when bin area is positive.

        Parameters
        ----------
        params: param is a dict, contains multiple keys

        Returns
        -------
        None
        """
        bin_area_fp_ub = self.tik_instance.Tensor(self.dtype,
                                                  (self.vec_elem_num,), name="bin_area_fp_ub",
                                                  scope=tbe_platform.scope_ubuf)
        if self.is_hisi_cs:
            bin_area_16_scalar = self.tik_instance.Scalar(INT16, name="bin_area_16_scalar")
            bin_area_16_scalar.set_as(params.get("bin_area"))
            bin_area_int16 = self.tik_instance.Tensor(
                INT16, (DIGIT_128,), name="bin_area_int16", scope=tbe_platform.scope_ubuf)
            self.tik_instance.vector_dup(DIGIT_128, bin_area_int16,
                                          bin_area_16_scalar, REPEAT_1, STRIDE_ONE, REP_STRIDE_EIGHT)
            self.tik_instance.vconv(DIGIT_128, '', bin_area_fp_ub,
                                    bin_area_int16, REPEAT_1, STRIDE_ONE,
                                    STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        else:
            bin_area_int32 = self.tik_instance.Tensor(INT32,
                                                      (DIGIT_64 * REP_TIMES.get(self.dtype),),
                                                      name="bin_area_int32",
                                                      scope=tbe_platform.scope_ubuf)
            self.tik_instance.vector_dup(MASK64, bin_area_int32,
                                          params.get("bin_area"), REP_TIMES.get(self.dtype), STRIDE_ONE,
                                          REP_STRIDE_EIGHT)
            # s322f16:vconv.deq, or s322f32:vconv
            if self.dtype == FP16:
                self.tik_instance.vconv(MASK64, '', bin_area_fp_ub,
                                        bin_area_int32, REP_TIMES.get(self.dtype),
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE.get(self.dtype),
                                        REP_STRIDE_EIGHT, deqscale=DEQSCALE)
            else:
                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    bin_area_fp16_ub = \
                        self.tik_instance.Tensor(FP16, (self.vec_elem_num,), name="bin_area_fp16_ub",
                                                  scope=tbe_platform.scope_ubuf)
                    self.tik_instance.vconv(MASK64, '', bin_area_fp16_ub,
                                            bin_area_int32, REP_TIMES.get(self.dtype),
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE.get(FP16), REP_STRIDE_EIGHT,
                                            deqscale=DEQSCALE)
                    self.tik_instance.vconv(MASK64, '', bin_area_fp_ub,
                                            bin_area_fp16_ub, REP_TIMES.get(self.dtype),
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE.get(FP16))
                else:
                    self.tik_instance.vconv(MASK64, '', bin_area_fp_ub,
                                            bin_area_int32, REP_TIMES.get(self.dtype),
                                            STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE.get(self.dtype),
                                            REP_STRIDE_EIGHT)
        params["bin_area_fp_ub"] = bin_area_fp_ub

    def _process_one_bin(self, params):
        """
        process one bin of roi.

        Parameters
        ----------
        params: param is a dict, contains multiple keys

        Returns
        -------
        None
        """
        output_dim_shape = (self.output_dim_align,)
        ub_output_dim = self.tik_instance.Tensor(self.dtype, output_dim_shape,
                                                 name="ub_output_dim",
                                                 scope=tbe_platform.scope_ubuf)
        self.tik_instance.vmuls(self.mask, ub_output_dim, ub_output_dim, 0.,
                                self.output_dim_align // self.mask,
                                STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        params["ub_output_dim"] = ub_output_dim

        with self.tik_instance.if_scope(params["bin_area"] > 0):
            self._process_one_bin_area_positive(params)
            if self.output_dim <= 1:
                thread_num = 1
            else:
                thread_num = 2
            with self.tik_instance.for_range(0, self.output_dim, thread_num=thread_num) as out_dim:
                params["output_dim_mask"] = out_dim
                params["output_dim_bias"] = out_dim * self.k2
                self._process_one_bin_2(params)
        with self.tik_instance.else_scope():
            pass

        burst_len = C0 * self.dsize // BLOCK_SIZE
        self.tik_instance.data_move(self.y[params["cur_roi_output_offset"], 0, params["ph"], params["pw"], 0],
                                    ub_output_dim, SID, _ceil_value(self.output_dim_align_c0, C0),
                                    burst_len, STRIDE_ZERO, (self.k2 - 1) * burst_len)

    def _process_one_roi(self, params):
        """
        process one roi.

        Parameters
        ----------
        params: param is a dict, contains multiple keys

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.group_size) as ph:
            params["ph"] = ph
            # h coordinates of bin
            h_start = self.tik_instance.Scalar(INT32, name="h_start")
            h_start.set_as(params["bin_start_h_floor"][ph])
            h_end = self.tik_instance.Scalar(INT32, name="h_end")
            h_end.set_as(params["bin_end_h_ceil"][ph])
            params["h_start"] = h_start
            params["h_end"] = h_end

            with self.tik_instance.for_range(0, self.group_size) as pw:
                params["pw"] = pw
                # w coordinates of bin
                w_start = self.tik_instance.Scalar(INT32, name="w_start")
                w_start.set_as(params["bin_start_w_floor"][pw])
                w_end = self.tik_instance.Scalar(INT32, name="w_end")
                w_end.set_as(params["bin_end_w_ceil"][pw])
                params["w_start"] = w_start
                params["w_end"] = w_end

                bin_i_offset = self.tik_instance.Scalar(INT32,
                                                        name="bin_i_offset")
                # bin_i offset of in roi, 0~(group_size^2-1)
                bin_i_offset.set_as(ph * self.group_size + pw)
                params["bin_i_offset"] = bin_i_offset % self.k2

                w_width = self.tik_instance.Scalar(INT32, name="w_width")
                h_width = self.tik_instance.Scalar(INT32, name="h_width")
                bin_area = self.tik_instance.Scalar(INT32, name="bin_area")
                w_width.set_as(w_end - w_start)
                with self.tik_instance.if_scope(w_end <= w_start):
                    w_width.set_as(0)
                h_width.set_as(h_end - h_start)
                with self.tik_instance.if_scope(h_end <= h_start):
                    h_width.set_as(0)
                bin_area.set_as(w_width * h_width)
                params["w_width"] = w_width
                params["h_width"] = h_width
                params["bin_area"] = bin_area

                bin_all_dsize = self.tik_instance.Scalar(INT32,
                                                         name="bin_all_dsize")
                bin_all_dsize.set_as(bin_area * self.inner_c_size)
                bin_c0_dsize = self.tik_instance.Scalar(INT32,
                                                        name="bin_c0_dsize")
                bin_c0_dsize.set_as(bin_area * C0 * self.dsize)
                params["bin_all_dsize"] = bin_all_dsize
                params["bin_c0_dsize"] = bin_c0_dsize

                self._process_one_bin(params)

    def _bias_f32_f16_int32(self, in_f32, in_int32, out_int32, mode, repeat_time):
        temp_f32 = self.tik_instance.Tensor(FP32, in_f32.shape,
                                            name="temp_f32",
                                            scope=tbe_platform.scope_ubuf)
        temp_f16 = self.tik_instance.Tensor(FP16, in_f32.shape,
                                            name="temp_f16",
                                            scope=tbe_platform.scope_ubuf)
        bin_error = self.tik_instance.Tensor(INT32, out_int32.shape,
                                             name="bin_error",
                                             scope=tbe_platform.scope_ubuf)
        # int32 to f16
        self.tik_instance.vconv(MASK64, '', temp_f16, in_int32,
                                repeat_time, STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_FOUR, REP_STRIDE_EIGHT, 1.0)
        # f16 to f32
        self.tik_instance.vconv(MASK64, '', temp_f32, temp_f16,
                                repeat_time, STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_FOUR)
        # f32 error after vconv
        self.tik_instance.vsub(MASK64, temp_f32, in_f32, temp_f32,
                               repeat_time,
                               STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                               REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                               REP_STRIDE_EIGHT)
        # f32 error to f16 error
        self.tik_instance.vconv(MASK64, '', temp_f16, temp_f32,
                                repeat_time, STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_FOUR, REP_STRIDE_EIGHT)
        # round error for accuracy compensation
        self.tik_instance.vconv(MASK64, mode, bin_error, temp_f16,
                                repeat_time, STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_FOUR)

        self.tik_instance.vadd(MASK64, out_int32, out_int32, bin_error,
                               repeat_time,
                               STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                               REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                               REP_STRIDE_EIGHT)

    def _process_step1_roi(self, rois_floor_ub, rois_spatial_ub,
                           rois_num_offset, step_i_offset, step_i_num):
        """
        process roi_num roi cyclically, and process roi_step roi each time.

        Parameters
        ----------
        rois_floor_ub: rois data, s32
        rois_spatial_ub: a tensor, the width and height of rois and bin
        step_i_offset: the roi offset of this loop in block_id aicore
        rois_num_offset: a Scalar, the offset in block_id aicore
        step_i_num: the number of rois of one loop in process

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, step_i_num) as roi_i:
            params = {}
            scalar_roi_batch_id = self.tik_instance.Scalar(INT32, \
                                                           name="scalar_roi_batch_id")
            scalar_roi_batch_id.set_as(rois_floor_ub[0, roi_i])
            params["scalar_roi_batch_id"] = scalar_roi_batch_id

            cur_roi_output_offset = self.tik_instance.Scalar(INT32, \
                                                             name="cur_roi_output_offset")
            cur_roi_output_offset.set_as(rois_num_offset + step_i_offset +
                                         roi_i)
            params["cur_roi_output_offset"] = cur_roi_output_offset

            scalar_bin_width = self.tik_instance.Scalar(self.dtype, name="scalar_bin_width")
            scalar_bin_width.set_as(rois_spatial_ub[6, roi_i])
            scalar_bin_height = self.tik_instance.Scalar(self.dtype, name="scalar_bin_height")
            scalar_bin_height.set_as(rois_spatial_ub[7, roi_i])

            bin_start_w_floor = self.tik_instance.Tensor(INT32, (DIGIT_128,), \
                                                         name="bin_start_w_floor", scope=tbe_platform.scope_ubuf)
            bin_end_w_ceil = self.tik_instance.Tensor(INT32, (DIGIT_128,), \
                                                      name="bin_end_w_ceil", scope=tbe_platform.scope_ubuf)
            bin_start_h_floor = self.tik_instance.Tensor(INT32, (DIGIT_128,), \
                                                         name="bin_start_h_floor", scope=tbe_platform.scope_ubuf)
            bin_end_h_ceil = self.tik_instance.Tensor(INT32, (DIGIT_128,), \
                                                      name="bin_end_h_ceil", scope=tbe_platform.scope_ubuf)
            # `vmax(,0)`
            dup_tmp_ub = self.tik_instance.Tensor(INT32, (DIGIT_64,), \
                                                  name="dup_tmp_ub", scope=tbe_platform.scope_ubuf)

            if self.dtype == FP16:
                rois_spatial_fp32_ub = self.tik_instance.Tensor(FP32, (DIGIT_8, DIGIT_128),
                                                                name="rois_spatial_fp32_ub",
                                                                scope=tbe_platform.scope_ubuf)
                self.tik_instance.vconv(MASK64, '', rois_spatial_fp32_ub, rois_spatial_ub,
                                        DIGIT_8 * DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                        REP_STRIDE_FOUR)

                scalar_roi_start_w = self.tik_instance.Scalar(FP32, name="scalar_roi_start_w")
                scalar_roi_start_w.set_as(rois_spatial_fp32_ub[0, roi_i])
                scalar_roi_start_h = self.tik_instance.Scalar(FP32, name="scalar_roi_start_h")
                scalar_roi_start_h.set_as(rois_spatial_fp32_ub[1, roi_i])

                bin_start_w_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_128,),
                                                          name="bin_start_w_ub", scope=tbe_platform.scope_ubuf)
                bin_start_w_ub_fp32 = self.tik_instance.Tensor(FP32, (DIGIT_128,),
                                                               name="bin_start_w_ub_fp32",
                                                               scope=tbe_platform.scope_ubuf)
                # get start width floor
                self.tik_instance.vmuls(self.mask, bin_start_w_ub,
                                        self.const_0_127_ub, scalar_bin_width,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                self.tik_instance.vconv(MASK64, '', bin_start_w_ub_fp32, bin_start_w_ub,
                                        DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                        REP_STRIDE_FOUR)
                self.tik_instance.vadds(MASK64, bin_start_w_ub_fp32, bin_start_w_ub_fp32,
                                        scalar_roi_start_w,
                                        DIGIT_128 // DIGIT_64,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                # vconv.floor: f162s32f or f322s32f
                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) not in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    self.tik_instance.vconv(MASK64, 'floor', bin_start_w_floor,
                                            bin_start_w_ub_fp32, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE_EIGHT)
                else:
                    temp_f16_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_128,),
                                                           name="temp_f16_ub", scope=tbe_platform.scope_ubuf)
                    self.tik_instance.vconv(MASK64, '', temp_f16_ub, bin_start_w_ub_fp32,
                                            DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE_FOUR, REP_STRIDE_EIGHT)
                    self.tik_instance.vconv(MASK64, 'floor', bin_start_w_floor,
                                            temp_f16_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE_FOUR)
                    self._bias_f32_f16_int32(bin_start_w_ub_fp32, bin_start_w_floor,
                                             bin_start_w_floor, 'floor', DIGIT_128 // DIGIT_64)
                # get end width ceil
                self.tik_instance.vmuls(self.mask, bin_start_w_ub,
                                        self.const_1_128_ub, scalar_bin_width,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                self.tik_instance.vconv(MASK64, '', bin_start_w_ub_fp32, bin_start_w_ub,
                                        DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                        REP_STRIDE_FOUR)
                self.tik_instance.vadds(MASK64, bin_start_w_ub_fp32, bin_start_w_ub_fp32,
                                        scalar_roi_start_w,
                                        DIGIT_128 // DIGIT_64,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                # vconv.floor: f162s32f or f322s32f
                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) not in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    self.tik_instance.vconv(MASK64, 'ceil', bin_end_w_ceil,
                                            bin_start_w_ub_fp32, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE_EIGHT)
                else:
                    self.tik_instance.vconv(MASK64, '', temp_f16_ub, bin_start_w_ub_fp32,
                                            DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE_FOUR, REP_STRIDE_EIGHT)
                    self.tik_instance.vconv(MASK64, 'ceil', bin_end_w_ceil,
                                            temp_f16_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE_FOUR)
                    self._bias_f32_f16_int32(bin_start_w_ub_fp32, bin_end_w_ceil,
                                             bin_end_w_ceil, 'ceil', DIGIT_128 // DIGIT_64)

                bin_start_h_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_128,),
                                                          name="bin_start_h_ub", scope=tbe_platform.scope_ubuf)
                bin_start_h_ub_fp32 = self.tik_instance.Tensor(FP32, (DIGIT_128,),
                                                               name="bin_start_h_ub_fp32",
                                                               scope=tbe_platform.scope_ubuf)
                # scalar_roi_start_h + scalar_bin_height*(0...127)
                self.tik_instance.vmuls(self.mask, bin_start_h_ub,
                                        self.const_0_127_ub, scalar_bin_height,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                self.tik_instance.vconv(MASK64, '', bin_start_h_ub_fp32, bin_start_h_ub,
                                        DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                        REP_STRIDE_FOUR)
                self.tik_instance.vadds(MASK64, bin_start_h_ub_fp32, bin_start_h_ub_fp32,
                                        scalar_roi_start_h, DIGIT_128 // DIGIT_64,
                                        STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                # vconv.floor: f162s32f or f322s32f
                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) not in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    self.tik_instance.vconv(MASK64, 'floor', bin_start_h_floor,
                                            bin_start_h_ub_fp32, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                else:
                    self.tik_instance.vconv(MASK64, '', temp_f16_ub, bin_start_h_ub_fp32,
                                            DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE_FOUR, REP_STRIDE_EIGHT)
                    self.tik_instance.vconv(MASK64, 'floor', bin_start_h_floor,
                                            temp_f16_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE_FOUR)
                    self._bias_f32_f16_int32(bin_start_h_ub_fp32, bin_start_h_floor,
                                             bin_start_h_floor, 'floor', DIGIT_128 // DIGIT_64)

                # scalar_roi_start_h add scalar_bin_height*(1...128)
                self.tik_instance.vmuls(self.mask, bin_start_h_ub,
                                        self.const_1_128_ub, scalar_bin_height,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                self.tik_instance.vconv(MASK64, '', bin_start_h_ub_fp32, bin_start_h_ub,
                                        DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                        REP_STRIDE_FOUR)
                self.tik_instance.vadds(MASK64, bin_start_h_ub_fp32, bin_start_h_ub_fp32,
                                        scalar_roi_start_h, DIGIT_128 // DIGIT_64,
                                        STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)

                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) not in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    self.tik_instance.vconv(MASK64, 'ceil', bin_end_h_ceil,
                                            bin_start_h_ub_fp32, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                else:
                    self.tik_instance.vconv(MASK64, '', temp_f16_ub, bin_start_h_ub_fp32,
                                            DIGIT_128 // DIGIT_64, STRIDE_ONE, STRIDE_ONE,
                                            REP_STRIDE_FOUR, REP_STRIDE_EIGHT)
                    self.tik_instance.vconv(MASK64, 'ceil', bin_end_h_ceil,
                                            temp_f16_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE_FOUR)
                    self._bias_f32_f16_int32(bin_start_h_ub_fp32, bin_end_h_ceil,
                                             bin_end_h_ceil, 'ceil', DIGIT_128 // DIGIT_64)
            else:
                scalar_roi_start_w = self.tik_instance.Scalar(self.dtype, name="scalar_roi_start_w")
                scalar_roi_start_w.set_as(rois_spatial_ub[0, roi_i])
                scalar_roi_start_h = self.tik_instance.Scalar(self.dtype, name="scalar_roi_start_h")
                scalar_roi_start_h.set_as(rois_spatial_ub[1, roi_i])
                bin_start_w_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_128,),
                                                          name="bin_start_w_ub", scope=tbe_platform.scope_ubuf)
                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    bin_start_w_fp16_ub = \
                        self.tik_instance.Tensor(FP16, (DIGIT_128,), name="bin_start_w_fp16_ub",
                                                 scope=tbe_platform.scope_ubuf)
                # scalar_roi_start_w + scalar_bin_width*(0...127)
                self.tik_instance.vmuls(self.mask, bin_start_w_ub,
                                        self.const_0_127_ub, scalar_bin_width,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                self.tik_instance.vadds(self.mask, bin_start_w_ub, bin_start_w_ub,
                                        scalar_roi_start_w,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    self.tik_instance.vconv(MASK64, '', bin_start_w_fp16_ub,
                                            bin_start_w_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE.get(FP16), REP_STRIDE_EIGHT)
                    self.tik_instance.vconv(MASK64, 'floor', bin_start_w_floor,
                                            bin_start_w_fp16_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE.get(FP16))
                    self._bias_f32_f16_int32(bin_start_w_ub, bin_start_w_floor,
                                             bin_start_w_floor, 'floor', DIGIT_128 // DIGIT_64)
                else:
                    # vconv.floor: f162s32f or f322s32f
                    self.tik_instance.vconv(MASK64, 'floor', bin_start_w_floor,
                                            bin_start_w_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE.get(self.dtype))

                # scalar_roi_start_w + scalar_bin_width*(0...127)
                self.tik_instance.vmuls(self.mask, bin_start_w_ub,
                                        self.const_1_128_ub, scalar_bin_width,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                self.tik_instance.vadds(self.mask, bin_start_w_ub, bin_start_w_ub,
                                        scalar_roi_start_w,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    self.tik_instance.vconv(MASK64, '', bin_start_w_fp16_ub,
                                            bin_start_w_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE.get(FP16),
                                            REP_STRIDE_EIGHT)
                    self.tik_instance.vconv(MASK64, 'ceil', bin_end_w_ceil,
                                            bin_start_w_fp16_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE.get(FP16))
                    self._bias_f32_f16_int32(bin_start_w_ub, bin_end_w_ceil,
                                             bin_end_w_ceil, 'ceil', DIGIT_128 // DIGIT_64)
                else:
                    self.tik_instance.vconv(MASK64, 'ceil', bin_end_w_ceil,
                                            bin_start_w_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE.get(self.dtype))

                bin_start_h_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_128,), \
                                                          name="bin_start_h_ub", scope=tbe_platform.scope_ubuf)

                # scalar_roi_start_h + scalar_bin_height*(0...127)
                self.tik_instance.vmuls(self.mask, bin_start_h_ub,
                                        self.const_0_127_ub, scalar_bin_height,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                self.tik_instance.vadds(self.mask, bin_start_h_ub, bin_start_h_ub,
                                        scalar_roi_start_h,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    self.tik_instance.vconv(MASK64, '', bin_start_w_fp16_ub,
                                            bin_start_h_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE.get(FP16),
                                            REP_STRIDE_EIGHT)
                    self.tik_instance.vconv(MASK64, 'floor', bin_start_h_floor,
                                            bin_start_w_fp16_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE.get(FP16))
                    self._bias_f32_f16_int32(bin_start_h_ub, bin_start_h_floor,
                                             bin_start_h_floor, 'floor', DIGIT_128 // DIGIT_64)
                else:
                    self.tik_instance.vconv(MASK64, 'floor', bin_start_h_floor,
                                            bin_start_h_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE.get(self.dtype))

                self.tik_instance.vmuls(self.mask, bin_start_h_ub,
                                        self.const_1_128_ub, scalar_bin_height,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                self.tik_instance.vadds(self.mask, bin_start_h_ub, bin_start_h_ub,
                                        scalar_roi_start_h,
                                        DIGIT_128 // self.vec_elem_num,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    self.tik_instance.vconv(MASK64, '', bin_start_w_fp16_ub,
                                            bin_start_h_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE.get(FP16),
                                            REP_STRIDE_EIGHT)
                    self.tik_instance.vconv(MASK64, 'ceil', bin_end_h_ceil,
                                            bin_start_w_fp16_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE.get(FP16))
                    self._bias_f32_f16_int32(bin_start_h_ub, bin_end_h_ceil,
                                             bin_end_h_ceil, 'ceil', DIGIT_128 // DIGIT_64)
                else:
                    self.tik_instance.vconv(MASK64, 'ceil', bin_end_h_ceil,
                                            bin_start_h_ub, DIGIT_128 // DIGIT_64,
                                            STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                            REP_STRIDE.get(self.dtype))

            self.tik_instance.vector_dup(MASK64, dup_tmp_ub, 0, REPEAT_1,
                                         STRIDE_ONE, REP_STRIDE_EIGHT)
            self.tik_instance.vmax(MASK64, bin_start_w_floor, bin_start_w_floor,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            self.tik_instance.vmax(MASK64, bin_end_w_ceil, bin_end_w_ceil,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            self.tik_instance.vmax(MASK64, bin_start_h_floor, bin_start_h_floor,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            self.tik_instance.vmax(MASK64, bin_end_h_ceil, bin_end_h_ceil,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)

            # `vmin(,width/height)`
            self.tik_instance.vector_dup(MASK64, dup_tmp_ub, self.fm_w,
                                         REPEAT_1, STRIDE_ONE, REP_STRIDE_EIGHT)
            self.tik_instance.vmin(MASK64, bin_start_w_floor, bin_start_w_floor,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            self.tik_instance.vmin(MASK64, bin_end_w_ceil, bin_end_w_ceil,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)

            self.tik_instance.vector_dup(MASK64, dup_tmp_ub, self.fm_h,
                                         REPEAT_1, STRIDE_ONE, REP_STRIDE_EIGHT)
            self.tik_instance.vmin(MASK64, bin_start_h_floor, bin_start_h_floor,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            self.tik_instance.vmin(MASK64, bin_end_h_ceil, bin_end_h_ceil,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            params["bin_start_h_floor"] = bin_start_h_floor
            params["bin_end_h_ceil"] = bin_end_h_ceil
            params["bin_start_w_floor"] = bin_start_w_floor
            params["bin_end_w_ceil"] = bin_end_w_ceil

            self._process_one_roi(params)

    def _process_rois(self, roi_step, rois_num_offset, roi_loop, roi_step_l):
        """
        process roi_num roi cyclically, and process roi_step roi each time.

        Parameters
        ----------
        roi_step: the number of rois per loop in process, 64(fp32) or 128(fp16)
        rois_num_offset: the offset in block_id aicore
        roi_loop: loops of processing roi_num rois
        roi_step_l: the number of rois in last loop

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, roi_loop) as inner_i:
            rois_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_5, roi_step),
                                               name="rois_ub",
                                               scope=tbe_platform.scope_ubuf)
            rois_offset = self.tik_instance.Scalar(INT32, name="rois_offset")
            rois_offset.set_as(rois_num_offset + roi_step * inner_i)
            # move rois data to ub from gm
            self._load_rois_to_ub(rois_ub, rois_offset, roi_step)

            # calculate spatial scale rois
            rois_floor_ub = self.tik_instance.Tensor(INT32, (DIGIT_5, roi_step), \
                                                     name="rois_floor_ub", scope=tbe_platform.scope_ubuf)
            rois_spatial_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_8, roi_step), \
                                                       name="rois_spatial_ub", scope=tbe_platform.scope_ubuf)
            self._spatial_scale_rois(rois_ub, rois_floor_ub, rois_spatial_ub, roi_step)

            with self.tik_instance.if_scope(inner_i == (roi_loop - 1)):
                self._process_step1_roi(rois_floor_ub, rois_spatial_ub, \
                                        rois_num_offset, roi_step * inner_i, roi_step_l)

            with self.tik_instance.else_scope():
                self._process_step1_roi(rois_floor_ub, rois_spatial_ub, \
                                        rois_num_offset, roi_step * inner_i, roi_step)

    def _process_rois_multi_batch(self, roi_step, rois_num_offset, roi_loop,
                                  roi_step_l, batch_id):
        """
        process roi_num roi cyclically, and process roi_step roi each time.

        Parameters
        ----------
        roi_step: the number of rois per loop in process, 64(fp32) or 128(fp16)
        rois_num_offset: the offset in block_id aicore
        roi_loop: loops of processing roi_num rois
        roi_step_l: the number of rois in last loop
        batch_id: batch id

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, roi_loop) as inner_i:
            rois_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_5, roi_step),
                                               name="rois_ub", scope=tbe_platform.scope_ubuf)
            # rois addr offset
            rois_offset = self.tik_instance.Scalar(INT32, name="rois_offset")
            rois_offset.set_as(rois_num_offset + \
                               batch_id * (self.roi_num_b * self.roi_shape[1]) + \
                               roi_step * inner_i)
            # move rois data to ub from gm
            self._load_rois_to_ub(rois_ub, rois_offset, roi_step)

            # calculate spatial scale rois
            rois_floor_ub = self.tik_instance.Tensor(INT32, (DIGIT_5, roi_step),
                                                     name="rois_floor_ub",
                                                     scope=tbe_platform.scope_ubuf)
            rois_spatial_ub = self.tik_instance.Tensor(self.dtype,
                                                       (DIGIT_8, roi_step),
                                                       name="rois_spatial_ub",
                                                       scope=tbe_platform.scope_ubuf)
            self._spatial_scale_rois(rois_ub, rois_floor_ub, rois_spatial_ub,
                                     roi_step)

            with self.tik_instance.if_scope(inner_i == (roi_loop - 1)):
                self._process_step1_roi(rois_floor_ub, rois_spatial_ub, \
                                        rois_num_offset + self.roi_num_b * batch_id, \
                                        roi_step * inner_i, roi_step_l)

            with self.tik_instance.else_scope():
                self._process_step1_roi(rois_floor_ub, rois_spatial_ub, \
                                        rois_num_offset + self.roi_num_b * batch_id, \
                                        roi_step * inner_i, roi_step)

    def _init_const_0_127_ub(self):
        """
        init const_0_127_ub, which store 0.0 - 127.0.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.const_0_127_ub = self.tik_instance.Tensor(
            self.dtype, (DIGIT_128,), name="const_0_127_ub", scope=tbe_platform.scope_ubuf)
        if self.is_hisi_cs:
            # s162f16: vconv
            const_0_127_int16 = self.tik_instance.Tensor(INT16, (DIGIT_128,), \
                                                         name="const_0_127_int16", scope=tbe_platform.scope_ubuf)
            with self.tik_instance.for_range(0, DIGIT_128) as i:
                const_0_127_int16[i].set_as(i)
            self.tik_instance.vconv(DIGIT_128, '', self.const_0_127_ub,
                                    const_0_127_int16, REPEAT_1, STRIDE_ONE,
                                    STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        else:
            # s322f16:vconv.deq, or s322f32:vconv
            const_0_127_int32 = self.tik_instance.Tensor(INT32, (DIGIT_128,), \
                                                         name="const_0_127_int32", scope=tbe_platform.scope_ubuf)
            with self.tik_instance.for_range(0, DIGIT_128) as i:
                const_0_127_int32[i].set_as(i)

            if self.dtype == FP16:
                self.tik_instance.vconv(MASK64, '', self.const_0_127_ub,
                                        const_0_127_int32, REPEAT_2, STRIDE_ONE,
                                        STRIDE_ONE, REP_STRIDE.get(self.dtype),
                                        REP_STRIDE_EIGHT, deqscale=DEQSCALE)
            else:
                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    const_0_127_fp16 = self.tik_instance.Tensor(FP16, (DIGIT_128,),
                                                                name="const_0_127_fp16",
                                                                scope=tbe_platform.scope_ubuf)
                    self.tik_instance.vconv(MASK64, '', const_0_127_fp16,
                                            const_0_127_int32, REPEAT_2, STRIDE_ONE,
                                            STRIDE_ONE, REP_STRIDE.get(FP16),
                                            REP_STRIDE.get(self.dtype), deqscale=DEQSCALE)
                    self.tik_instance.vconv(MASK64, '', self.const_0_127_ub,
                                            const_0_127_fp16, REPEAT_2, STRIDE_ONE,
                                            STRIDE_ONE, REP_STRIDE.get(self.dtype), REP_STRIDE.get(FP16))
                else:
                    self.tik_instance.vconv(MASK64, '', self.const_0_127_ub,
                                            const_0_127_int32, REPEAT_2, STRIDE_ONE,
                                            STRIDE_ONE, REP_STRIDE.get(self.dtype),
                                            REP_STRIDE_EIGHT)

    def _init_const_1_128_ub(self):
        """
        init _init_const_1_128_ub, which store 1.0 - 128.0.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.const_1_128_ub = self.tik_instance.Tensor(
            self.dtype, (DIGIT_128,), name="const_1_128_ub", scope=tbe_platform.scope_ubuf)
        if self.is_hisi_cs:
            # s162f16: vconv
            const_1_128_int16 = self.tik_instance.Tensor(INT16, (DIGIT_128,), \
                                                         name="const_1_128_int16", scope=tbe_platform.scope_ubuf)
            with self.tik_instance.for_range(0, DIGIT_128) as i:
                const_1_128_int16[i].set_as(i + 1)
            self.tik_instance.vconv(DIGIT_128, '', self.const_1_128_ub,
                                    const_1_128_int16, REPEAT_1, STRIDE_ONE,
                                    STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        else:
            # s322f16:vconv.deq, or s322f32:vconv
            const_1_128_int32 = self.tik_instance.Tensor(INT32, (DIGIT_128,), \
                                                         name="const_0_127_int32", scope=tbe_platform.scope_ubuf)
            with self.tik_instance.for_range(0, DIGIT_128) as i:
                const_1_128_int32[i].set_as(i + 1)

            if self.dtype == FP16:
                self.tik_instance.vconv(MASK64, '', self.const_1_128_ub,
                                        const_1_128_int32, REPEAT_2, STRIDE_ONE,
                                        STRIDE_ONE, REP_STRIDE.get(self.dtype),
                                        REP_STRIDE_EIGHT, deqscale=DEQSCALE)
            else:
                if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in \
                        (tbe_platform.ASCEND_310, tbe_platform.ASCEND_310B):
                    const_1_128_fp16 = self.tik_instance.Tensor(FP16, (DIGIT_128,),
                                                                name="const_1_128_fp16",
                                                                scope=tbe_platform.scope_ubuf)
                    self.tik_instance.vconv(MASK64, '', const_1_128_fp16,
                                            const_1_128_int32, REPEAT_2, STRIDE_ONE,
                                            STRIDE_ONE, REP_STRIDE.get(FP16),
                                            REP_STRIDE.get(self.dtype), deqscale=DEQSCALE)
                    self.tik_instance.vconv(MASK64, '', self.const_1_128_ub,
                                            const_1_128_fp16, REPEAT_2, STRIDE_ONE,
                                            STRIDE_ONE, REP_STRIDE.get(self.dtype), REP_STRIDE.get(FP16))
                else:
                    self.tik_instance.vconv(MASK64, '', self.const_1_128_ub,
                                            const_1_128_int32, REPEAT_2, STRIDE_ONE,
                                            STRIDE_ONE, REP_STRIDE.get(self.dtype),
                                            REP_STRIDE_EIGHT)

    def psroi_pooling_compute(self):
        """
        compute of PSROIPoolingV2.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        outer_loop = self.roi_num_b // self.aicore_num
        outer_tail = self.roi_num_b % self.aicore_num
        roi_step = self.roi_num_step

        # outer_loop is 0
        num1, num2 = 1, 1
        block_num = outer_tail
        roi_loop1, roi_step1_l = 1, 1
        roi_loop2, roi_step2_l = 1, 1
        if outer_loop > 0:
            block_num = self.aicore_num
            if outer_tail > 0:
                num1 = outer_loop + 1
                num2 = outer_loop
            else:
                num1 = outer_loop
                num2 = outer_loop

            roi_loop1 = _ceil_value(num1, roi_step)
            if (num1 % roi_step == 0):
                roi_step1_l = roi_step
            else:
                roi_step1_l = num1 % roi_step
            roi_loop2 = _ceil_value(num2, roi_step)
            if (num2 % roi_step == 0):
                roi_step2_l = roi_step
            else:
                roi_step2_l = num2 % roi_step

        with self.tik_instance.for_range(0, block_num, block_num=block_num) as block_id:
            # process of one aicore
            self._init_const_0_127_ub()
            self._init_const_1_128_ub()

            rois_num_offset = self.tik_instance.Scalar(INT32, name="rois_num_offset")

            if self.fm_batch == 1:
                # process roi nums: num1
                with self.tik_instance.if_scope(block_id < outer_tail):
                    # rois_num_offset is the offset in block_id aicore
                    rois_num_offset.set_as(block_id * num1)
                    self._process_rois(roi_step, rois_num_offset, roi_loop1, roi_step1_l)
                # process roi nums: num2
                with self.tik_instance.else_scope():
                    if outer_loop > 0:
                        rois_num_offset.set_as(outer_tail * num1 + (block_id - outer_tail) * num2)
                        self._process_rois(roi_step, rois_num_offset, roi_loop2, roi_step2_l)

            else:
                # process roi nums: num1*fm_batch
                with self.tik_instance.if_scope(block_id < outer_tail):
                    # rois_num_offset is the offset in block_id aicore
                    with self.tik_instance.for_range(0, self.fm_batch) as batch_id:
                        rois_num_offset.set_as(block_id * num1)
                        self._process_rois_multi_batch(roi_step, rois_num_offset, roi_loop1, \
                                                       roi_step1_l, batch_id)
                # process roi nums: num2*fm_batch
                with self.tik_instance.else_scope():
                    if outer_loop > 0:
                        with self.tik_instance.for_range(0, self.fm_batch) as batch_id:
                            rois_num_offset.set_as(outer_tail * num1 + (block_id - outer_tail) * num2)
                            self._process_rois_multi_batch(roi_step, rois_num_offset, roi_loop2, \
                                                           roi_step2_l, batch_id)

    def psroi_pooling_main(self):
        """
        Main process of PSROIPoolingV2.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.x = self.tik_instance.Tensor(self.dtype, self.x_shape,
                                          name="x", scope=tbe_platform.scope_gm)
        rois_shape = (self.roi_shape[0] * self.roi_shape[2] * self.roi_shape[1] + self.vec_elem_num,)
        self.rois = self.tik_instance.Tensor(self.dtype, rois_shape,
                                             name="rois", scope=tbe_platform.scope_gm)
        self.y = self.tik_instance.Tensor(self.dtype, shape=self.y_shape,
                                          name="y", scope=tbe_platform.scope_gm)

        self.psroi_pooling_compute()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x, self.rois), outputs=(self.y,))


# 'pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def p_s_r_o_i_pooling_v2(x_dict, rois_dict, y_dict, output_dim, group_size,
                         spatial_scale, kernel_name="PSROIPoolingV2"):
    """
    PSROIPoolingV2 interface.

    Parameters
    ----------
    x_dict: feature map size and data type, 5HD
    rois_dict: rois_dict size and data type, (batch, 5, rois_num), rois all
                nums is batch*rois_num
    y_dict: output size and data type, 5HD
    output_dim: number of output channels
    group_size: number of groups encoding position sensitive score maps
    spatial_scale: spatial scale
    kernel_name: kernel name of PSROIPoolingV2 op

    Returns
    -------
    tik_instance
    """
    psroi_instance = PSROIPoolingV2Class(x_dict, rois_dict, y_dict,
                                         (output_dim, group_size, spatial_scale), kernel_name)
    psroi_instance.psroi_pooling_main()

    return psroi_instance.tik_instance
