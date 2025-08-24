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
nms_with_mask
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_common import is_unknown_rank_input
from tbe.common.platform import get_bit_len
import tbe.common.register as tbe_register


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    # shape's dim of input must be 2
    INPUT_DIMS = 2
    # scaling factor
    DOWN_FACTOR = 0.054395
    # vector unit can compute 256 bytes in one cycle
    BYTES_ONE_CYCLE_VECTOR = 256
    # process 128 proposals at a time for fp16
    BURST_PROPOSAL_NUM = 128
    # valid proposal column contains x1,y1,x2,y2,score
    VALID_COLUMN_NUM = 5
    # each region proposal contains eight elements
    ELEMENT_NUM = 8
    # data align size, also size of one block
    CONFIG_DATA_ALIGN = 32
    REPEAT_TIMES_MAX = 255
    # next_nonzero_idx shape0 is 16 for 32B aligned, 16 is enough
    SHAPE_NEXT_NONZERO = 16
    # mask used for vcmax in update_next_nonzero, 256//2=128, fixed fp16 here but enough for input_dtype
    MASK_VCMAX_FP16 = 128
    BLOCK_INT32 = 8
    TILING_PARAMS_NUM = 8
    # size of some data types
    BYTES_SIZE_INT8 = get_bit_len('int8') // 8
    BYTES_SIZE_UINT8 = get_bit_len('uint8') // 8
    BYTES_SIZE_UINT16 = get_bit_len('uint16') // 8
    BYTES_SIZE_FP16 = get_bit_len('float16') // 8
    BYTES_SIZE_INT32 = get_bit_len('int32') // 8
    BYTES_SIZE_UINT32 = get_bit_len('uint32') // 8
    BYTES_SIZE_FP32 = get_bit_len('fp32') // 8


def _ceiling(value, factor):
    """
    Compute the smallest integer value that is greater than or equal to value and can divide factor
    """
    result = (value + (factor - 1)) // factor * factor
    return result


def _ceiling_scalar(tik_instance, value, factor):
    """
    Compute the smallest integer value that is greater than or equal to value and can divide factor
    """
    result = tik_instance.Scalar(dtype="int32",
                                 name="ceiling_scalar",
                                 init_value=(value + (factor - 1)) // factor * factor)
    return result


def _ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


def _ceil_div_scalar(tik_instance, value, factor):
    """
    Compute the smallest integer value that is greater than or equal to value/factor
    """
    repeat_reg = tik_instance.Scalar(dtype="int32", name="ceil_div_scalar",
                                     init_value=(value + (factor - 1)) // factor)
    return repeat_reg


class _NMSHelper():
    """
    handle all input proposals, e.g. N may > 128

    idea:
                        sn's mask: sn+1     andMask(means: which idx still exists in dst), vand or vmul
        init:           [1 1 1 1 1 1 1 1]   [1 1 1 1 1 1 1 1]  init state, from 0.elem, now 0.elem
        s0's result:    [0 0 1 0 1 1 0 1]   [0 0 1 0 1 1 0 1]  after one loop, get s1 is result of s0, now 2.elem
        s2's result:    [0 0 0 1 1 0 1 0]   [0 0 0 0 1 0 0 0]  now 4.elem
        s4's result:    [0 0 0 0 0 0 0 0]   [0 0 0 0 0 0 0 0]  end

        dst: 0.2.4. elem, so [1 0 1 0 1 0 0 0]
        so far, get output_mask_ub

    note:
        output mask: uint8
        output index: int32
        output proposals: float16 or float32
    """
    def __init__(self, tik_instance, input_boxes_max_num, input_dtype, iou_thres, boxes_num_scalar):
        """
        Parameters:
        ----------
        tik_instance: tik instance
        all_inp_proposals_gm: size is N*8
        input_shape: corresponds to all_inp_proposals_ub
        input_dtype: new soc supports: float16 and float32
        iou_thres: iou threshold, one box is valid if its iou is lower than the threshold

        Returns
        -------
        None
        """
        self.tik_instance = tik_instance
        self.input_dtype = input_dtype

        if input_dtype == 'float16':
            self.input_bytes_each_elem = Constant.BYTES_SIZE_FP16
            self.input_vector_mask_max = Constant.BURST_PROPOSAL_NUM
        elif input_dtype == 'float32':
            self.input_bytes_each_elem = Constant.BYTES_SIZE_FP32
            self.input_vector_mask_max = Constant.BURST_PROPOSAL_NUM // 2

        self.data_type = 'float32'
        self.bytes_each_elem = 4
        self.vector_mask_max = 64

        # note: N canbe used in size, but not for def tensor, should use ceil_n
        self.n_max = input_boxes_max_num
        self.ceil_n_max = _ceiling(self.n_max, self.vector_mask_max)

        # set input gm
        self.all_inp_proposals_gm_max = tik_instance.Tensor(input_dtype, (self.n_max, Constant.ELEMENT_NUM),
                                                            name="in_proposals", scope=tik.scope_gm)
        # init scalar
        self._init_scalar(tik_instance, boxes_num_scalar, iou_thres)

        # note: defed size need to 32b aligned
        self.x1_max_ub = tik_instance.Tensor(shape=(self.ceil_n_max, ), dtype=self.data_type,
                                             name='x1_max_ub', scope=tik.scope_ubuf)
        self.x2_max_ub = tik_instance.Tensor(shape=(self.ceil_n_max, ), dtype=self.data_type,
                                             name='x2_max_ub', scope=tik.scope_ubuf)
        self.y1_max_ub = tik_instance.Tensor(shape=(self.ceil_n_max, ), dtype=self.data_type,
                                             name='y1_max_ub', scope=tik.scope_ubuf)
        self.y2_max_ub = tik_instance.Tensor(shape=(self.ceil_n_max, ), dtype=self.data_type,
                                             name='y2_max_ub', scope=tik.scope_ubuf)

        # 1980's maximum input => new soc's output_mask_ub
        self.all_inp_proposals_ub_fp32_max = tik_instance.Tensor('float32', (self.ceil_n_max, Constant.ELEMENT_NUM),
                                                                 name="all_inp_proposals_ub_fp32_max",
                                                                 scope=tik.scope_ubuf)
        # def tmp ub tensor
        self.tmp_tensor_ub_fp16_max = tik_instance.Tensor('float16', (self.ceil_n_max, ), tik.scope_ubuf,
                                                          'tmp_tensor_ub_fp16')
        self.tmp_tensor_ub_fp16_burst = tik_instance.Tensor('float16', (Constant.BURST_PROPOSAL_NUM, ), tik.scope_ubuf,
                                                            'tmp_tensor_ub_fp16_burst')

        # change input data format
        self._input_trans()

        # cache area, calc once is enough
        self.total_areas_ub = None

        # [0] stores next nonzero idx, shape[0]=16 same as idx_fp16_ub.shape in order to conv
        self.next_nonzero_int32_idx_ub = tik_instance.Tensor('int32', (Constant.SHAPE_NEXT_NONZERO, ), tik.scope_ubuf,
                                                             'next_nonzero_int32_idx_ub')

        # init for valid mask
        self._init_for_valid_mask()

        # selected_idx_ub generate
        self.selected_idx_ub = self._selected_idx_gen()

        # init for vcmax
        self._init_for_vcmax()

        # for inter
        self._init_for_inter()

        self.area_cur_scalar = self.tik_instance.Scalar(self.data_type, 'area_cur_scalar')

        # output mask, dtype is int8 fixed
        self.max_output_mask_int8_ub = self.tik_instance.Tensor('int8', (self.max_valid_mask_size_int8, ),
                                                                tik.scope_ubuf, "max_output_mask_int8_ub")

        self._init_for_cmpmask2bitmask()

        # scaling
        self._scaling()

    def _init_scalar(self, tik_instance, boxes_num_scalar, iou_thres):
        self.n_actual_scalar = tik_instance.Scalar(dtype="int32", name="n_actual_scalar")
        self.n_actual_scalar.set_as(boxes_num_scalar)
        self.ceil_n_actual_scalar = _ceiling_scalar(tik_instance, self.n_actual_scalar, self.vector_mask_max)

        self.input_size_max = self.n_max * Constant.ELEMENT_NUM
        self.input_size_actual_scalar = tik_instance.Scalar(dtype="int32",
                                                            init_value=self.n_actual_scalar * Constant.ELEMENT_NUM)
        # set iou thres
        if isinstance(iou_thres, float):
            self.iou_thres_factor = iou_thres / (iou_thres + 1)
        else:
            self.iou_thres_factor = tik_instance.Scalar(dtype=self.data_type, name="iou_thres_factor")
            self.iou_thres_factor.set_as(iou_thres / (iou_thres + 1))

        # cache frequently used
        self.negone_int8_scalar = tik_instance.Scalar('int8', 'negone_int8_scalar', init_value=-1)
        self.zero_int8_scalar = tik_instance.Scalar('int8', 'zero_int8_scalar', init_value=0)
        self.zero_int16_scalar = tik_instance.Scalar('int16', 'zero_int16_scalar', init_value=0)
        self.one_uint8_scalar = tik_instance.Scalar('uint8', 'one_uint8_scalar', init_value=1)
        self.one_int16_scalar = tik_instance.Scalar('int16', 'one_int16_scalar', init_value=1)

        # scalar: zero of dtype, one
        self.zero_datatype_scalar = tik_instance.Scalar(self.data_type, 'zero_dtype_scalar', init_value=0.)
        self.one_datatype_scalar = tik_instance.Scalar(self.data_type, 'one_dtype_scalar', init_value=1.)
        
    def _input_trans(self):
        """
        inputs format reshape
        Note: should use vreduce, not vgather
        all_inp_proposals_ub:
            before:
                shape is (N, 8), only one addr_base
                [
                [x1, y1, x2, y2, score, /, /, /]
                [x1, y1, x2, y2, score, /, /, /]
                [x1, y1, x2, y2, score, /, /, /]
                [x1, y1, x2, y2, score, /, /, /]
                ...
                ]

            after:
                5 addr_bases
                x1[] with N elems
                x2[]
                y1[]
                y2[]
                score[]

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # 2 ** 0 + 2 ** 8 + 2 ** 16 + 2 ** 24
        pattern_value_fp32_x1 = 16843009
        # 2 ** 1 + 2 ** 9 + 2 ** 17 + 2 ** 25
        pattern_value_fp32_y1 = 33686018
        # 2 ** 2 + 2 ** 10 + 2 ** 18 + 2 ** 26
        pattern_value_fp32_x2 = 67372036
        # 2 ** 3 + 2 ** 11 + 2 ** 19 + 2 ** 27
        pattern_value_fp32_y2 = 134744072
        if self.input_dtype == 'float16':
            # Constant.BURST_PROPOSAL_NUM is shape0 of tmp_tensor_ub_fp16_burst
            tmp_actual_n_scalar = self.tik_instance.Scalar(dtype="int32",
                                                           init_value=self.n_actual_scalar * Constant.ELEMENT_NUM)
            repeat_actual_scalar = _ceil_div_scalar(self.tik_instance, tmp_actual_n_scalar,
                                                    Constant.BURST_PROPOSAL_NUM)

            with self.tik_instance.for_range(0, repeat_actual_scalar) as i:
                offset = i * Constant.BURST_PROPOSAL_NUM
                self.tik_instance.data_move(self.tmp_tensor_ub_fp16_burst,
                                            self.all_inp_proposals_gm_max[offset],
                                            0,
                                            1,
                                            Constant.BURST_PROPOSAL_NUM * self.input_bytes_each_elem //
                                            Constant.CONFIG_DATA_ALIGN,
                                            src_stride=0,
                                            dst_stride=0)
                self._tailing_handle_vec_conv(self.all_inp_proposals_ub_fp32_max[offset],
                                              self.tmp_tensor_ub_fp16_burst,
                                              Constant.BURST_PROPOSAL_NUM, Constant.BYTES_SIZE_FP32,
                                              Constant.BYTES_SIZE_FP16)
        else:
            burst_len_scalar = self.tik_instance.Scalar(dtype="int32",
                                                        init_value=self.ceil_n_actual_scalar * Constant.ELEMENT_NUM *
                                                        self.input_bytes_each_elem // Constant.CONFIG_DATA_ALIGN)
            self.tik_instance.data_move(self.all_inp_proposals_ub_fp32_max,
                                        self.all_inp_proposals_gm_max,
                                        0,
                                        1,
                                        burst_len_scalar,
                                        src_stride=0,
                                        dst_stride=0)

        # fp32. uint32 covers 32 elems, so shape[0] is 256/32=8
        pattern_x1_ub = self.tik_instance.Tensor('uint32', (8, ), tik.scope_ubuf, name='pattern_x1_ub')
        pattern_y1_ub = self.tik_instance.Tensor('uint32', (8, ), tik.scope_ubuf, name='pattern_y1_ub')
        pattern_x2_ub = self.tik_instance.Tensor('uint32', (8, ), tik.scope_ubuf, name='pattern_x2_ub')
        pattern_y2_ub = self.tik_instance.Tensor('uint32', (8, ), tik.scope_ubuf, name='pattern_y2_ub')

        self.tik_instance.vector_dup(8, pattern_x1_ub,
                                     self.tik_instance.Scalar('uint32', init_value=pattern_value_fp32_x1), 1, 1, 1)
        self.tik_instance.vector_dup(8, pattern_y1_ub,
                                     self.tik_instance.Scalar('uint32', init_value=pattern_value_fp32_y1), 1, 1, 1)
        self.tik_instance.vector_dup(8, pattern_x2_ub,
                                     self.tik_instance.Scalar('uint32', init_value=pattern_value_fp32_x2), 1, 1, 1)
        self.tik_instance.vector_dup(8, pattern_y2_ub,
                                     self.tik_instance.Scalar('uint32', init_value=pattern_value_fp32_y2), 1, 1, 1)

        self._tailing_handle_vreduce_input(self.x1_max_ub, self.all_inp_proposals_ub_fp32_max, pattern_x1_ub)
        self._tailing_handle_vreduce_input(self.y1_max_ub, self.all_inp_proposals_ub_fp32_max, pattern_y1_ub)
        self._tailing_handle_vreduce_input(self.x2_max_ub, self.all_inp_proposals_ub_fp32_max, pattern_x2_ub)
        self._tailing_handle_vreduce_input(self.y2_max_ub, self.all_inp_proposals_ub_fp32_max, pattern_y2_ub)

    def _tailing_handle_vec_conv(self, dst_ub, src_ub, size, dst_bytes, src_bytes, mode="none", deqscale=None):
        """
        transfer data dtype

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src ub
        size: scalar totol size of elems
        dst_bytes: bytes of each elem of dst
        src_bytes: bytes of each elem of src

        Returns
        -------
        None
        """
        # max. is vector_mask_max. src_bytes can be 1
        mask_max = min(Constant.BYTES_ONE_CYCLE_VECTOR // src_bytes, self.vector_mask_max)
        if dst_bytes == Constant.BYTES_SIZE_INT32:
            mask_max = Constant.BURST_PROPOSAL_NUM // 2

        offset_reg = self.tik_instance.Scalar('int32', 'offset_reg', init_value=0)

        # repeat num
        repeat_reg = self.tik_instance.Scalar('int32',
                                              'repeat_reg',
                                              init_value=size % (mask_max * Constant.REPEAT_TIMES_MAX) // mask_max)

        with self.tik_instance.if_scope(repeat_reg > 0):
            self.tik_instance.vec_conv(mask_max,
                                       mode,
                                       dst=dst_ub[offset_reg],
                                       src=src_ub[offset_reg],
                                       repeat_times=repeat_reg,
                                       dst_rep_stride=mask_max * dst_bytes // Constant.CONFIG_DATA_ALIGN,
                                       src_rep_stride=mask_max * src_bytes // Constant.CONFIG_DATA_ALIGN,
                                       deqscale=deqscale)

        # last num
        last_num_reg = self.tik_instance.Scalar('int32', 'last_num_reg', init_value=size % mask_max)
        with self.tik_instance.if_scope(last_num_reg > 0):
            offset_reg.set_as(offset_reg + repeat_reg * mask_max)
            self.tik_instance.vec_conv(last_num_reg,
                                       mode,
                                       dst=dst_ub[offset_reg],
                                       src=src_ub[offset_reg],
                                       repeat_times=1,
                                       dst_rep_stride=0,
                                       src_rep_stride=0,
                                       deqscale=deqscale)

    def _tailing_handle_vreduce_input(self, dst_ub, src0_ub, src1_pattern_ub):
        """
        transfer data format from
        x1, y1, x2, y2, score
        .   .   .   .   .
        .   .   .   .   .
        .   .   .   .   .

        To

        x1 ...
        y1 ...
        x2 ...
        y2 ...
        score ...

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0 in ub
        src1_pattern_ub: pattern for src1

        Returns
        -------
        None
        """
        # 16 for fp16, 8 for fp32
        vector_proposals_max = self.vector_mask_max // 8
        dst_offset_reg = self.tik_instance.Scalar('int32', 'dst_offset_reg', init_value=0)
        src_offset_reg = self.tik_instance.Scalar('int32', 'src_offset_reg', init_value=0)

        # max repeat
        # only this tailing need the max repeat, other tailings don't need it,
        # as ceil_n may > vector_proposals_max * 255
        loop_num_actual_reg = self.tik_instance.Scalar('int32',
                                                       'loop_num_actual_reg',
                                                       init_value=self.ceil_n_actual_scalar //
                                                       (vector_proposals_max * Constant.REPEAT_TIMES_MAX))

        with self.tik_instance.for_range(0, loop_num_actual_reg, name='loop_num_reg') as i:
            self.tik_instance.vreduce(mask=self.vector_mask_max, dst=dst_ub[dst_offset_reg],
                src0=src0_ub[src_offset_reg], src1_pattern=src1_pattern_ub,
                repeat_times=Constant.REPEAT_TIMES_MAX, src0_blk_stride=1,
                src0_rep_stride=self.vector_mask_max * self.bytes_each_elem // Constant.CONFIG_DATA_ALIGN,
                # here 0 means: pattern is reused in each repeat
                src1_rep_stride=0)
            dst_offset_reg.set_as((i + 1) * vector_proposals_max * Constant.REPEAT_TIMES_MAX)
            src_offset_reg.set_as(dst_offset_reg * 8)

        # repeat num
        repeat_actual_scalar = self.tik_instance.Scalar('int32',
                                                        'repeat_actual_scalar',
                                                        init_value=self.ceil_n_actual_scalar %
                                                        (vector_proposals_max * Constant.REPEAT_TIMES_MAX) //
                                                        vector_proposals_max)

        with self.tik_instance.if_scope(repeat_actual_scalar > 0):
            self.tik_instance.vreduce(mask=self.vector_mask_max, dst=dst_ub[dst_offset_reg],
                src0=src0_ub[src_offset_reg], src1_pattern=src1_pattern_ub,
                repeat_times=repeat_actual_scalar, src0_blk_stride=1,
                src0_rep_stride=self.vector_mask_max * self.bytes_each_elem // Constant.CONFIG_DATA_ALIGN,
                # here 0 means: pattern is reused in each repeat
                src1_rep_stride=0)

        # last num
        last_num_actual_reg = self.tik_instance.Scalar('int32',
                                                       'last_num_actual_reg',
                                                       init_value=self.ceil_n_actual_scalar % vector_proposals_max)

        with self.tik_instance.if_scope(last_num_actual_reg > 0):
            repeat_tmp_reg = self.tik_instance.Scalar('int32',
                                                      'repeat_tmp_reg',
                                                      init_value=repeat_actual_scalar * vector_proposals_max)
            last_tmp_reg = self.tik_instance.Scalar('int32', 'last_tmp_reg', init_value=last_num_actual_reg * 8)

            dst_offset_reg.set_as(dst_offset_reg + repeat_tmp_reg)
            src_offset_reg.set_as(dst_offset_reg * 8)

            self.tik_instance.vreduce(mask=last_tmp_reg, dst=dst_ub[dst_offset_reg], src0=src0_ub[src_offset_reg],
                src1_pattern=src1_pattern_ub, repeat_times=1, src0_blk_stride=1,
                # no need to repeat, so 0
                src0_rep_stride=0,
                # here 0 means: pattern is reused in each repeat
                src1_rep_stride=0)

    def _init_for_valid_mask(self):
        """
        note:
            for update_valid_mask, valid_mask uses int16, which is for using vand,
            vand supports fp162int16 (use round ...)

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        # ceiling vector_mask_max for handling tailing
        self.actual_valid_mask_size_int8_scalar = _ceiling_scalar(tik_instance, self.n_actual_scalar,
                                                                  self.vector_mask_max)
        self.max_valid_mask_size_int8 = _ceiling(self.n_max, self.vector_mask_max)
        self.max_valid_mask_int8_ub = tik_instance.Tensor('int8', (self.max_valid_mask_size_int8, ), tik.scope_ubuf,
                                                          'max_valid_mask_int8_ub')
        self.valid_mask_fp16_ub_max = self.tmp_tensor_ub_fp16_max

        scalar_i = tik_instance.Scalar('float16', 'scalar_i', init_value=1)

        self._tailing_handle_vector_dup(self.valid_mask_fp16_ub_max, scalar_i, self.actual_valid_mask_size_int8_scalar,
                                        Constant.BYTES_SIZE_FP16)
        self._tailing_handle_vec_conv(self.max_valid_mask_int8_ub, self.valid_mask_fp16_ub_max,
                                      self.actual_valid_mask_size_int8_scalar, Constant.BYTES_SIZE_INT8,
                                      Constant.BYTES_SIZE_FP16, 'round')

        # update valid mask, here float16 fixed, ensure 32b aligned. note: size `below = valid_mask_size_int8`
        self.tmp_valid_mask_float16_ub_max = self.tik_instance.Tensor('float16', (self.max_valid_mask_size_int8, ),
                                                                      tik.scope_ubuf, 'tmp_valid_mask_float16_ub_max')
        self.tmp_mask_float16_ub_max = self.tmp_tensor_ub_fp16_max

    def _tailing_handle_vector_dup(self, dst_ub, scalar, size, src_bytes):
        """
        handle tailing of vector dup

        Parameters
        ----------
        dst_ub: dst tensor in ub
        scalar: scalar used to dup
        size: totol size of elems
        src_bytes: bytes of each elem of src

        Returns
        -------
        None
        """
        # max. is vector_mask_max. src_bytes can be 1
        mask_max = min(Constant.BYTES_ONE_CYCLE_VECTOR // src_bytes, self.vector_mask_max)
        offset_reg = self.tik_instance.Scalar('int32', 'offset_reg', init_value=0)

        # repeat num
        repeat_reg = self.tik_instance.Scalar('int32',
                                              'repeat_reg',
                                              init_value=size % (mask_max * Constant.REPEAT_TIMES_MAX) // mask_max)

        with self.tik_instance.if_scope(repeat_reg > 0):
            self.tik_instance.vector_dup(mask=mask_max,
                                         dst=dst_ub[offset_reg],
                                         scalar=scalar,
                                         repeat_times=repeat_reg,
                                         dst_blk_stride=1,
                                         dst_rep_stride=mask_max * src_bytes // Constant.CONFIG_DATA_ALIGN)

        # last num
        last_num_reg = self.tik_instance.Scalar('int32', 'last_num_reg', init_value=size % mask_max)
        with self.tik_instance.if_scope(last_num_reg > 0):
            offset_reg.set_as(repeat_reg * mask_max)
            self.tik_instance.vector_dup(mask=last_num_reg,
                                         dst=dst_ub[offset_reg],
                                         scalar=scalar,
                                         repeat_times=1,
                                         dst_blk_stride=0,
                                         dst_rep_stride=0)

    def _selected_idx_gen(self):
        """
        selected_idx generate

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # int32 is fixed for output index
        selected_idx_ub = self.tik_instance.Tensor('int32', (self.ceil_n_max, ), tik.scope_ubuf, 'selected_idx_ub')
        with self.tik_instance.for_range(0, self.ceil_n_actual_scalar) as i:
            selected_idx_ub[i].set_as(i)

        return selected_idx_ub

    def _init_for_vcmax(self):
        """
        init for vcmax, which is used in _update_next_nonzero_idx()

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        tik_instance = self.tik_instance
        # dscend sorted list in ub, fixed dtype is fp16. use selected_idx_ub to generate dsorts_ub
        max_dsorts_size = _ceiling(self.n_max, self.vector_mask_max)
        actual_dsorts_size_scalar = _ceiling_scalar(tik_instance, self.n_actual_scalar, self.vector_mask_max)

        scalar_dsorts_size_actual = tik_instance.Scalar("float16", init_value=actual_dsorts_size_scalar)

        self.dsorts_ub = tik_instance.Tensor('float16', (max_dsorts_size, ), tik.scope_ubuf, 'dsorts_ub')

        self._tailing_handle_vector_dup(self.dsorts_ub, scalar_dsorts_size_actual, actual_dsorts_size_scalar,
                                        Constant.BYTES_SIZE_FP16)

        selected_idx_ub_fp16 = self.tmp_tensor_ub_fp16_max
        self._tailing_handle_vector_dup(selected_idx_ub_fp16, scalar_dsorts_size_actual, actual_dsorts_size_scalar,
                                        Constant.BYTES_SIZE_FP16)
        self._tailing_handle_vec_conv(selected_idx_ub_fp16, self.selected_idx_ub, self.ceil_n_actual_scalar,
                                      Constant.BYTES_SIZE_FP16, Constant.BYTES_SIZE_INT32, '', 1.)

        self._tailing_handle_vsub(self.dsorts_ub, self.dsorts_ub, selected_idx_ub_fp16, actual_dsorts_size_scalar,
                                  Constant.BYTES_SIZE_FP16, Constant.BYTES_SIZE_FP16, Constant.BYTES_SIZE_FP16)

        self.vcmax_ub = tik_instance.Tensor('float16', (Constant.MASK_VCMAX_FP16, ), tik.scope_ubuf, 'vcmax_ub')
        self.middle_max_val_ub = tik_instance.Tensor('float16', (Constant.MASK_VCMAX_FP16, ), tik.scope_ubuf,
                                                     'middle_max_val_ub')
        self.dst_max_val_ub = tik_instance.Tensor('float16', (Constant.SHAPE_NEXT_NONZERO, ), tik.scope_ubuf,
                                                  'dst_max_val_ub')

        # idx_fp16_ub stores next nonzero idx, dtype needs conv to int8
        self.idx_fp16_ub = tik_instance.Tensor('float16', (Constant.SHAPE_NEXT_NONZERO, ), tik.scope_ubuf,
                                               'idx_fp16_ub')

        # practically ceil_n is less than Constant.MASK_VCMAX_FP16 * REPEAT_TIMES_MAX
        self.repeat_vmul_vcmax_actual_scalar = tik_instance.Scalar(
            "int32",
            "repeat_vmul_vcmax_actual_scalar",
            init_value=self.ceil_n_actual_scalar % (Constant.MASK_VCMAX_FP16 * Constant.REPEAT_TIMES_MAX) //
            Constant.MASK_VCMAX_FP16)

        self.last_num_vmul_vcmax_actual_scalar = tik_instance.Scalar("int32",
                                                                     "last_num_vmul_vcmax_actual_scalar",
                                                                     init_value=self.ceil_n_actual_scalar %
                                                                     Constant.MASK_VCMAX_FP16)

        self.vcmax_mask_actual_scalar = tik_instance.Scalar("int32", "vcmax_mask_actual_scalar", init_value=0)
        with tik_instance.if_scope(self.last_num_vmul_vcmax_actual_scalar > 0):
            self.vcmax_mask_actual_scalar.set_as(self.repeat_vmul_vcmax_actual_scalar + 1)
        with tik_instance.else_scope():
            self.vcmax_mask_actual_scalar.set_as(self.repeat_vmul_vcmax_actual_scalar)

    def _tailing_handle_vsub(self, dst_ub, src0_ub, src1_ub, size, dst_bytes, src0_bytes, src1_bytes):
        """
        handle tailing of vsub

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0
        src1_ub: src1
        size: totol size of elems

        Returns
        -------
        None
        """
        offset_reg = self.tik_instance.Scalar('int32', 'offset_reg', init_value=0)

        # repeat num
        repeat_reg = self.tik_instance.Scalar('int32',
                                              'repeat_reg',
                                              init_value=size % (self.vector_mask_max * Constant.REPEAT_TIMES_MAX) //
                                              self.vector_mask_max)

        with self.tik_instance.if_scope(repeat_reg > 0):
            self.tik_instance.vsub(mask=self.vector_mask_max,
                                   dst=dst_ub[offset_reg],
                                   src0=src0_ub[offset_reg],
                                   src1=src1_ub[offset_reg],
                                   repeat_times=repeat_reg,
                                   dst_blk_stride=1,
                                   src0_blk_stride=1,
                                   src1_blk_stride=1,
                                   dst_rep_stride=self.vector_mask_max * dst_bytes // Constant.CONFIG_DATA_ALIGN,
                                   src0_rep_stride=self.vector_mask_max * src0_bytes // Constant.CONFIG_DATA_ALIGN,
                                   src1_rep_stride=self.vector_mask_max * src1_bytes // Constant.CONFIG_DATA_ALIGN)

        # last num
        last_num_reg = self.tik_instance.Scalar('int32', 'last_num_reg', init_value=size % self.vector_mask_max)
        with self.tik_instance.if_scope(last_num_reg > 0):
            offset_reg.set_as(offset_reg + repeat_reg * self.vector_mask_max)
            self.tik_instance.vsub(mask=last_num_reg,
                                   dst=dst_ub[offset_reg],
                                   src0=src0_ub[offset_reg],
                                   src1=src1_ub[offset_reg],
                                   repeat_times=1,
                                   dst_blk_stride=1,
                                   src0_blk_stride=1,
                                   src1_blk_stride=1,
                                   dst_rep_stride=8,
                                   src0_rep_stride=8,
                                   src1_rep_stride=8)

    def _tailing_handle_vabs(self, dst_ub, src_ub, size, dst_bytes, src_bytes):
        """
        handle tailing of vabs

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0
        src1_ub: src1
        size: totol size of elems

        Returns
        -------
        None
        """
        offset_reg = self.tik_instance.Scalar('int32', 'offset_reg', init_value=0)

        # repeat num
        repeat_reg = self.tik_instance.Scalar('int32',
                                              'repeat_reg',
                                              init_value=size % (self.vector_mask_max * Constant.REPEAT_TIMES_MAX) //
                                              self.vector_mask_max)
        with self.tik_instance.if_scope(repeat_reg > 0):
            self.tik_instance.vec_abs(mask=self.vector_mask_max,
                                      dst=dst_ub[offset_reg],
                                      src=src_ub[offset_reg],
                                      repeat_times=repeat_reg,
                                      dst_rep_stride=self.vector_mask_max * dst_bytes // Constant.CONFIG_DATA_ALIGN,
                                      src_rep_stride=self.vector_mask_max * src_bytes // Constant.CONFIG_DATA_ALIGN)

        # last num
        last_num_reg = self.tik_instance.Scalar('int32', 'last_num_reg', init_value=size % self.vector_mask_max)
        with self.tik_instance.if_scope(last_num_reg > 0):
            offset_reg.set_as(offset_reg + repeat_reg * self.vector_mask_max)
            self.tik_instance.vec_abs(mask=last_num_reg,
                                      dst=dst_ub[offset_reg],
                                      src=src_ub[offset_reg],
                                      repeat_times=1,
                                      dst_rep_stride=8,
                                      src_rep_stride=8)

    def _init_for_inter(self):
        """
        init tensors for inter

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.xx1_ub = self.tik_instance.Tensor(self.data_type, (self.ceil_n_max, ), tik.scope_ubuf, "xx1_ub")
        self.yy1_ub = self.tik_instance.Tensor(self.data_type, (self.ceil_n_max, ), tik.scope_ubuf, "yy1_ub")
        # xx2 is reused several times
        self.xx2_ub = self.tik_instance.Tensor(self.data_type, (self.ceil_n_max, ), tik.scope_ubuf, "xx2_ub")
        self.x1i_scalar = self.tik_instance.Scalar(self.data_type, name='x1i_scalar')
        self.y1i_scalar = self.tik_instance.Scalar(self.data_type, name='y1i_scalar')
        self.x2i_scalar = self.tik_instance.Scalar(self.data_type, name='x2i_scalar')
        self.y2i_scalar = self.tik_instance.Scalar(self.data_type, name='y2i_scalar')

    def _init_for_cmpmask2bitmask(self):
        """
        for cmpmask2bitmask, fp16 fixed is OK, this is used in one repeat, so Constant.BURST_PROPOSAL_NUM below is OK
        To compare two nums which is bigger.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.output_mask_f16_ub = self.tik_instance.Tensor('float16', (Constant.BURST_PROPOSAL_NUM, ),
                                                           name="output_mask_f16_ub",
                                                           scope=tik.scope_ubuf)
        zero_fp16_scalar = self.tik_instance.Scalar(dtype="float16", name="zero_scalar", init_value=0.0)
        one_fp16_scalar = self.tik_instance.Scalar(dtype="float16", name="one_scalar", init_value=1.0)
        self.data_fp16_zero_ub = self.tik_instance.Tensor("float16", (Constant.BURST_PROPOSAL_NUM, ),
                                                          name="data_zero_ub",
                                                          scope=tik.scope_ubuf)
        self.data_fp16_one_ub = self.tik_instance.Tensor("float16", (Constant.BURST_PROPOSAL_NUM, ),
                                                         name="data_one_ub",
                                                         scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(Constant.BURST_PROPOSAL_NUM, self.data_fp16_zero_ub, zero_fp16_scalar, 1, 1, 8)
        self.tik_instance.vector_dup(Constant.BURST_PROPOSAL_NUM, self.data_fp16_one_ub, one_fp16_scalar, 1, 1, 8)

    def _scaling(self):
        """
        scaling of input, scaling factor is Constant.DOWN_FACTOR

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._tailing_handle_vmuls(self.x1_max_ub, self.x1_max_ub, Constant.DOWN_FACTOR, self.ceil_n_actual_scalar)
        self._tailing_handle_vmuls(self.x2_max_ub, self.x2_max_ub, Constant.DOWN_FACTOR, self.ceil_n_actual_scalar)
        self._tailing_handle_vmuls(self.y1_max_ub, self.y1_max_ub, Constant.DOWN_FACTOR, self.ceil_n_actual_scalar)
        self._tailing_handle_vmuls(self.y2_max_ub, self.y2_max_ub, Constant.DOWN_FACTOR, self.ceil_n_actual_scalar)

    def loops(self):
        """
        comparing nums and IOU
        A ∩ B > (A + B) * (iou_thr / (iou_thr + 1))

        Parameters
        ----------
        None

        Returns
        -------
        selected_mask_ub
        """
        # def and init selected_mask_ub
        selected_mask_ub_max = self.tik_instance.Tensor('uint8', (self.ceil_n_max, ),
                                                        name="selected_mask_ub",
                                                        scope=tik.scope_ubuf)

        selected_mask_ub_tmp = self.tmp_tensor_ub_fp16_max
        scalar_i = self.tik_instance.Scalar('float16', 'scalar_i', init_value=0)

        self._tailing_handle_vector_dup(selected_mask_ub_tmp,
                                        scalar_i,
                                        size=self.ceil_n_actual_scalar,
                                        src_bytes=Constant.BYTES_SIZE_FP16)
        self._tailing_handle_vec_conv(selected_mask_ub_max, selected_mask_ub_tmp, self.ceil_n_actual_scalar,
                                      Constant.BYTES_SIZE_UINT8, Constant.BYTES_SIZE_FP16, 'round')

        cur_scalar = self.tik_instance.Scalar(dtype='int32', name='cur_scalar', init_value=0)

        with self.tik_instance.for_range(0, self.n_actual_scalar):
            with self.tik_instance.if_scope(cur_scalar < self.n_actual_scalar):
                # set 1, means valid
                selected_mask_ub_max[cur_scalar] = self.one_uint8_scalar
                mask_ub = self._one_loop(cur_scalar)
                self._update_valid_mask(mask_ub)
                self._update_next_nonzero_idx(self.max_valid_mask_int8_ub)
                cur_scalar.set_as(self.next_nonzero_int32_idx_ub[0])

        return selected_mask_ub_max

    def _tailing_handle_vmuls(self, dst_ub, src_ub, scalar, size):
        """
        handle tailing of vmuls

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src ub
        scalar: scalar
        size: totol size of elems

        Returns
        -------
        None
        """
        offset_reg = self.tik_instance.Scalar('int32', 'offset_reg', init_value=0)

        # repeat num
        repeat_reg = self.tik_instance.Scalar('int32',
                                              'repeat_reg',
                                              init_value=size % (self.vector_mask_max * Constant.REPEAT_TIMES_MAX) //
                                              self.vector_mask_max)

        with self.tik_instance.if_scope(repeat_reg > 0):
            self.tik_instance.vmuls(mask=self.vector_mask_max,
                                    dst=dst_ub[offset_reg],
                                    src=src_ub[offset_reg],
                                    scalar=scalar,
                                    repeat_times=repeat_reg,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=8)

        # last num
        last_num_reg = self.tik_instance.Scalar('int32', 'last_num_reg', init_value=size % self.vector_mask_max)
        with self.tik_instance.if_scope(last_num_reg > 0):
            offset_reg.set_as(offset_reg + repeat_reg * self.vector_mask_max)
            self.tik_instance.vmuls(mask=last_num_reg,
                                    dst=dst_ub[offset_reg],
                                    src=src_ub[offset_reg],
                                    scalar=scalar,
                                    repeat_times=1,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=8)

    def _one_loop(self, cur_scalar):
        """
        in one loop: iou, generate bitmask and return output_mask_int8_ub

        logic of nms, new way:
            output mask = 1 if intersection < (area_i + area_j) * iou_thres / (iou_thres + 1)

        Parameters
        ----------
        cur: compute cur proposal and the others

        Returns
        -------
        output_mask_int8_ub
        """
        inter_ub = self._intersection(cur_scalar)

        areas_ub = self._area()
        self.area_cur_scalar.set_as(areas_ub[cur_scalar])
        adds_ub = self.xx2_ub
        self._tailing_handle_vadds(adds_ub, areas_ub, self.area_cur_scalar, self.ceil_n_actual_scalar)
        self._tailing_handle_vmuls(adds_ub, adds_ub, self.iou_thres_factor, self.ceil_n_actual_scalar)

        # cmpmask 2 bitmask
        max_output_mask_int8_ub = self._tailing_handle_cmp_le_and_2bitmask(inter_ub, adds_ub,
                                                                           self.ceil_n_actual_scalar)

        # set output_mask[cur] = 0, because will be added into DST, and deleted from SRC proposal list
        max_output_mask_int8_ub[cur_scalar].set_as(self.zero_int8_scalar)
        return max_output_mask_int8_ub

    def _intersection(self, cur_scalar):
        """
        intersection calculation
        A ∩ B

        Parameters
        ----------
        cur_scalar: intersection of cur proposal and the others

        Returns
        -------
        None
        """
        self.x1i_scalar.set_as(self.x1_max_ub[cur_scalar])
        self.y1i_scalar.set_as(self.y1_max_ub[cur_scalar])
        self.x2i_scalar.set_as(self.x2_max_ub[cur_scalar])
        self.y2i_scalar.set_as(self.y2_max_ub[cur_scalar])

        # `xx1 = max(x1[i], x1[1:]),  yy1 = max(y1[i], y1[1:]), xx2=min(x2[i], x2[1:]),  yy2=min(y2[i], y2[1:])`
        self._tailing_handle_vmaxs(self.xx1_ub, self.x1_max_ub, self.x1i_scalar, self.ceil_n_actual_scalar)
        self._tailing_handle_vmins(self.xx2_ub, self.x2_max_ub, self.x2i_scalar, self.ceil_n_actual_scalar)

        # `w = max(0, xx2-xx1+offset), h = max(0, yy2-yy1+offset), offset=0 here`
        self._tailing_handle_vsub(self.xx1_ub, self.xx2_ub, self.xx1_ub, self.ceil_n_actual_scalar,
                                  self.bytes_each_elem, self.bytes_each_elem, self.bytes_each_elem)
        # w stores in xx1
        self._tailing_handle_vmaxs(self.xx1_ub, self.xx1_ub, self.zero_datatype_scalar, self.ceil_n_actual_scalar)

        # reuse tmp tensor
        # 'pylint: disable=attribute-defined-outside-init
        self.yy2 = self.xx2_ub
        self._tailing_handle_vmaxs(self.yy1_ub, self.y1_max_ub, self.y1i_scalar, self.ceil_n_actual_scalar)
        self._tailing_handle_vmins(self.yy2, self.y2_max_ub, self.y2i_scalar, self.ceil_n_actual_scalar)
        self._tailing_handle_vsub(self.yy1_ub, self.yy2, self.yy1_ub, self.ceil_n_actual_scalar, self.bytes_each_elem,
                                  self.bytes_each_elem, self.bytes_each_elem)
        # h stores in yy1
        self._tailing_handle_vmaxs(self.yy1_ub, self.yy1_ub, self.zero_datatype_scalar, self.ceil_n_actual_scalar)
        # inter stores in xx1
        self._tailing_handle_vmul(self.xx1_ub, self.xx1_ub, self.yy1_ub, self.ceil_n_actual_scalar, None,
                                  self.bytes_each_elem, self.bytes_each_elem, self.bytes_each_elem)

        return self.xx1_ub

    def _tailing_handle_vmaxs(self, dst_ub, src_ub, scalar, size):
        """
        handle tailing of vmaxs

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src
        scalar: scalar
        size: totol size of elems

        Returns
        -------
        None
        """
        offset_reg = self.tik_instance.Scalar('int32', 'offset_reg', init_value=0)

        # repeat num
        repeat_reg = self.tik_instance.Scalar('int32',
                                              'repeat_reg',
                                              init_value=size % (self.vector_mask_max * Constant.REPEAT_TIMES_MAX) //
                                              self.vector_mask_max)

        with self.tik_instance.if_scope(repeat_reg > 0):
            self.tik_instance.vmaxs(mask=self.vector_mask_max,
                                    dst=dst_ub[offset_reg],
                                    src=src_ub[offset_reg],
                                    scalar=scalar,
                                    repeat_times=repeat_reg,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=8)

        # last num
        last_num_reg = self.tik_instance.Scalar('int32', 'last_num_reg', init_value=size % self.vector_mask_max)
        with self.tik_instance.if_scope(last_num_reg > 0):
            offset_reg.set_as(offset_reg + repeat_reg * self.vector_mask_max)
            self.tik_instance.vmaxs(mask=last_num_reg,
                                    dst=dst_ub[offset_reg],
                                    src=src_ub[offset_reg],
                                    scalar=scalar,
                                    repeat_times=1,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=8)

    def _tailing_handle_vmins(self, dst_ub, src_ub, scalar, size):
        """
        handle tailing of vmins

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src
        scalar: scalar
        size: totol size of elems

        Returns
        -------
        None
        """
        offset_reg = self.tik_instance.Scalar('int32', 'offset_reg', init_value=0)

        # repeat num
        repeat_reg = self.tik_instance.Scalar('int32',
                                              'repeat_reg',
                                              init_value=size % (self.vector_mask_max * Constant.REPEAT_TIMES_MAX) //
                                              self.vector_mask_max)

        with self.tik_instance.if_scope(repeat_reg > 0):
            self.tik_instance.vmins(mask=self.vector_mask_max,
                                    dst=dst_ub[offset_reg],
                                    src=src_ub[offset_reg],
                                    scalar=scalar,
                                    repeat_times=repeat_reg,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=8)

        # last num
        last_num_reg = self.tik_instance.Scalar('int32', 'last_num_reg', init_value=size % self.vector_mask_max)
        with self.tik_instance.if_scope(last_num_reg > 0):
            offset_reg.set_as(offset_reg + repeat_reg * self.vector_mask_max)
            self.tik_instance.vmins(mask=last_num_reg,
                                    dst=dst_ub[offset_reg],
                                    src=src_ub[offset_reg],
                                    scalar=scalar,
                                    repeat_times=1,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=8)

    def _tailing_handle_vmul(self,
                             dst_ub,
                             src0_ub,
                             src1_ub,
                             size,
                             mask_max=None,
                             dst_bytes=None,
                             src0_bytes=None,
                             src1_bytes=None):
        """
        handle tailing of vmul

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0
        src1_ub: src1
        size: totol size of elems
        mask_max: max. mask
        dst_bytes: dst bytes
        src0_bytes: src0 bytes
        src1_bytes: src1 bytes

        Returns
        -------
        None
        """
        if mask_max is None:
            mask_max = self.vector_mask_max

        offset_reg = self.tik_instance.Scalar('int32', 'offset_reg', init_value=0)

        # repeat num
        repeat_reg = self.tik_instance.Scalar('int32',
                                              'repeat_reg',
                                              init_value=size % (mask_max * Constant.REPEAT_TIMES_MAX) // mask_max)

        with self.tik_instance.if_scope(repeat_reg > 0):
            self.tik_instance.vmul(mask=mask_max,
                                   dst=dst_ub[offset_reg],
                                   src0=src0_ub[offset_reg],
                                   src1=src1_ub[offset_reg],
                                   repeat_times=repeat_reg,
                                   dst_blk_stride=1,
                                   src0_blk_stride=1,
                                   src1_blk_stride=1,
                                   dst_rep_stride=mask_max * dst_bytes // Constant.CONFIG_DATA_ALIGN,
                                   src0_rep_stride=mask_max * src0_bytes // Constant.CONFIG_DATA_ALIGN,
                                   src1_rep_stride=mask_max * src1_bytes // Constant.CONFIG_DATA_ALIGN)

        # last num
        last_num_reg = self.tik_instance.Scalar('int32', 'last_num_reg', init_value=size % mask_max)
        with self.tik_instance.if_scope(last_num_reg > 0):
            offset_reg.set_as(offset_reg + repeat_reg * mask_max)
            self.tik_instance.vmul(mask=last_num_reg,
                                   dst=dst_ub[offset_reg],
                                   src0=src0_ub[offset_reg],
                                   src1=src1_ub[offset_reg],
                                   repeat_times=1,
                                   dst_blk_stride=1,
                                   src0_blk_stride=1,
                                   src1_blk_stride=1,
                                   dst_rep_stride=8,
                                   src0_rep_stride=8,
                                   src1_rep_stride=8)

    def _area(self):
        """
        calculate area
        area = (x2-x1) * (y2-y1), this is vector computing
        area can be reused in loops

        Parameters
        ----------
        None


        Returns
        -------
        None
        """
        if self.total_areas_ub is not None:
            return self.total_areas_ub

        tik_instance = self.tik_instance
        self.total_areas_ub = tik_instance.Tensor(self.data_type, (self.ceil_n_max, ),
                                                  name="total_areas_ub",
                                                  scope=tik.scope_ubuf)

        # reuse tmp tensor xx2 for y2suby1
        y2suby1_ub = self.xx2_ub

        self._tailing_handle_vsub(self.total_areas_ub, self.x2_max_ub, self.x1_max_ub, self.ceil_n_actual_scalar,
                                  self.bytes_each_elem, self.bytes_each_elem, self.bytes_each_elem)

        self._tailing_handle_vabs(self.total_areas_ub, self.total_areas_ub, self.ceil_n_actual_scalar,
                                  self.bytes_each_elem, self.bytes_each_elem)

        self._tailing_handle_vsub(y2suby1_ub, self.y2_max_ub, self.y1_max_ub, self.ceil_n_actual_scalar,
                                  self.bytes_each_elem, self.bytes_each_elem, self.bytes_each_elem)

        self._tailing_handle_vmul(self.total_areas_ub, self.total_areas_ub, y2suby1_ub, self.ceil_n_actual_scalar,
                                  None, self.bytes_each_elem, self.bytes_each_elem, self.bytes_each_elem)

        return self.total_areas_ub

    def _tailing_handle_vadds(self, dst_ub, src_ub, scalar, size):
        """
        handle tailing of vadds

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src
        scalar: scalar
        size: totol size of elems

        Returns
        -------
        None
        """
        offset_reg = self.tik_instance.Scalar('int32', 'offset_reg', init_value=0)

        # repeat num
        repeat_reg = self.tik_instance.Scalar('int32',
                                              'repeat_reg',
                                              init_value=size % (self.vector_mask_max * Constant.REPEAT_TIMES_MAX) //
                                              self.vector_mask_max)

        with self.tik_instance.if_scope(repeat_reg > 0):
            self.tik_instance.vadds(mask=self.vector_mask_max,
                                    dst=dst_ub[offset_reg],
                                    src=src_ub[offset_reg],
                                    scalar=scalar,
                                    repeat_times=repeat_reg,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=8)

        # last num
        last_num_reg = self.tik_instance.Scalar('int32', 'last_num_reg', init_value=size % self.vector_mask_max)
        with self.tik_instance.if_scope(last_num_reg > 0):
            offset_reg.set_as(offset_reg + repeat_reg * self.vector_mask_max)
            self.tik_instance.vadds(mask=last_num_reg,
                                    dst=dst_ub[offset_reg],
                                    src=src_ub[offset_reg],
                                    scalar=scalar,
                                    repeat_times=1,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=8)

    def _tailing_handle_cmp_le_and_2bitmask(self, src0_ub, src1_ub, size):
        """
        combine vcmp_le() and cmpmask2bitmask()
        vcmp handle max. 128 mask, repeat = 1

        size: total size of proposals

        Parameters
        ----------
        src0_ub: src0 in ub
        src1_ub: src1 in ub

        Returns
        -------
        output_mask_int8_ub
        """
        loops_reg = self.tik_instance.Scalar("int32",
                                             "loops_reg",
                                             init_value=size // (self.vector_mask_max * Constant.BYTES_SIZE_INT8))
        offset_reg = self.tik_instance.Scalar('int32', 'offset_reg', init_value=0)

        # max. mask * max. repeat  * loops times
        with self.tik_instance.if_scope(loops_reg > 0):
            with self.tik_instance.for_range(0, loops_reg, name='loop_index') as loop_index:
                # vcmp only run once, so repeat = 1
                cmpmask = self.tik_instance.vcmp_le(
                    mask=self.vector_mask_max,
                    src0=src0_ub[offset_reg],
                    src1=src1_ub[offset_reg],
                    # 1 is fixed
                    src0_stride=1,
                    src1_stride=1)
                self._cmpmask2bitmask(dst_ub=self.max_output_mask_int8_ub[offset_reg],
                                      cmpmask=cmpmask,
                                      handle_dst_size=self.vector_mask_max)

                offset_reg.set_as((loop_index + 1) * self.vector_mask_max * Constant.BYTES_SIZE_INT8)

        # last num
        last_num_reg = self.tik_instance.Scalar('int32', 'last_num_reg', init_value=size % self.vector_mask_max)
        with self.tik_instance.if_scope(last_num_reg > 0):
            cmpmask = self.tik_instance.vcmp_le(mask=last_num_reg,
                                                src0=src0_ub[offset_reg],
                                                src1=src1_ub[offset_reg],
                                                src0_stride=1,
                                                src1_stride=1)
            self._cmpmask2bitmask(dst_ub=self.max_output_mask_int8_ub[offset_reg],
                                  cmpmask=cmpmask,
                                  handle_dst_size=last_num_reg)

        return self.max_output_mask_int8_ub

    def _cmpmask2bitmask(self, dst_ub, cmpmask, handle_dst_size):
        """
        in one repeat, handle max. 128 elems. so tensor defed below has 128 shape
        bitmask is like [1 0 1 1 0 0 0 1]

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        tik_instance.vsel(Constant.BURST_PROPOSAL_NUM, 0, self.output_mask_f16_ub, cmpmask, self.data_fp16_one_ub,
                          self.data_fp16_zero_ub, 1, 1, 1, 1, 8, 8, 8)

        tik_instance.vec_conv(handle_dst_size, "none", dst_ub, self.output_mask_f16_ub, 1, 8, 8)

    def _update_valid_mask(self, mask_ub_int8_ub):
        """
        update valid mask
        note: use vand instead of vmul, but vand only compute uint16/int16,
            so use int16 for out_mask, support f162s16 using round mode in cmpmask2bitmask()

        Parameters
        ----------
        mask_ub_int8_ub: which will be used to update valid_mask_ub

        Returns
        -------
        None
        """
        self._tailing_handle_vec_conv(self.tmp_valid_mask_float16_ub_max,
                                      self.max_valid_mask_int8_ub,
                                      self.actual_valid_mask_size_int8_scalar,
                                      dst_bytes=2,
                                      src_bytes=1)
        self._tailing_handle_vec_conv(self.tmp_mask_float16_ub_max,
                                      mask_ub_int8_ub,
                                      self.actual_valid_mask_size_int8_scalar,
                                      dst_bytes=2,
                                      src_bytes=1)

        # mask: [0 0 1 1] * [1 0 1 0] = [0 0 1 0]
        self._tailing_handle_vmul(self.tmp_valid_mask_float16_ub_max, self.tmp_valid_mask_float16_ub_max,
                                  self.tmp_mask_float16_ub_max, self.actual_valid_mask_size_int8_scalar,
                                  self.vector_mask_max, Constant.BYTES_SIZE_FP16, Constant.BYTES_SIZE_FP16,
                                  Constant.BYTES_SIZE_FP16)

        # float16 to int8
        self._tailing_handle_vec_conv(self.max_valid_mask_int8_ub,
                                      self.tmp_valid_mask_float16_ub_max,
                                      self.actual_valid_mask_size_int8_scalar,
                                      dst_bytes=1,
                                      src_bytes=2)

    def selected_boxes_gen(self):
        """
        selected_boxes generate from proposals_ub

        original box_scores: [N, 8]
        selected_boxes:      [N, 5]

        Parameters
        ----------
        None

        Returns
        -------
        selected_boxes_ub:
        """
        # 7967 is [1 1 1 1 1 0 0 0 1 1 1 1 1 0 0 0] for 16 inputs
        pattern_value_uint16 = 7967
        # 522133279 is [1 1 1 1 1 0 0 0   1 1 1 1 1 0 0 0   1 1 1 1 1 0 0 0
        # 1 1 1 1 1 0 0 0] one uint32 can handle selection of 32 elems
        pattern_value_uint32 = 522133279
        # def selected_boxes_ub
        # ceil n 64 align
        selected_boxes_ub = self.tik_instance.Tensor(self.input_dtype, (self.ceil_n_max, Constant.VALID_COLUMN_NUM),
                                                     tik.scope_ubuf, 'selected_boxes_ub')

        # create pattern, shape is 16 or 8, which is enough and it'll be reused in vreduce, and vreduce output
        if self.input_dtype == 'float16':
            pattern_ub = self.tik_instance.Tensor('uint16', (16, ), tik.scope_ubuf, 'pattern_ub')
            # init pattern
            self.tik_instance.vector_dup(
                16, pattern_ub, self.tik_instance.Scalar('uint16', 'pattern_s', init_value=pattern_value_uint16), 1, 1,
                1)

            repeat_reg = _ceil_div_scalar(self.tik_instance, self.n_actual_scalar * Constant.ELEMENT_NUM,
                                          Constant.BURST_PROPOSAL_NUM)

            # only select first 5 elements
            with self.tik_instance.for_range(0, repeat_reg) as i:
                offset = i * Constant.BURST_PROPOSAL_NUM
                self.tik_instance.data_move(
                    self.tmp_tensor_ub_fp16_burst, self.all_inp_proposals_gm_max[offset], 0, 1,
                    Constant.BURST_PROPOSAL_NUM * self.input_bytes_each_elem // Constant.CONFIG_DATA_ALIGN, 0, 0)
                self.tik_instance.vreduce(mask=Constant.BURST_PROPOSAL_NUM,
                                          dst=selected_boxes_ub[offset * Constant.VALID_COLUMN_NUM //
                                                                Constant.ELEMENT_NUM],
                                          src0=self.tmp_tensor_ub_fp16_burst,
                                          src1_pattern=pattern_ub,
                                          repeat_times=1,
                                          src0_blk_stride=1,
                                          src0_rep_stride=0,
                                          src1_rep_stride=0)
        else:
            pattern_ub = self.tik_instance.Tensor('uint32', (8, ), tik.scope_ubuf, 'pattern_ub')
            self.tik_instance.vector_dup(
                8, pattern_ub, self.tik_instance.Scalar('uint32', 'pattern_s', init_value=pattern_value_uint32), 1, 1,
                1)
            self._tailing_handle_vreduce_output(selected_boxes_ub, self.all_inp_proposals_ub_fp32_max, pattern_ub)

        return selected_boxes_ub

    def _update_next_nonzero_idx(self, valid_mask_int8_ub):
        """
        update next nonzero idx
        note that using fp16 for dsorts_ub,valid_mask_fp16_ub,vcmax_ub... may cause precision problem if N > 2048

        Parameters
        ----------
        valid_mask: is [0 0 1 1 ]

        Returns
        -------
        tensor[0] contains the next nonzero idx
        """
        # int8 conv to fp16
        self._tailing_handle_vec_conv(self.valid_mask_fp16_ub_max,
                                      valid_mask_int8_ub,
                                      size=self.actual_valid_mask_size_int8_scalar,
                                      dst_bytes=2,
                                      src_bytes=1)

        # already compute repeat and last_num in _init_for_vcmax()
        repeat_reg = self.tik_instance.Scalar('int32', 'repeat_reg', init_value=self.repeat_vmul_vcmax_actual_scalar)
        last_num_reg = self.tik_instance.Scalar('int32',
                                                'last_num_reg',
                                                init_value=self.last_num_vmul_vcmax_actual_scalar)

        # vmul
        with self.tik_instance.if_scope(repeat_reg > 0):
            self.tik_instance.vmul(Constant.MASK_VCMAX_FP16, self.valid_mask_fp16_ub_max, self.valid_mask_fp16_ub_max,
                                   self.dsorts_ub, repeat_reg, 1, 1, 1, 8, 8, 8)

        with self.tik_instance.if_scope(last_num_reg > 0):
            vmul_offset_reg = self.tik_instance.Scalar('int32',
                                                       'vmul_offset_reg',
                                                       init_value=repeat_reg * Constant.MASK_VCMAX_FP16)
            self.tik_instance.vmul(last_num_reg, self.valid_mask_fp16_ub_max[vmul_offset_reg],
                                   self.valid_mask_fp16_ub_max[vmul_offset_reg], self.dsorts_ub[vmul_offset_reg], 1, 1,
                                   1, 1, 8, 8, 8)

        # vcmax
        with self.tik_instance.if_scope(repeat_reg > 0):
            self.tik_instance.vcmax(Constant.MASK_VCMAX_FP16, self.vcmax_ub, self.valid_mask_fp16_ub_max, repeat_reg,
                                    1, 1, 8)

        with self.tik_instance.if_scope(last_num_reg > 0):
            offset_reg = self.tik_instance.Scalar('int32',
                                                  'offset_reg',
                                                  init_value=repeat_reg * Constant.MASK_VCMAX_FP16)
            self.tik_instance.vcmax(last_num_reg, self.vcmax_ub[repeat_reg * 2],
                                    self.valid_mask_fp16_ub_max[offset_reg], 1, 1, 1, 8)

        # pattern here means 101010..., vreduce once is enough
        self.tik_instance.vreduce(Constant.MASK_VCMAX_FP16,
                                  self.middle_max_val_ub,
                                  self.vcmax_ub,
                                  src1_pattern=1,
                                  repeat_times=1,
                                  src0_blk_stride=1,
                                  src0_rep_stride=0,
                                  src1_rep_stride=0)

        # below: dst_max_val_ub[0], idx_fp16_ub[0], next_nonzero_int32_idx_ub[0] stores meaningful val
        self.tik_instance.vcmax(self.vcmax_mask_actual_scalar, self.dst_max_val_ub, self.middle_max_val_ub, 1, 0, 1, 0)

        # dst idx, note: idx maybe valid_mask_size
        self.tik_instance.vsub(Constant.SHAPE_NEXT_NONZERO, self.idx_fp16_ub, self.dsorts_ub, self.dst_max_val_ub,
                               1, 1, 1, 1, 8, 8, 8)

        # conv to int32
        self._tailing_handle_vec_conv(self.next_nonzero_int32_idx_ub,
                                      self.idx_fp16_ub,
                                      Constant.SHAPE_NEXT_NONZERO,
                                      Constant.BYTES_SIZE_INT32,
                                      Constant.BYTES_SIZE_FP16,
                                      mode='round')

    def _tailing_handle_vreduce_output(self, dst_ub, src0_ub, src1_pattern_ub):
        """
        [N, 8] => [N, 5]

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0 in ub
        src1_pattern_ub: pattern for src1

        Returns
        -------
        None
        """
        # `info: =16 for fp16, =8 for fp32. here 8 is ncols
        vector_proposals_max = self.input_vector_mask_max // 8
        offset_reg = self.tik_instance.Scalar('int32', 'offset_reg', init_value=0)

        # repeat num
        repeat_reg = self.tik_instance.Scalar('int32',
                                              'repeat_reg',
                                              init_value=self.ceil_n_actual_scalar %
                                              (vector_proposals_max * Constant.REPEAT_TIMES_MAX) //
                                              vector_proposals_max)

        with self.tik_instance.if_scope(repeat_reg > 0):
            dst_offset_reg = self.tik_instance.Scalar('int32', 'dst_offset_reg', init_value=offset_reg * 5)
            src_offset_reg = self.tik_instance.Scalar('int32', 'src_offset_reg', init_value=offset_reg * 8)

            self.tik_instance.vreduce(mask=self.input_vector_mask_max,
                                      dst=dst_ub[dst_offset_reg],
                                      src0=src0_ub[src_offset_reg],
                                      src1_pattern=src1_pattern_ub,
                                      repeat_times=repeat_reg,
                                      src0_blk_stride=1,
                                      src0_rep_stride=self.vector_mask_max * self.input_bytes_each_elem \
                                                      // Constant.CONFIG_DATA_ALIGN,
                                      src1_rep_stride=0)

        # last num
        last_num_reg = self.tik_instance.Scalar('int32',
                                                'last_num_reg',
                                                init_value=self.ceil_n_actual_scalar % vector_proposals_max)
        with self.tik_instance.if_scope(last_num_reg > 0):
            offset_reg.set_as(offset_reg + repeat_reg * vector_proposals_max)
            dst_offset_reg = self.tik_instance.Scalar('int32', 'dst_offset_reg', init_value=offset_reg * 5)
            src_offset_reg = self.tik_instance.Scalar('int32', 'src_offset_reg', init_value=offset_reg * 8)
            last_num_reg.set_as(last_num_reg * 8)
            self.tik_instance.vreduce(mask=last_num_reg,
                                      dst=dst_ub[dst_offset_reg],
                                      src0=src0_ub[src_offset_reg],
                                      src1_pattern=src1_pattern_ub,
                                      repeat_times=1,
                                      src0_blk_stride=1,
                                      src0_rep_stride=0,
                                      src1_rep_stride=0)


# 'pylint: disable=too-many-locals,too-many-arguments
def _nms_with_mask_compute(tik_instance, input_boxes_max_num, input_dtype, thresh, total_output_proposal_num,
                           boxes_num_scalar, tiling_gm, kernel_name_var):
    """
    Compute output boxes after non-maximum suppression

    Parameters
    ----------
    input_shape: dict
        shape of input boxes, including proposal boxes and corresponding confidence scores

    input_dtype: str
        input data type: options are float16 and float32

    thresh: float
        iou threshold

    total_output_proposal_num: int
        the number of output proposal boxes

    kernel_name: str
        cce kernel name

    Returns
    -------
    tik_instance: TIK API
    """
    nms_helper = _NMSHelper(tik_instance, input_boxes_max_num, input_dtype, thresh, boxes_num_scalar)

    proposals_gm = nms_helper.all_inp_proposals_gm_max

    output_proposals_ub = nms_helper.selected_boxes_gen()
    output_index_ub = nms_helper.selected_idx_ub
    output_mask_ub = nms_helper.loops()

    # data move from ub to gm. def tensor in gm can be real shape, dont need to ceiling
    out_proposals_gm = tik_instance.Tensor(input_dtype, (total_output_proposal_num, Constant.VALID_COLUMN_NUM),
                                           name="out_proposals_gm",
                                           scope=tik.scope_gm)
    # address is 32B aligned
    out_index_gm = tik_instance.Tensor("int32", (total_output_proposal_num, ), name="out_index_gm", scope=tik.scope_gm)
    out_mask_gm = tik_instance.Tensor("uint8", (total_output_proposal_num, ), name="out_mask_gm", scope=tik.scope_gm)

    # `max. burst is 65535, unit is 32B, so support: 65535*32/2/8=131070 proposals if fp16.`
    tik_instance.data_move(out_proposals_gm, output_proposals_ub, 0, nburst=1,
                           burst=(nms_helper.ceil_n_actual_scalar * Constant.VALID_COLUMN_NUM * \
                                  nms_helper.bytes_each_elem // Constant.CONFIG_DATA_ALIGN),
                           src_stride=0, dst_stride=0)
    tik_instance.data_move(out_index_gm,
                           output_index_ub,
                           0,
                           nburst=1,
                           burst=(nms_helper.ceil_n_actual_scalar * Constant.BYTES_SIZE_INT32 //
                                  Constant.CONFIG_DATA_ALIGN),
                           src_stride=0,
                           dst_stride=0)
    # here need _ceiling() as ceilN can be 16; 16*1//32=0 is wrong
    tik_instance.data_move(
        out_mask_gm,
        output_mask_ub,
        0,
        nburst=1,
        burst=_ceiling(nms_helper.ceil_n_actual_scalar * Constant.BYTES_SIZE_UINT8, Constant.CONFIG_DATA_ALIGN) //
        Constant.CONFIG_DATA_ALIGN,
        src_stride=0,
        dst_stride=0)

    tik_instance.BuildCCE(kernel_name=kernel_name_var,
                          inputs=[proposals_gm],
                          outputs=[out_proposals_gm, out_index_gm, out_mask_gm],
                          flowtable=(tiling_gm, ),
                          output_files_path=None,
                          enable_l2=False)
    return tik_instance


def _cal_used_ub_remain_size(input_dtype):
    """
    used size in ub

    Parameters
    ----------
    N: int
        value of input_shape[0]
    input_dtype: str
        input data type

    Returns
    -------
    size used in ub
    """
    ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

    if input_dtype == 'float16':
        input_bytes_each_elem = Constant.BYTES_SIZE_FP16
    elif input_dtype == 'float32':
        input_bytes_each_elem = Constant.BYTES_SIZE_FP32
    else:
        error_manager_vector.raise_err_specific_reson("nms_with_mask",
                                                      "nms_with_mask's first input only support float16 and float32!")
    vector_mask_max = Constant.BURST_PROPOSAL_NUM // 2

    next_nonzero_size = Constant.SHAPE_NEXT_NONZERO * Constant.BYTES_SIZE_INT32
    output_mask_f16_size = Constant.BURST_PROPOSAL_NUM * Constant.BYTES_SIZE_FP16
    # size of data_fp16_zero and data_fp16_one
    data_fp16_zero_one_size = Constant.BURST_PROPOSAL_NUM * Constant.BYTES_SIZE_FP16 * 2
    # size used in _input_trans(), 16 is size of pattern, 4 means x1/y1/x2/y2
    input_trans_size = 8 * Constant.BYTES_SIZE_UINT32 * 4
    pattern_size = 16 * Constant.BYTES_SIZE_UINT16

    ub_remain_tmp = ub_size_bytes - next_nonzero_size + output_mask_f16_size + data_fp16_zero_one_size + \
                    input_trans_size + pattern_size

    bytes_each_elem = Constant.BYTES_SIZE_FP32

    ceil_n_related_num = bytes_each_elem * 4 + Constant.ELEMENT_NUM * bytes_each_elem + Constant.BYTES_SIZE_FP16 + \
                         Constant.BYTES_SIZE_INT8 + bytes_each_elem * 3 + Constant.BYTES_SIZE_INT8 + \
                         Constant.BYTES_SIZE_FP16 + Constant.BYTES_SIZE_FP16 + Constant.VALID_COLUMN_NUM * \
                         input_bytes_each_elem + Constant.BYTES_SIZE_INT32 + bytes_each_elem + \
                         Constant.BYTES_SIZE_UINT8

    remain_ub = ub_remain_tmp - Constant.BURST_PROPOSAL_NUM * Constant.BYTES_SIZE_FP16 - (
        Constant.MASK_VCMAX_FP16 * 2 + Constant.SHAPE_NEXT_NONZERO * 2) * Constant.BYTES_SIZE_FP16 - 4096

    max_n_num = remain_ub / ceil_n_related_num

    boxes_num_max_align = max_n_num // vector_mask_max * vector_mask_max
    return boxes_num_max_align


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
@register_operator("NMSWithMask")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def nms_with_mask_single_core(box_scores,
                              selected_boxes,
                              selected_idx,
                              selected_mask,
                              iou_thr,
                              kernel_name="nms_with_mask"):
    """
    algorithm: nms_with_mask

    find the best target bounding box and eliminate redundant bounding boxes

    Parameters
    ----------
    box_scores: dict
        2-D shape and dtype of input tensor, only support [N, 8]
        including proposal boxes and corresponding confidence scores

    selected_boxes: dict
        2-D shape and dtype of output boxes tensor, only support [N,5]
        including proposal boxes and corresponding confidence scores

    selected_idx: dict
        the index of output proposal boxes

    selected_mask: dict
        the symbol judging whether the output proposal boxes is valid

    iou_thr: float
        iou threshold

    kernel_name: str
        cce kernel name, default value is "nms_with_mask"

    Returns
    -------
    None
    """
    input_shape = box_scores.get("shape")
    input_dtype = box_scores.get("dtype").lower()

    # check dtype
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="box_scores")
    # check shape
    para_check.check_shape(input_shape,
                           min_rank=Constant.INPUT_DIMS,
                           max_rank=Constant.INPUT_DIMS,
                           param_name="box_scores")
    if is_unknown_rank_input([box_scores]):
        input_shape = (-1, Constant.ELEMENT_NUM)
        box_scores["shape"] = input_shape
    if input_shape[1] != Constant.ELEMENT_NUM:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "the 2nd-dim of input boxes must be equal to 8",
                                                          "box_scores.shape", input_shape)

    tik_instance = tik.Tik()
    tiling_gm = tik_instance.Tensor("int32", (Constant.TILING_PARAMS_NUM, ), name="tiling_gm", scope=tik.scope_gm)
    tiling_ub = tik_instance.Tensor("int32", (Constant.TILING_PARAMS_NUM, ), name="tiling_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(tiling_ub, tiling_gm, 0, 1, Constant.TILING_PARAMS_NUM // Constant.BLOCK_INT32, 0, 0)

    boxes_num_scalar = tik_instance.Scalar(dtype="int32", name="boxes_num_scalar", init_value=0)
    boxes_num_scalar.set_as(tiling_ub[0])

    input_boxes_max_num = _cal_used_ub_remain_size(input_dtype)
    output_size = input_boxes_max_num

    if iou_thr is None:
        iou_thr = tik_instance.Scalar(dtype="float32", name="iou_thr")
        iou_thr.set_as(tiling_ub[1])
    # add compile info
    tbe_context.get_context().add_compile_info("vars", {"max_boxes_num": input_boxes_max_num})

    return _nms_with_mask_compute(tik_instance, input_boxes_max_num, input_dtype, iou_thr, output_size,
                                  boxes_num_scalar, tiling_gm, kernel_name)
