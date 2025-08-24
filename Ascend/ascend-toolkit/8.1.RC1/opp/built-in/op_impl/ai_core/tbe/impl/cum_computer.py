# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
cum_computer
"""
from collections import namedtuple
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik

from impl.constant_util import DATA_TYPE_FP16
from impl.constant_util import BLOCK_SIZE
from impl.constant_util import DATA_TYPE_INT8
from impl.constant_util import VECTOR_BYTE_SIZE
from impl.constant_util import STRIDE_ONE
from impl.constant_util import REPEAT_STRIDE_EIGHT
from impl.constant_util import DEFAULT_BURST_LEN
from impl.constant_util import DEFAULT_REPEAT_TIME
from impl.constant_util import STRIDE_ZERO
from impl.constant_util import DATA_TYPE_UINT8
from impl.common_util import get_data_size


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    Constant
    """
    # A maximum of 25k can be calculated at a time.
    MAX_COMPUTE_SIZE = 25 * 1024

    # tensor num for sum/prod op 
    UB_TENSOR_NUM_THREE = 3

    # tensor num for logsumexp op
    UB_TENSOR_NUM_FIVE = 5

    # tensor num times: 2(db buffer) + 1(tail)
    UB_TENSOR_NUM_TIMES = 3

    # const value 1
    VALUE_ONE = 1

    # const value 0
    VALUE_ZERO = 0

    # const value 2
    VALUE_TWO = 2

    # const value -1
    NEG_ONE = -1

    # repeat stride 4 for vconv
    STRIDE_FOUR = 4

    # type of cumsum op
    SUM_TYPE = "sum"

    # type of cumprod op
    PROD_TYPE = "prod"

    # type of cumlogsumexp
    LOGSUMEXP_TYPE = "logsumexp"

    # handle position of tail
    TAIL = "tail"

    # handle postiion of head
    HEAD = "head"


# 'pylint: disable=useless-object-inheritance,too-many-statements,too-many-locals
# 'pylint: disable=too-many-lines,too-many-instance-attributes
class CumBase(object):
    """
        Function: use to store cumsum base parameters
        Modify : 2022-07-18
    """

    def __init__(self, shape, axis, dtype, ctype):
        """
        init the base param of cumsum

        Parameters
        ----------
        shape: the shape of tensor
        axis: cumulative axis
        dtype: the data type of tensor
        ctype: computer type , "sum" "prod" or "logsumexp"

        Returns
        -------
        None

        """
        self.tik_instance = tik.Tik()
        self.each_loop = shape[axis]
        self.dsize = get_data_size(dtype)
        self.each, self.each_tail = self.get_each(shape, axis)
        self.reserved = self.get_reserved()
        self.dtype = dtype
        self.ctype = ctype
        self.axis = axis
        self.is_last_axis = (axis - len(shape)) == -1

    def get_each(self, shape, axis):
        """
        Calculate the length of each separate accumulation.

        Parameters
        ----------
        shape: the shape of tensor
        axis: cumulative axis

        Returns
        -------
        each: the length of each separate accumulation

        """
        self.each = Constant.VALUE_ONE
        for k in range(axis + Constant.VALUE_ONE, len(shape)):
            self.each = self.each * shape[k]
        each_tail = self.each % (BLOCK_SIZE // self.dsize)

        return self.each, each_tail

    def get_reserved(self):
        """
        Prevent tensor overflow and calculate the additional length.

        Returns
        -------
        reserved: the additional length

        """
        if self.each * self.dsize % BLOCK_SIZE != Constant.VALUE_ZERO:
            reserved = BLOCK_SIZE // (self.each_loop * self.each) + Constant.VALUE_ONE
        else:
            reserved = Constant.VALUE_ZERO

        return reserved


# 'pylint: disable=global-statement,super-with-arguments
class CumTensor(CumBase):
    """
        Function: use to store cumsum tensor
        Modify : 2022-07-18
    """

    def __init__(self, shape, axis, dtype, ctype):
        """
        init cumsum tensor

        Parameters
        ----------
        shape: the shape of tensor
        axis: cumulative axis
        dtype: the data type of tensor
        ctype: computer type , "sum" "prod" or "logsumexp"

        Returns
        -------
        None

        """
        super(CumTensor, self).__init__(shape, axis, dtype, ctype)
        # maximum size of ub tensor can be calculated at a time.
        self.ub_tensor_size = int(self.get_ub_tensor_size())
        self.mov_len = int(self.get_mov_len())
        self.mov_loop = int(self.each // self.mov_len)
        self.mov_tail = int((self.each - self.mov_len * self.mov_loop) % self.mov_len)
        self.rdtype = DATA_TYPE_FP16 if dtype in (DATA_TYPE_UINT8, DATA_TYPE_INT8) else dtype
        self.rdsize = Constant.VALUE_TWO if dtype in (DATA_TYPE_UINT8, DATA_TYPE_INT8) else self.dsize
        self.mask = VECTOR_BYTE_SIZE // self.rdsize

    def check_dtype_in_u8s8(self):
        """
        check data type whether in int8 or uint8

        Returns
        -------
        None

        """
        return self.dtype in (DATA_TYPE_UINT8, DATA_TYPE_INT8)

    def get_total_loop(self, shape, axis):
        """
        get total loop of shape

        Parameters
        ----------
        shape: the shape of tensor
        axis: the cumulative axis

        Returns
        -------
        total_loop: the total loop

        """
        self.dsize = self.dsize
        total_loop = Constant.VALUE_ONE
        for j in range(Constant.VALUE_ZERO, axis):
            total_loop = total_loop * shape[j]

        return total_loop

    def get_ub_tensor_size(self):
        """
        Calculate maximum of ub tensor size calculated at a time.

        Returns
        -------
        the maximum size of ub tensor at a time

        """
        if self.ctype == Constant.LOGSUMEXP_TYPE:
            ub_tensor_num = Constant.UB_TENSOR_NUM_FIVE
        else:
            ub_tensor_num = Constant.UB_TENSOR_NUM_THREE
        ub_tensor_num = ub_tensor_num * Constant.UB_TENSOR_NUM_TIMES
        ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        ub_size = (ub_size // ub_tensor_num - VECTOR_BYTE_SIZE) // VECTOR_BYTE_SIZE * VECTOR_BYTE_SIZE
        if ub_size > Constant.MAX_COMPUTE_SIZE:
            ub_size = Constant.MAX_COMPUTE_SIZE
        return ub_size

    def get_mov_len(self):
        """
        Calculate the size of one move.

        Returns
        -------
        mov_len: the size of one move

        """
        max_size = self.ub_tensor_size
        rdsize = Constant.VALUE_TWO if self.check_dtype_in_u8s8() else self.dsize
        if max_size >= (self.each * rdsize):
            mov_len = self.each
        else:
            mov_len = max_size // rdsize

        return mov_len

    def get_temp_ubtensor(self):
        """
        get temp tensor for multi core

        Returns
        -------
        last_32b: ub temp tensor

        """

        return self.tik_instance.Tensor(self.dtype, (BLOCK_SIZE,), tik.scope_ubuf, "last_32B")

    def get_outer_loop(self, shape, axis):
        """
        Calculate the number of times the outer loop.

        Parameters
        ----------
        shape: the shape of tensor
        axis: cumulative axis

        Returns
        -------
        outer_loop: the number of times the outer loop

        """
        total_loop = self.get_total_loop(shape, axis)

        block_num = tbe_platform.get_soc_spec(
            tbe_platform.CORE_NUM)
        if block_num > total_loop:
            outer_loop = Constant.VALUE_ONE
            block_num = total_loop
            outer_tail = Constant.VALUE_ZERO
        else:
            outer_loop = total_loop // block_num
            outer_tail = total_loop - block_num * outer_loop
        if self.each * self.dsize < BLOCK_SIZE or (
                self.mov_tail > Constant.VALUE_ZERO and self.mov_tail
                * self.dsize < BLOCK_SIZE):
            block_num = Constant.VALUE_ONE
            outer_loop = total_loop
            outer_tail = Constant.VALUE_ZERO
        return_args = namedtuple("args", "block_num total_loop outer_loop outer_tail")
        return return_args(block_num, total_loop, outer_loop, outer_tail)


class CumTilingParam(CumTensor):
    """
        Function: Used to calculate the tiling parameter.
        Modify: 2022-07-18
    """

    def __init__(self, shape, axis, dtype, ctype):
        """
        init the tiling input param

        Parameters
        ----------
        shape: the shape of tensor
        axis: cumulative axis
        dtype: the data type of tensor
        ctype: computer type , "sum" "prod" or "logsumexp"

        Returns
        -------
        None
        """
        super(CumTilingParam, self).__init__(shape, axis, dtype, ctype)
        self.block_num, self.total_loop, self.outer_loop, self.outer_tail = \
            self.get_outer_loop(shape, axis)
        self.exclusive = False
        self.reverse = False

    def get_repeat(self, length):
        """
        Calculate the times of repeat

        Returns
        -------
        repeat: the times of repeat

        """
        # head 256B align, tail 32B align
        if length * self.rdsize % VECTOR_BYTE_SIZE == Constant.VALUE_ZERO:
            repeat = length * self.rdsize // VECTOR_BYTE_SIZE
        else:
            repeat = length * self.rdsize // VECTOR_BYTE_SIZE + Constant.VALUE_ONE

        return repeat

    def get_offset(self, e_cycle):
        """
        Calculate the offset.

        Parameters
        ----------
        e_cycle: the loop time of each

        Returns
        -------
        in_offset: the offset of move in
        out_offset: the offset of move out

        """
        if not self.exclusive and self.reverse:
            in_offset = self.each_loop - Constant.VALUE_ONE - Constant.VALUE_TWO * e_cycle
            out_offset = self.each_loop - Constant.VALUE_ONE - Constant.VALUE_TWO * e_cycle
        elif self.exclusive and self.reverse:
            in_offset = self.each_loop - Constant.VALUE_TWO * e_cycle
            out_offset = self.each_loop - Constant.VALUE_ONE - Constant.VALUE_TWO * e_cycle
        elif self.exclusive and not self.reverse:
            in_offset = Constant.NEG_ONE
            out_offset = Constant.VALUE_ZERO
        else:
            in_offset = Constant.VALUE_ZERO
            out_offset = Constant.VALUE_ZERO

        return in_offset, out_offset

    def get_burlen_by_mlen(self, length):
        """
        Calculate the tail length of a move in instruct

        Returns
        -------
        burstlen: the tail length of a move in instruct

        """
        if (length * self.dsize) % BLOCK_SIZE == Constant.VALUE_ZERO:
            burstlen = length * self.dsize // BLOCK_SIZE
        else:
            burstlen = length * self.dsize // BLOCK_SIZE + Constant.VALUE_ONE

        return burstlen

    def set_ext_params(self, exclusive, reverse):
        """
        set expansion param

        Parameters
        ----------
        exclusive: if `True`, perform exclusive cumsum
        reverse: indicates whether to reverse calculation

        Returns
        -------
        None

        """
        self.exclusive = exclusive
        self.reverse = reverse


# 'pylint: disable=too-many-arguments
class CumComputer(CumTilingParam):
    """
        Function: use to compute the cumsum
        Modify: 2019-10-08
    """
    def __init__(self, input_x, axis, kernel_name, ctype):
        """
        init the input param

        Parameters
        ----------
        input_x: shape and dtype
        axis: cumulative axis
        kernel_name: kernel name
        ctype: computer type , "sum" "prod" or "logsumexp"

        """
        super(CumComputer, self).__init__(input_x.get("shape"), axis,
                                          input_x.get("dtype"), ctype)
        self.kernel_name = kernel_name
        self.need_special, self.spe_position = self.get_multi_special_position()
        # max value scalar
        if ctype == Constant.LOGSUMEXP_TYPE:
            self.scalar_one = self.tik_instance.Scalar(dtype=self.dtype,
                                                       init_value=1.0)

        # gm tensor
        self.input_x_gm = self.tik_instance \
            .Tensor(self.dtype, (self.total_loop + self.reserved,
                                 self.each_loop, self.each),
                    name="input_x_gm",
                    scope=tik.scope_gm)
        self.output_out_gm = self.tik_instance. \
            Tensor(self.dtype, (self.total_loop + self.reserved,
                                self.each_loop, self.each),
                   name="output_out_gm",
                   scope=tik.scope_gm)

    def get_multi_special_position(self):
        """
        Calculate the position where the multi-core special processing is
        required.

        Returns
        -------
        need_special: where the multi-core special processing is required
        position: the position

        """
        need_special = False
        position = Constant.HEAD
        if self.block_num > Constant.VALUE_ONE and self.each_tail != Constant.VALUE_ZERO \
                and self.each * self.dsize > BLOCK_SIZE:
            need_special = True
            position = Constant.HEAD if self.mov_tail == Constant.VALUE_ZERO else Constant.TAIL

        return need_special, position

    def post_multicore(self, burlen, idx, real_out, tail_idx):
        """
        Multi-core postprocessing

        Parameters
        ----------
        burlen: the param of dma instruction
        idx: index
        real_out: ub out
        tail_idx: tail index

        Returns
        -------
        last_32B: last 32B store data not aligned with 32B

        """
        last_32b = self.get_temp_ubtensor()
        self.tik_instance.data_move(last_32b, self.output_out_gm[
            idx[0], idx[1], idx[2] + tail_idx],
                                    Constant.VALUE_ZERO, Constant.VALUE_ONE, Constant.VALUE_ONE,
                                    STRIDE_ZERO, STRIDE_ZERO)
        tmp_scalar = self.tik_instance.Scalar(self.dtype)
        for i in range(self.each_tail):
            tmp_scalar.set_as(real_out[burlen * BLOCK_SIZE // self.dsize + i])
            last_32b[BLOCK_SIZE // self.dsize - self.each_tail + i] \
                .set_as(tmp_scalar)

        return last_32b

    def pre_multicore(self, burlen, position):
        """
        Multi-core preprocessing

        Parameters
        ----------
        burlen: the param of dma instruction
        position: the position of tensor

        Returns
        -------
        burlen: Processed burlen
        tail_idx: Indicates the offset index

        """
        tail_idx = Constant.VALUE_ZERO
        if self.need_special and position == self.spe_position:
            burlen = burlen - Constant.VALUE_ONE
            tail_idx = burlen * BLOCK_SIZE // self.dsize + \
                       self.each_tail - BLOCK_SIZE // self.dsize

        return burlen, tail_idx

    def get_tik_instance(self):
        """
        get the instance of tik

        Returns
        -------
        tik_instance: the instance of tik

        """
        self.cum_computer()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.input_x_gm,),
                                   outputs=(self.output_out_gm,),
                                   enable_l2=False)

        return self.tik_instance

    def t_vdup_to_gm(self, last_ret, last_ori, idx, position):
        """
        It is used to process the special first axis and is compatible with
        u8s8.

        Parameters
        ----------
        last_ret: Stores temporary results.
        last_ori: Used to store the original type.
        idx: index
        position: the position of each , "head" or "tail"

        Returns
        -------
        None

        """
        if position == Constant.HEAD:
            burlen = self.get_burlen_by_mlen(self.mov_len)
            repeat = self.get_repeat(self.mov_len)
        else:
            burlen = self.get_burlen_by_mlen(self.mov_tail)
            repeat = self.get_repeat(self.mov_tail)

        # Check whether multi-core special processing is required.
        burlen, tail_idx = self.pre_multicore(burlen, position)
        # SUM_TYE: 0, Constant.PROD_TYPE: 1, LOGSUMEXP: min
        if self.ctype == Constant.SUM_TYPE:
            value = Constant.VALUE_ZERO
        elif self.ctype == Constant.PROD_TYPE:
            value = Constant.VALUE_ONE
        elif self.ctype == Constant.LOGSUMEXP_TYPE:
            if self.dtype == "float16":
                value = -2 ** 15 * 1.9991
            elif self.dtype == "float32":
                value = -2 ** 127 * 1.9999999

        self.tik_instance.vector_dup(self.mask, last_ret, value, repeat,
                                     STRIDE_ONE, REPEAT_STRIDE_EIGHT)
        ub_out = last_ret
        if self.check_dtype_in_u8s8():
            self.tik_instance.vconv(self.mask, "", last_ori, last_ret,
                                    repeat, STRIDE_ONE,
                                    STRIDE_ONE, Constant.STRIDE_FOUR,
                                    REPEAT_STRIDE_EIGHT)
            ub_out = last_ori
        if burlen != 0:
            self.tik_instance.data_move(self.output_out_gm[idx], ub_out,
                                        Constant.VALUE_ZERO, DEFAULT_BURST_LEN,
                                        burlen, STRIDE_ZERO, STRIDE_ZERO)
        if self.need_special and position == self.spe_position:
            last_32b = self.post_multicore(burlen, idx, ub_out, tail_idx)

            self.tik_instance.data_move(
                self.output_out_gm[idx[0], idx[1], idx[2] + tail_idx],
                last_32b, Constant.VALUE_ZERO, DEFAULT_BURST_LEN,
                Constant.VALUE_ONE, STRIDE_ZERO, STRIDE_ZERO)
        if self.ctype == Constant.LOGSUMEXP_TYPE:
            if self.need_special and self.spe_position == position:
                offset = Constant.VALUE_ONE
            else:
                offset = Constant.VALUE_ZERO

            self.tik_instance.data_move(last_ret, last_ori, Constant.VALUE_ZERO,
                                        DEFAULT_BURST_LEN, burlen+offset,
                                        STRIDE_ZERO, STRIDE_ZERO)

    def t_dma_in(self, ub_in, ori, idx, burlen):
        """
        The dma command is compatible with the u8s8 type.

        Parameters
        ----------
        ub_in: ub tensor
        ori: ub tensor with original type
        idx: index
        burlen: dma param

        Returns
        -------
        None

        """
        real_in = ori if self.check_dtype_in_u8s8() else ub_in
        repeat = self.get_repeat(self.mov_len)

        self.tik_instance.data_move(real_in, self.input_x_gm[idx], Constant.VALUE_ZERO,
                                    DEFAULT_BURST_LEN, burlen, STRIDE_ZERO, STRIDE_ZERO)

        if self.check_dtype_in_u8s8():
            self.tik_instance.vconv(self.mask, "", ub_in, ori, repeat,
                                    Constant.VALUE_ONE, Constant.VALUE_ONE, REPEAT_STRIDE_EIGHT, Constant.STRIDE_FOUR)

    def t_dma_direct_out(self, last_ret, last_ori, idx, position):
        """
        The dma command is compatible with the u8s8 type.

        Parameters
        ----------
        last_ret: the tensor store last result
        last_ori: the tensor store last result with original type
        idx: index
        position: the position of each , "head" or "tail"

        Returns
        -------
        None

        """
        if position == Constant.HEAD:
            burlen = self.get_burlen_by_mlen(self.mov_len)
        else:
            burlen = self.get_burlen_by_mlen(self.mov_tail)

        # Check whether multi-core special processing is required.
        burlen, tail_idx = self.pre_multicore(burlen, position)

        real_out = last_ori if self.check_dtype_in_u8s8() else last_ret
        if burlen != 0:
            self.tik_instance.data_move(self.output_out_gm[idx], real_out,
                                        Constant.VALUE_ZERO, DEFAULT_BURST_LEN, burlen,
                                        STRIDE_ZERO, STRIDE_ZERO)

        if self.need_special and position == self.spe_position:
            last_32b = self.post_multicore(burlen, idx, real_out, tail_idx)

            self.tik_instance.data_move(
                self.output_out_gm[idx[0], idx[1], idx[2] + tail_idx],
                last_32b, Constant.VALUE_ZERO, DEFAULT_BURST_LEN,
                Constant.VALUE_ONE, STRIDE_ZERO, STRIDE_ZERO)

        # to store last_ret and
        if self.ctype == Constant.LOGSUMEXP_TYPE:
            if self.need_special and self.spe_position == position:
                offset = Constant.VALUE_ONE
            else:
                offset = Constant.VALUE_ZERO
            self.tik_instance.data_move(last_ori, last_ret, Constant.VALUE_ZERO,
                                        DEFAULT_BURST_LEN, burlen+offset, STRIDE_ZERO,
                                        STRIDE_ZERO)

    def t_dma_trans_out(self, last_ret, last_ori, idx, position):
        """
        The dma command is compatible with the u8s8 type.

        Parameters
        ----------
        last_ret: the tensor store last result
        last_ori: the tensor store last result with original type
        idx: index
        position: the position of each , "head" or "tail"

        Returns
        -------
        None

        """
        before = self.tik_instance.Tensor(self.dtype, (BLOCK_SIZE,),
                                          name="before",
                                          scope=tik.scope_ubuf)
        if position == Constant.HEAD:
            burlen = self.get_burlen_by_mlen(self.mov_len)
            repeat = self.get_repeat(self.mov_len)
        else:
            burlen = self.get_burlen_by_mlen(self.mov_tail)
            repeat = self.get_repeat(self.mov_tail)

        # Check whether multi-core special processing is required.
        burlen, tail_idx = self.pre_multicore(burlen, position)

        real_out = last_ret
        if self.check_dtype_in_u8s8():
            self.tik_instance.vconv(self.mask, "", last_ori, last_ret,
                                    repeat, STRIDE_ONE,
                                    STRIDE_ONE, Constant.STRIDE_FOUR,
                                    REPEAT_STRIDE_EIGHT)
            real_out = last_ori

        if self.mov_loop == Constant.VALUE_ONE and self.reverse and \
                (self.mov_len * self.dsize) % BLOCK_SIZE != Constant.VALUE_ZERO:
            self.tik_instance.data_move(
                before, self.output_out_gm[idx[0], idx[1] + Constant.VALUE_ONE, idx[2]],
                Constant.VALUE_ZERO, DEFAULT_BURST_LEN,
                Constant.VALUE_ONE, STRIDE_ZERO, STRIDE_ZERO)
            tail_idx_fix = self.get_burlen_by_mlen(self.mov_len) * BLOCK_SIZE \
                           // self.dsize + self.each_tail - BLOCK_SIZE // \
                           self.dsize

            with self.tik_instance.for_range(
                    0, BLOCK_SIZE // self.dsize - self.each_tail) as t_idx:
                temp_scalar = self.tik_instance.Scalar(dtype=self.dtype)
                temp_scalar.set_as(before[t_idx])
                real_out[tail_idx_fix + t_idx].set_as(temp_scalar)

        if burlen != 0:
            self.tik_instance.data_move(self.output_out_gm[idx],
                                        real_out, Constant.VALUE_ZERO,
                                        DEFAULT_BURST_LEN,
                                        burlen, STRIDE_ZERO, STRIDE_ZERO)

        if self.need_special and position == self.spe_position:
            last_32b = self.post_multicore(burlen, idx, real_out, tail_idx)

            self.tik_instance.data_move(
                self.output_out_gm[idx[0], idx[1], idx[2] + tail_idx],
                last_32b, Constant.VALUE_ZERO, DEFAULT_BURST_LEN,
                Constant.VALUE_ONE, STRIDE_ZERO, STRIDE_ZERO)

    def calc_cell(self, mask_process, repeat_process, last_ret, input_x_ub, idx):
        """
        Mul calculation

        Returns
        -------
        None

        """
        if self.ctype == Constant.SUM_TYPE:
            self.tik_instance.vadd(mask_process, last_ret[idx], input_x_ub[idx], last_ret[idx],
                                   repeat_process, STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT)
        else:
            self.tik_instance.vmul(mask_process, last_ret[idx], input_x_ub[idx], last_ret[idx],
                                   repeat_process, STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT)

    def calc_tail_process(self, mov_length, repeat, last_ret, input_x_ub):
        """
        Check tail exit and special process

        Returns
        -------
        None

        """
        align_repeat = mov_length * self.rdsize // VECTOR_BYTE_SIZE
        if align_repeat != Constant.VALUE_ZERO:
            self.calc_cell(self.mask, align_repeat, last_ret, input_x_ub, Constant.VALUE_ZERO)
        mask_vector = mov_length % (VECTOR_BYTE_SIZE // self.rdsize)
        self.calc_cell(mask_vector, DEFAULT_REPEAT_TIME, last_ret, input_x_ub, self.mask * align_repeat)

    def calc_process(self, mov_length, repeat, e_cycle, last_ret, input_x_ub):
        """
        Different process by checking revserse and e_cycle

        Returns
        -------
        None

        """
        with self.tik_instance.if_scope(mov_length % (VECTOR_BYTE_SIZE // self.rdsize) != Constant.VALUE_ZERO):
            self.calc_tail_process(mov_length, repeat, last_ret, input_x_ub)
        with self.tik_instance.else_scope():
            self.calc_cell(self.mask, repeat, last_ret, input_x_ub, Constant.VALUE_ZERO)

    def cum_computer(self):
        """
        Calculation process of the operator

        Returns
        -------
        None

        """
        if self.ctype in (Constant.SUM_TYPE, Constant.PROD_TYPE):
            with self.tik_instance.for_range(0, self.block_num,
                                             block_num=self.block_num) as block_i:
                self.handle_out_loop(block_i, self.outer_loop)

            # handle kernel tail
            if self.outer_tail != Constant.VALUE_ZERO:
                self.handle_out_loop(self.block_num, self.outer_tail)
        else:
            if self.total_loop > 65535 or self.is_last_axis or self.each*self.dsize < BLOCK_SIZE:
                with self.tik_instance.for_range(0, self.block_num,
                                                 block_num=self.block_num) as block_i:
                    self.handle_out_loop(block_i, self.outer_loop)

                # handle kernel tail
                if self.outer_tail != Constant.VALUE_ZERO:
                    self.handle_out_loop(self.block_num, self.outer_tail)
            else:
                with self.tik_instance.for_range(0, self.total_loop,
                                                 block_num=self.total_loop) as block_i:
                    self.handle_every_loop(block_i)

    def handle_every_loop(self, block_i):
        """
        handle_every_loop
        """
        o_idx = block_i

        if self.mov_tail != Constant.VALUE_ZERO:
            self.handle_mov_tail(o_idx)

        self.handle_mov_loop(o_idx)

    def handle_out_loop(self, block_i, loop_num):
        """
        Multi-core processing data of the entire block

        Parameters
        ----------
        block_i: block index
        loop_num: loop number

        Returns
        -------
        None

        """
        with self.tik_instance.for_range(Constant.VALUE_ZERO, loop_num) as o_cycle:
            o_idx = o_cycle + block_i * self.outer_loop

            # handle mov tail first because overlap
            if self.mov_tail != Constant.VALUE_ZERO:
                self.handle_mov_tail(o_idx)

            self.handle_mov_loop(o_idx)

    def handle_mov_loop(self, o_cycle):
        """
        Calculation Process

        Parameters
        ----------
        o_cycle: outer cycle

        Returns
        -------
        None

        """

        thread_num = Constant.VALUE_TWO if self.mov_loop > Constant.VALUE_ONE else Constant.VALUE_ONE
        with self.tik_instance.for_range(Constant.VALUE_ZERO, self.mov_loop,
                                         thread_num=thread_num) as m_cycle:
            # ub tensor
            input_x_ub = self.tik_instance. \
                Tensor(self.rdtype,
                       ((self.ub_tensor_size + VECTOR_BYTE_SIZE) // self.rdsize,),
                       name="input_x_ub",
                       scope=tik.scope_ubuf)
            last_ret = self.tik_instance. \
                Tensor(self.rdtype,
                       ((self.ub_tensor_size + VECTOR_BYTE_SIZE) // self.rdsize,),
                       name="last_ret",
                       scope=tik.scope_ubuf)

            last_ori = self.tik_instance. \
                Tensor(self.dtype,
                       ((self.ub_tensor_size + VECTOR_BYTE_SIZE) // self.rdsize,),
                       name="last_ori",
                       scope=tik.scope_ubuf)
            if self.ctype == Constant.LOGSUMEXP_TYPE:
                max_v = self.tik_instance. \
                    Tensor(self.dtype,
                           ((self.ub_tensor_size + VECTOR_BYTE_SIZE) // self.rdsize,),
                           name="max_v",
                           scope=tik.scope_ubuf)
                min_v = self.tik_instance. \
                    Tensor(self.dtype,
                           ((self.ub_tensor_size + VECTOR_BYTE_SIZE) // self.rdsize,),
                           name="min_v",
                           scope=tik.scope_ubuf)

            burstlen = self.get_burlen_by_mlen(self.mov_len)
            repeat = self.get_repeat(self.mov_len)
            if self.exclusive and not self.reverse:
                idx = [o_cycle, Constant.VALUE_ZERO, m_cycle * self.mov_len]
                self.t_vdup_to_gm(last_ret, last_ori, idx, Constant.HEAD)
                if self.ctype == Constant.LOGSUMEXP_TYPE and self.each_loop > Constant.VALUE_ONE:
                    idx_out = [o_cycle, Constant.VALUE_ONE, m_cycle * self.mov_len]
                    self.t_dma_in(last_ret, last_ori, idx, burstlen)
                    self.t_dma_direct_out(last_ret, last_ori, idx_out, Constant.HEAD)

            elif not self.exclusive and not self.reverse:
                idx = [o_cycle, Constant.VALUE_ZERO, m_cycle * self.mov_len]
                self.t_dma_in(last_ret, last_ori, idx, burstlen)
                self.t_dma_direct_out(last_ret, last_ori, idx, Constant.HEAD)

            elif not self.exclusive and self.reverse:
                idx = [o_cycle, self.each_loop - Constant.VALUE_ONE,
                       m_cycle * self.mov_len]
                self.t_dma_in(last_ret, last_ori, idx, burstlen)
                self.t_dma_direct_out(last_ret, last_ori, idx, Constant.HEAD)

            elif self.exclusive and self.reverse:
                if self.ctype == Constant.LOGSUMEXP_TYPE and self.each_loop > Constant.VALUE_ONE:
                    idx_in = [o_cycle, self.each_loop - Constant.VALUE_ONE,
                              m_cycle * self.mov_len]
                    idx_out = [o_cycle, self.each_loop - Constant.VALUE_TWO,
                               m_cycle * self.mov_len]
                    self.t_dma_in(last_ret, last_ori, idx_in, burstlen)
                    self.t_dma_direct_out(last_ret, last_ori, idx_out, Constant.HEAD)
                idx = [o_cycle, self.each_loop - Constant.VALUE_ONE,
                       m_cycle * self.mov_len]
                self.t_vdup_to_gm(last_ret, last_ori, idx, Constant.HEAD)

            if self.ctype == Constant.LOGSUMEXP_TYPE:
                if self.exclusive and self.each_loop > Constant.VALUE_ONE:
                    each_loop_start = Constant.VALUE_TWO
                else:
                    each_loop_start = Constant.VALUE_ONE
            else:
                each_loop_start = Constant.VALUE_ONE
            with self.tik_instance.for_range(each_loop_start,
                                             self.each_loop) as e_cycle:
                in_offset, out_offset = self.get_offset(e_cycle)

                idx = [o_cycle, e_cycle + in_offset,
                       m_cycle * self.mov_len]
                self.t_dma_in(input_x_ub, last_ori, idx, burstlen)

                if self.ctype in (Constant.PROD_TYPE, Constant.SUM_TYPE):
                    self.calc_process(self.mov_len, repeat, e_cycle, last_ret, input_x_ub)
                    idx = [o_cycle, e_cycle + out_offset,
                           m_cycle * self.mov_len]
                    self.t_dma_trans_out(last_ret, last_ori, idx, Constant.HEAD)

                elif self.ctype == Constant.LOGSUMEXP_TYPE:
                    # compare input_x_ub with last_ret, find min and max
                    self.tik_instance.vec_max(self.mask, max_v, last_ret,
                                              input_x_ub, repeat,
                                              REPEAT_STRIDE_EIGHT,
                                              REPEAT_STRIDE_EIGHT,
                                              REPEAT_STRIDE_EIGHT)
                    self.tik_instance.vec_min(self.mask, min_v, last_ret,
                                              input_x_ub, repeat,
                                              REPEAT_STRIDE_EIGHT,
                                              REPEAT_STRIDE_EIGHT,
                                              REPEAT_STRIDE_EIGHT)

                    # do 1 + exp(min - max)
                    self.tik_instance.vec_sub(self.mask, last_ret,
                                              min_v, max_v, repeat,
                                              REPEAT_STRIDE_EIGHT,
                                              REPEAT_STRIDE_EIGHT,
                                              REPEAT_STRIDE_EIGHT)
                    self.tik_instance.vec_exp(self.mask, last_ret, last_ret,
                                              repeat,
                                              REPEAT_STRIDE_EIGHT,
                                              REPEAT_STRIDE_EIGHT)
                    self.tik_instance.vec_adds(self.mask, last_ret, last_ret,
                                               self.scalar_one, repeat,
                                               REPEAT_STRIDE_EIGHT,
                                               REPEAT_STRIDE_EIGHT)

                    # do ln
                    self.tik_instance.vec_ln(self.mask, last_ret, last_ret,
                                             repeat,
                                             REPEAT_STRIDE_EIGHT,
                                             REPEAT_STRIDE_EIGHT)

                    # do + max
                    self.tik_instance.vec_add(self.mask, last_ret, last_ret,
                                              max_v, repeat,
                                              REPEAT_STRIDE_EIGHT,
                                              REPEAT_STRIDE_EIGHT,
                                              REPEAT_STRIDE_EIGHT)

                    idx = [o_cycle, e_cycle + out_offset,
                           m_cycle * self.mov_len]
                    self.t_dma_trans_out(last_ret, last_ori, idx, Constant.HEAD)

    def handle_mov_tail(self, o_cycle):
        """
        Calculation Process

        Parameters
        ----------
        o_cycle: outer cycle

        Returns
        -------
        None

        """
        # ub tensor
        input_x_ub = self.tik_instance. \
            Tensor(self.rdtype,
                   ((self.ub_tensor_size + VECTOR_BYTE_SIZE) // self.rdsize,),
                   name="input_x_ub",
                   scope=tik.scope_ubuf)
        last_ret = self.tik_instance. \
            Tensor(self.rdtype,
                   ((self.ub_tensor_size + VECTOR_BYTE_SIZE) // self.rdsize,),
                   name="last_ret",
                   scope=tik.scope_ubuf)
        last_ori = self.tik_instance. \
            Tensor(self.dtype,
                   ((self.ub_tensor_size + VECTOR_BYTE_SIZE) // self.rdsize,),
                   name="last_ori",
                   scope=tik.scope_ubuf)
        if self.ctype == Constant.LOGSUMEXP_TYPE:
            max_v = self.tik_instance. \
                Tensor(self.dtype,
                       ((self.ub_tensor_size + VECTOR_BYTE_SIZE) // self.rdsize,),
                       name="max_v",
                       scope=tik.scope_ubuf)
            min_v = self.tik_instance. \
                Tensor(self.dtype,
                       ((self.ub_tensor_size + VECTOR_BYTE_SIZE) // self.rdsize,),
                       name="min_v",
                       scope=tik.scope_ubuf)

        burstlen = self.get_burlen_by_mlen(self.mov_tail)
        repeat = self.get_repeat(self.mov_tail)
        if self.exclusive and not self.reverse:
            idx = [o_cycle, Constant.VALUE_ZERO, self.mov_loop * self.mov_len]
            self.t_vdup_to_gm(last_ret, last_ori, idx, Constant.TAIL)
            if self.ctype == Constant.LOGSUMEXP_TYPE and self.each_loop > Constant.VALUE_ONE:
                idx_out = [o_cycle, Constant.VALUE_ONE, self.mov_loop * self.mov_len]
                self.t_dma_in(last_ret, last_ori, idx, burstlen)
                self.t_dma_direct_out(last_ret, last_ori, idx_out, Constant.TAIL)

        elif not self.exclusive and not self.reverse:
            idx = [o_cycle, Constant.VALUE_ZERO, self.mov_loop * self.mov_len]
            self.t_dma_in(last_ret, last_ori, idx, burstlen)
            self.t_dma_direct_out(last_ret, last_ori, idx, Constant.TAIL)

        elif not self.exclusive and self.reverse:
            idx = [o_cycle, self.each_loop - Constant.VALUE_ONE,
                   self.mov_loop * self.mov_len]
            self.t_dma_in(last_ret, last_ori, idx, burstlen)
            self.t_dma_direct_out(last_ret, last_ori, idx, Constant.TAIL)

        elif self.exclusive and self.reverse:
            if self.ctype == Constant.LOGSUMEXP_TYPE and self.each_loop > Constant.VALUE_ONE:
                idx_in = [o_cycle, self.each_loop - Constant.VALUE_ONE,
                          self.mov_loop * self.mov_len]
                idx_out = [o_cycle, self.each_loop - Constant.VALUE_TWO,
                           self.mov_loop * self.mov_len]
                self.t_dma_in(last_ret, last_ori, idx_in, burstlen)
                self.t_dma_direct_out(last_ret, last_ori, idx_out, Constant.TAIL)
            idx = [o_cycle, self.each_loop - Constant.VALUE_ONE,
                   self.mov_loop * self.mov_len]
            self.t_vdup_to_gm(last_ret, last_ori, idx, Constant.TAIL)

        if self.ctype == Constant.LOGSUMEXP_TYPE:
            if self.each_loop > Constant.VALUE_ONE and self.exclusive:
                each_loop_start = Constant.VALUE_TWO
            else:
                each_loop_start = Constant.VALUE_ONE
        else:
            each_loop_start = Constant.VALUE_ONE
        with self.tik_instance.for_range(each_loop_start,
                                         self.each_loop) as e_cycle:
            in_offset, out_offset = self.get_offset(e_cycle)

            idx = [o_cycle, e_cycle + in_offset, self.mov_loop * self.mov_len]
            self.t_dma_in(input_x_ub, last_ori, idx, burstlen)

            if self.ctype in (Constant.PROD_TYPE, Constant.SUM_TYPE):
                self.calc_process(self.mov_tail, repeat, e_cycle, last_ret, input_x_ub)
                idx = [o_cycle, e_cycle + out_offset,
                       self.mov_loop * self.mov_len]
                self.t_dma_trans_out(last_ret, last_ori, idx, Constant.TAIL)

            elif self.ctype == Constant.LOGSUMEXP_TYPE:
                # compare input_x_ub with last_ret,find min and max
                self.tik_instance.vec_max(self.mask, max_v, last_ret,
                                          input_x_ub, repeat,
                                          REPEAT_STRIDE_EIGHT,
                                          REPEAT_STRIDE_EIGHT,
                                          REPEAT_STRIDE_EIGHT)
                self.tik_instance.vec_min(self.mask, min_v, last_ret,
                                          input_x_ub, repeat,
                                          REPEAT_STRIDE_EIGHT,
                                          REPEAT_STRIDE_EIGHT,
                                          REPEAT_STRIDE_EIGHT)

                # do 1 + exp(min - max)
                self.tik_instance.vec_sub(self.mask, last_ret,
                                          min_v, max_v, repeat,
                                          REPEAT_STRIDE_EIGHT,
                                          REPEAT_STRIDE_EIGHT,
                                          REPEAT_STRIDE_EIGHT)
                self.tik_instance.vec_exp(self.mask, last_ret, last_ret,
                                          repeat,
                                          REPEAT_STRIDE_EIGHT,
                                          REPEAT_STRIDE_EIGHT)
                self.tik_instance.vec_adds(self.mask, last_ret, last_ret,
                                           self.scalar_one, repeat,
                                           REPEAT_STRIDE_EIGHT,
                                           REPEAT_STRIDE_EIGHT)

                # do ln
                self.tik_instance.vec_ln(self.mask, last_ret, last_ret,
                                         repeat,
                                         REPEAT_STRIDE_EIGHT,
                                         REPEAT_STRIDE_EIGHT)

                # do + max
                self.tik_instance.vec_add(self.mask, last_ret, last_ret,
                                          max_v, repeat,
                                          REPEAT_STRIDE_EIGHT,
                                          REPEAT_STRIDE_EIGHT,
                                          REPEAT_STRIDE_EIGHT)

                idx = [o_cycle, e_cycle + out_offset,
                       self.mov_loop * self.mov_len]
                self.t_dma_trans_out(last_ret, last_ori, idx, Constant.TAIL)


def get_computer_by_ctype(input_x, axis, kernel_name, ctype):
    """
    Obtain the computer template.

    Parameters
    ----------
    input_x: dict, shape and dtype
    axis: the cumulative axis
    kernel_name: kernel name
    ctype: computer type

    Returns
    -------
    the instance of computer template

    """
    return CumComputer(input_x, axis, kernel_name, ctype)
