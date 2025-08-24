# Copyright 2020 Huawei Technologies Co., Ltd
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
unsorted_segment
"""
# pylint: disable=too-many-lines
from ..util.platform_adapter import tik
from ..util.platform_adapter import tbe_platform
from ..util.platform_adapter import tbe_context
from ..util.platform_adapter import error_manager_vector
from tbe.common.platform import get_bit_len

ONE_BLOCK_E = 1
ONE_DIV_E = 2
SMALL_E = 3
BIG_E = 4

# SMALL_IID
SELECT_KEY_DIV_OID_BLOCK_E_SMALL_IID_SMALL_OID = \
    [{"select_key": 10, "ub_div_id": 0, "block_align": False, "e_level": ONE_BLOCK_E, "big_iid": False,
      "div_oid": True}]
SELECT_KEY_DIV_OID_ONE_E_SMALL_IID_SMALL_OID = \
    [{"select_key": 20, "ub_div_id": 0, "block_align": False, "e_level": ONE_DIV_E, "big_iid": False, "div_oid": True}]

SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID_ALIGN = \
    [{"select_key": 30, "ub_div_id": 0, "block_align": True, "e_level": SMALL_E, "big_iid": False, "div_oid": True},
     {"select_key": 31, "ub_div_id": 1, "block_align": True, "e_level": SMALL_E, "big_iid": False, "div_oid": True},
     {"select_key": 32, "ub_div_id": 2, "block_align": True, "e_level": SMALL_E, "big_iid": False, "div_oid": True}]

SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID = \
    [{"select_key": 40, "ub_div_id": 0, "block_align": False, "e_level": SMALL_E, "big_iid": False, "div_oid": True},
     {"select_key": 41, "ub_div_id": 1, "block_align": False, "e_level": SMALL_E, "big_iid": False, "div_oid": True},
     {"select_key": 42, "ub_div_id": 2, "block_align": False, "e_level": SMALL_E, "big_iid": False, "div_oid": True}]

SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID_ALIGN = \
    [{"select_key": 50, "ub_div_id": 0, "block_align": True, "e_level": SMALL_E, "big_iid": False, "div_oid": False},
     {"select_key": 51, "ub_div_id": 1, "block_align": True, "e_level": SMALL_E, "big_iid": False, "div_oid": False},
     {"select_key": 52, "ub_div_id": 2, "block_align": True, "e_level": SMALL_E, "big_iid": False, "div_oid": False}]

SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID = \
    [{"select_key": 60, "ub_div_id": 0, "block_align": False, "e_level": SMALL_E, "big_iid": False, "div_oid": False},
     {"select_key": 61, "ub_div_id": 1, "block_align": False, "e_level": SMALL_E, "big_iid": False, "div_oid": False},
     {"select_key": 62, "ub_div_id": 2, "block_align": False, "e_level": SMALL_E, "big_iid": False, "div_oid": False}]

SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID_ALIGN = \
    [{"select_key": 70, "ub_div_id": 0, "block_align": True, "e_level": BIG_E, "big_iid": False, "div_oid": False},
     {"select_key": 71, "ub_div_id": 1, "block_align": True, "e_level": BIG_E, "big_iid": False, "div_oid": False},
     {"select_key": 72, "ub_div_id": 2, "block_align": True, "e_level": BIG_E, "big_iid": False, "div_oid": False}]

SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID = \
    [{"select_key": 80, "ub_div_id": 0, "block_align": False, "e_level": BIG_E, "big_iid": False, "div_oid": False},
     {"select_key": 81, "ub_div_id": 1, "block_align": False, "e_level": BIG_E, "big_iid": False, "div_oid": False},
     {"select_key": 82, "ub_div_id": 2, "block_align": False, "e_level": BIG_E, "big_iid": False, "div_oid": False}]

# BIG_IID
SELECT_KEY_DIV_OID_BLOCK_E_BIG_IID_SMALL_OID = \
    [{"select_key": 90, "ub_div_id": 0, "block_align": False, "e_level": ONE_BLOCK_E, "big_iid": True, "div_oid": True}]
SELECT_KEY_DIV_OID_ONE_E_BIG_IID_SMALL_OID = \
    [{"select_key": 100, "ub_div_id": 0, "block_align": False, "e_level": ONE_DIV_E, "big_iid": True, "div_oid": True}]

SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID_ALIGN = \
    [{"select_key": 110, "ub_div_id": 0, "block_align": True, "e_level": SMALL_E, "big_iid": True, "div_oid": True},
     {"select_key": 111, "ub_div_id": 1, "block_align": True, "e_level": SMALL_E, "big_iid": True, "div_oid": True},
     {"select_key": 112, "ub_div_id": 2, "block_align": True, "e_level": SMALL_E, "big_iid": True, "div_oid": True}]

SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID = \
    [{"select_key": 120, "ub_div_id": 0, "block_align": False, "e_level": SMALL_E, "big_iid": True, "div_oid": True},
     {"select_key": 121, "ub_div_id": 1, "block_align": False, "e_level": SMALL_E, "big_iid": True, "div_oid": True},
     {"select_key": 122, "ub_div_id": 2, "block_align": False, "e_level": SMALL_E, "big_iid": True, "div_oid": True}]

SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID_ALIGN = \
    [{"select_key": 130, "ub_div_id": 0, "block_align": True, "e_level": SMALL_E, "big_iid": True, "div_oid": False},
     {"select_key": 131, "ub_div_id": 1, "block_align": True, "e_level": SMALL_E, "big_iid": True, "div_oid": False},
     {"select_key": 132, "ub_div_id": 2, "block_align": True, "e_level": SMALL_E, "big_iid": True, "div_oid": False}]

SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID = \
    [{"select_key": 140, "ub_div_id": 0, "block_align": False, "e_level": SMALL_E, "big_iid": True, "div_oid": False},
     {"select_key": 141, "ub_div_id": 1, "block_align": False, "e_level": SMALL_E, "big_iid": True, "div_oid": False},
     {"select_key": 142, "ub_div_id": 2, "block_align": False, "e_level": SMALL_E, "big_iid": True, "div_oid": False}]

SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID_ALIGN = \
    [{"select_key": 150, "ub_div_id": 0, "block_align": True, "e_level": BIG_E, "big_iid": True, "div_oid": False},
     {"select_key": 151, "ub_div_id": 1, "block_align": True, "e_level": BIG_E, "big_iid": True, "div_oid": False},
     {"select_key": 152, "ub_div_id": 2, "block_align": True, "e_level": BIG_E, "big_iid": True, "div_oid": False}]

SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID = \
    [{"select_key": 160, "ub_div_id": 0, "block_align": False, "e_level": BIG_E, "big_iid": True, "div_oid": False},
     {"select_key": 161, "ub_div_id": 1, "block_align": False, "e_level": BIG_E, "big_iid": True, "div_oid": False},
     {"select_key": 162, "ub_div_id": 2, "block_align": False, "e_level": BIG_E, "big_iid": True, "div_oid": False}]

TILING_PARAM_DTYPE = "int32"

# max_int32
MAX_INT32 = 2 ** 31 - 1

# int32 byte
BYTE_INT32 = 4

# byte of one block
BYTE_BLOCK = 32

# min_tensor_ele_num
MIN_TENSOR_ELE_NUM = 32

# tiling params num
TILING_PARAMS_NUM = 64

BYTE_FULL_MASK = 256


# pylint: disable=invalid-name,too-many-instance-attributes,too-many-arguments,too-many-statements
# pylint: disable=too-many-locals,too-few-public-methods,unused-argument

def _get_dtype_byte(dtype):
    return get_bit_len(dtype) // 8


def _get_dtype_min_val(dtype):
    """
    get dtype min val
    """
    val_list = {'float16': -65536,
                'float32': -(2 - 2 ** -23) * 2 ** 127 + 1,
                'int32': -(2 ** 31),
                'uint32': 0}
    return val_list[dtype]


def _get_dtype_max_val(dtype):
    """
    get dtype max val
    """
    val_list = {'float16': 65536,
                'float32': (2 - 2 ** -23) * 2 ** 127,
                'int32': (2 ** 31 - 1),
                'uint32': (2 ** 32)}
    return val_list[dtype]


def _ceil_div(val, block):
    """
    compute ceil div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    ceil value
    """
    return (val + block - 1) // block


def _floor(val, block):
    """
    compute floor div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    floor value
    """
    return val // block * block


def _div(val, block):
    """
    compute front part and last part in ceil div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    front_part_num: front part in ceil div
    last_part: last part in ceil div
    """
    front_part_num = val // block
    last_part = val - front_part_num * block
    return front_part_num, last_part


def _prod(val_list):
    """
    compute prod of val_list

    Parameters
    ----------
    val_list: val_list

    Returns
    -------
    val: prod of val_list
    """
    val = 1
    for ele in val_list:
        val = val * ele
    return val


class FrontLast(object):
    """
    Function: loop partition. include front, last, times.
    """

    def __init__(self, tik_instance=0, name=0, obj=None):
        if tik_instance == 0:
            self.front = 0
            self.last = 0
            self.times = 0
        elif obj is None:
            self.times = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE, name=name + "_times")
            self.front = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE, name=name + "_front")
            self.last = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE, name=name + "_last")
        else:
            self.front = obj.front
            self.last = obj.last
            self.times = obj.times

    def div_by_num(self, total, times, min_part=1):
        """
        Function: cal partition by num.
        """
        self.front = _ceil_div(total, times)
        self.times = _ceil_div(total, self.front)
        self.last = total - (self.times - 1) * self.front

        while (self.front < min_part or self.last < min_part) and self.front < total:
            self.front = self.front + 1
            self.times = _ceil_div(total, self.front)
            self.last = total - (self.times - 1) * self.front

    def div_by_part(self, total, part, front_block=1, balance_num=1):
        """
        Function: cal partition by front part.
        """
        times = _ceil_div(total, part)
        self.front = _ceil_div(total, times)
        self.front = _ceil_div(self.front, front_block) * front_block
        if self.front > part:
            self.front = max(self.front // front_block * front_block, front_block)
        self.times = _ceil_div(total, self.front)

        if balance_num > 1:
            times = _ceil_div(self.times, balance_num) * balance_num
            self.front = _ceil_div(total, times)
            self.front = _ceil_div(self.front, front_block) * front_block
            if self.front > part:
                self.front = max(self.front // front_block * front_block, front_block)
            self.times = _ceil_div(total, self.front)
        self.last = total - (self.times - 1) * self.front


class CommonScalar(object):
    """
    Function: use to store concat base parameters
    """

    def __init__(self, tik_instance, num_segments_dtype, ids_dtype, obj_scalar):
        """
        constructor of class CommonScalar

        Parameters
        ----------
        tik_instance: tik_instance
        num_segments_dtype: num_segments dtype
        ids_dtype: ids dtype

        Returns
        -------
        None
        """

        self.num_segments_param = FrontLast(tik_instance, "num_segments",
                                            None if obj_scalar is None else obj_scalar.num_segments_param)

        self.num_segments_loop_param = FrontLast(tik_instance, "num_segments_loop",
                                                 None if obj_scalar is None else
                                                 obj_scalar.num_segments_loop_param)

        self.e_out_param = FrontLast(tik_instance, "e_out_param",
                                     None if obj_scalar is None else obj_scalar.e_out_param)
        self.ids_param = FrontLast(tik_instance, "ids_param",
                                   None if obj_scalar is None else obj_scalar.ids_param)
        self.e_out_loop_param = FrontLast(tik_instance, "e_out_loop_param",
                                          None if obj_scalar is None else
                                          obj_scalar.e_out_loop_param)

        if obj_scalar is None:
            self.ids_num = "ids_num"
            self.id_val_scalar = tik_instance.Scalar(dtype=ids_dtype, name="id_val_scalar")
            self.num_segments = tik_instance.Scalar(dtype=num_segments_dtype, name="num_segments")
            self.select_key = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE, name="select_key")
            self.ids_last_burst_len = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE, name="ids_last_burst_len")
            self.e_num = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE, name="e_num")
            self.repeat_time_front_part = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE,
                                                              name="repeat_time_front_part")
            self.repeat_time_last_part = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE,
                                                             name="repeat_time_last_part")
            self.align_scalar = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE, name="align_scalar")
            self.e_lenBurst_front = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE, name="e_lenBurst_front")
            self.e_lenBurst_last = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE, name="e_lenBurst_last")
            self.e_num_part_ub_num = tik_instance.Scalar(dtype=TILING_PARAM_DTYPE, name="e_num_part_ub_num")
        else:
            self.ids_num = obj_scalar.ids_num
            self.id_val_scalar = obj_scalar.id_val_scalar
            self.num_segments = obj_scalar.num_segments
            self.select_key = obj_scalar.select_key
            self.ids_last_burst_len = obj_scalar.ids_last_burst_len
            self.e_num = obj_scalar.e_num
            self.repeat_time_front_part = obj_scalar.repeat_time_front_part
            self.repeat_time_last_part = obj_scalar.repeat_time_last_part
            self.align_scalar = obj_scalar.align_scalar
            self.e_lenBurst_front = obj_scalar.e_lenBurst_front
            self.e_lenBurst_last = obj_scalar.e_lenBurst_last
            self.e_num_part_ub_num = obj_scalar.e_num_part_ub_num

    def set_consts(self, segment_ids, e_num, num_segments):
        """
        Function: set basic val.
        """
        if num_segments[0] == num_segments[1] and num_segments[0] is not None:
            self.num_segments = num_segments[0]

        if e_num[0] == e_num[1] and e_num[0] is not None:
            self.e_num = e_num[0]

        if segment_ids[0] == segment_ids[1] and segment_ids[0] is not None:
            self.ids_num = segment_ids[0]


class UnsortedSegmentTiling(object):
    """
    Function: do UnsortedSegmentTiling
    """

    def __init__(self, op_info, x_dict, segment_ids_dict, y_dict):
        # ===================basic param===============================
        self.core_num = op_info.core_num
        input_byte = _get_dtype_byte(op_info.input_dtype)
        self.mask = BYTE_FULL_MASK // input_byte
        self.ids_per_block = BYTE_BLOCK // _get_dtype_byte(op_info.ids_dtype)

        self.ele_num_per_block = BYTE_BLOCK // input_byte
        self.e_max_by_stride = 65535 * self.ele_num_per_block

        max_param_val = 2 ** 32
        min_param_val = 0

        def _get_prod_range(shape_range):
            range_min = [ele[0] for ele in shape_range]
            range_max = [ele[1] for ele in shape_range]

            range_min = min_param_val if None in range_min else _prod(range_min)
            range_max = max_param_val if None in range_max else _prod(range_max)
            return range_min, range_max

        if "range" in x_dict:
            x_range = x_dict.get("range")
        else:
            x_range = [[dim, dim] for dim in x_dict.get("shape")]

        if "range" in segment_ids_dict:
            segment_ids_range = segment_ids_dict.get("range")
        else:
            segment_ids_range = [[dim, dim] for dim in segment_ids_dict.get("shape")]

        if "range" in y_dict:
            num_segments_range = y_dict.get("range")[0]
        else:
            num_segments_range = [y_dict.get("shape")[0], y_dict.get("shape")[0]]

        self.segment_ids_min, self.segment_ids_max = _get_prod_range(segment_ids_range)
        self.e_num_min, self.e_num_max = _get_prod_range(x_range[len(segment_ids_range):])

        self.num_segments_min, self.num_segments_max = num_segments_range
        if self.num_segments_min is None:
            self.num_segments_min = min_param_val
        if self.num_segments_max is None:
            self.num_segments_max = max_param_val

        self.is_double_buffer = False
        self.ub_size = _tik_get_ub_size(self.is_double_buffer)
        self.ids_once_num = ((self.ub_size // 5 // BYTE_BLOCK) * BYTE_BLOCK) // BYTE_INT32
        res_ub_size = self.ub_size - self.ids_once_num * BYTE_INT32

        # ===================scalar param==============================
        ub_div_rates = [[1, 1], [1, 2], [2, 1]]
        obj_scalar = CommonScalar(op_info.tik_instance, op_info.num_segments_dtype, op_info.ids_dtype, None)
        obj_scalar.set_consts([self.segment_ids_min, self.segment_ids_max],
                              [self.e_num_min, self.e_num_max],
                              [self.num_segments_min, self.num_segments_max])

        self.scalars = []
        for a, b in ub_div_rates:
            scalar = CommonScalar(op_info.tik_instance, op_info.num_segments_dtype, op_info.ids_dtype, obj_scalar)
            scalar.input_once_num = self._floor_align_mask(res_ub_size // input_byte // (a + b) * a)
            scalar.output_once_num = self._floor_align_mask(res_ub_size // input_byte // (a + b) * b)
            self.scalars.append(self._get_tiling_params(scalar))

        self.select_keys_for_compile = self._get_select_keys_for_compile(input_byte, obj_scalar)
        self.obj_scalar = self.scalars[0]
        select_mode = self._get_tiling_mode(self.obj_scalar)

        if select_mode != False:
            ub_use_rate = 0
            for the_mode in select_mode:
                scalar = self.scalars[the_mode["ub_div_id"]]
                if scalar.ub_use_rate > ub_use_rate and the_mode in self.select_keys_for_compile:
                    ub_use_rate = scalar.ub_use_rate
                    scalar.select_key = the_mode
                    self.obj_scalar = scalar
                    if scalar.ids_param.times == 1:
                        break

    def _floor_align_mask(self, val):
        return max(val // self.mask * self.mask, self.mask)

    def _ceil_align_block(self, val):
        return _ceil_div(val, self.ele_num_per_block) * self.ele_num_per_block

    def _get_tiling_params(self, scalar):
        if isinstance(scalar.e_num, int):
            if scalar.e_num % self.ele_num_per_block == 0:
                align_scalar = 0
            else:
                align_scalar = self.ele_num_per_block - (
                        scalar.e_num - (scalar.e_num // self.ele_num_per_block) * self.ele_num_per_block)
            if isinstance(align_scalar, int):
                scalar.align_scalar = align_scalar

        if isinstance(scalar.e_num, int) and isinstance(scalar.num_segments, int):
            e_max_block_ub = min(self._floor_align_mask(scalar.output_once_num // scalar.num_segments), 255 * self.mask)
            scalar.e_out_param.div_by_part(scalar.e_num, e_max_block_ub, self.mask, self.core_num)

        if isinstance(scalar.e_out_param.times, int) and isinstance(scalar.num_segments, int):
            scalar.e_out_loop_param.div_by_num(scalar.e_out_param.times, self.core_num)
            scalar.num_segments_core_num = max(min(self.core_num // scalar.e_out_loop_param.times,
                                                   _ceil_div(scalar.num_segments * scalar.e_out_param.last, self.mask)),
                                               1)

        if isinstance(scalar.ids_num, int) and isinstance(scalar.e_out_param.front, int):
            scalar.ids_param.div_by_part(scalar.ids_num, min(scalar.input_once_num // scalar.e_out_param.front,
                                                             self.ids_once_num))

        if isinstance(scalar.e_out_param.front, int):
            scalar.e_num_part_ub_num = _ceil_div(scalar.e_out_param.front, self.mask) * self.mask
            if isinstance(scalar.num_segments, int) and isinstance(scalar.num_segments_core_num, int):
                scalar.num_segments_param.div_by_part(scalar.num_segments,
                                                      scalar.output_once_num // scalar.e_num_part_ub_num,
                                                      balance_num=scalar.num_segments_core_num)

        rate_out = scalar.e_out_param.front * scalar.num_segments_param.front / scalar.output_once_num
        rate_in = scalar.e_out_param.front * scalar.ids_param.front / scalar.input_once_num
        scalar.ub_use_rate = rate_in * rate_out

        if isinstance(scalar.num_segments_param.times, int) and isinstance(scalar.num_segments_core_num, int):
            scalar.num_segments_loop_param.div_by_num(scalar.num_segments_param.times, scalar.num_segments_core_num)

        if isinstance(scalar.e_out_param.front, int):
            scalar.e_lenBurst_front = _ceil_div(scalar.e_out_param.front, self.ele_num_per_block)
            scalar.e_lenBurst_last = _ceil_div(scalar.e_out_param.last, self.ele_num_per_block)

            scalar.repeat_time_front_part = _ceil_div(scalar.e_out_param.front, self.mask)
            scalar.repeat_time_last_part = _ceil_div(scalar.e_out_param.last, self.mask)

        if isinstance(scalar.ids_num, int):
            scalar.ids_last_burst_len = _ceil_div(scalar.ids_num, self.ids_per_block)

        return scalar

    def _get_select_keys_for_compile(self, input_byte, obj_scalar):
        select_keys = SELECT_KEY_DIV_OID_BLOCK_E_SMALL_IID_SMALL_OID + \
                      SELECT_KEY_DIV_OID_ONE_E_SMALL_IID_SMALL_OID + \
                      SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID_ALIGN + \
                      SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID + \
                      SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID_ALIGN + \
                      SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID + \
                      SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID_ALIGN + \
                      SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID + \
                      SELECT_KEY_DIV_OID_BLOCK_E_BIG_IID_SMALL_OID + \
                      SELECT_KEY_DIV_OID_ONE_E_BIG_IID_SMALL_OID + \
                      SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID_ALIGN + \
                      SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID + \
                      SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID_ALIGN + \
                      SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID + \
                      SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID_ALIGN + \
                      SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID

        select_keys_mark = [1 for _ in select_keys]

        for i, key in enumerate(select_keys):
            if self.segment_ids_min > self.ids_once_num and not key["big_iid"]:
                select_keys_mark[i] = 0
            if self.segment_ids_max <= self.ids_once_num and key["big_iid"]:
                select_keys_mark[i] = 0

            if isinstance(obj_scalar.e_num, int):
                e_vector_num = _ceil_div(obj_scalar.e_num, self.mask)
                if e_vector_num == 1 and obj_scalar.e_num % self.ele_num_per_block != 0:
                    if obj_scalar.e_num >= self.ele_num_per_block:
                        e_level = ONE_DIV_E
                    else:
                        e_level = ONE_BLOCK_E
                else:
                    if obj_scalar.e_num <= self.e_max_by_stride:
                        e_level = SMALL_E
                    else:
                        e_level = BIG_E

                if e_level != key["e_level"]:
                    select_keys_mark[i] = 0

                if (obj_scalar.e_num % self.ele_num_per_block == 0) != key["block_align"]:
                    select_keys_mark[i] = 0

            else:
                if self.e_num_min > self.e_max_by_stride and key["e_level"] != BIG_E:
                    select_keys_mark[i] = 0
                if self.e_num_max <= self.e_max_by_stride and key["e_level"] == BIG_E:
                    select_keys_mark[i] = 0

                if self.e_num_min >= self.mask and key["e_level"] == ONE_DIV_E:
                    select_keys_mark[i] = 0

                if self.e_num_min >= self.ele_num_per_block and key["e_level"] == ONE_BLOCK_E:
                    select_keys_mark[i] = 0

                if self.e_num_max < self.mask and key["e_level"] == BIG_E:
                    select_keys_mark[i] = 0

                if self.e_num_max < self.ele_num_per_block and (
                        key["e_level"] == SMALL_E or key["e_level"] == ONE_DIV_E):
                    select_keys_mark[i] = 0

            if self.e_num_min > self.mask * self.core_num // 2 and key["div_oid"]:
                select_keys_mark[i] = 0

            in_min = self.e_num_min * self.segment_ids_min * input_byte
            in_max = self.e_num_max * self.segment_ids_max * input_byte

            out_min = self.e_num_min * self.num_segments_min * input_byte
            out_max = self.e_num_max * self.num_segments_max * input_byte

            if in_max <= self.scalars[0].input_once_num \
                    and out_max <= self.scalars[0].output_once_num and key["ub_div_id"] != 0:
                select_keys_mark[i] = 0

        select_keys_for_compile = []
        for i, keep in enumerate(select_keys_mark):
            if keep:
                select_keys_for_compile.append(select_keys[i])

        return select_keys_for_compile

    def _get_tiling_mode(self, scalar):

        if not (isinstance(scalar.e_num, int) and isinstance(scalar.num_segments, int) and
                isinstance(scalar.ids_num, int)):
            return False

        e_vector_num = _ceil_div(scalar.e_num, self.mask)

        if e_vector_num == 1 and scalar.e_num % self.ele_num_per_block != 0:
            if scalar.e_num >= self.ele_num_per_block:
                if scalar.ids_num <= self.ids_once_num:
                    select_mode = SELECT_KEY_DIV_OID_ONE_E_SMALL_IID_SMALL_OID
                else:
                    select_mode = SELECT_KEY_DIV_OID_ONE_E_BIG_IID_SMALL_OID
            else:
                if scalar.ids_num <= self.ids_once_num:
                    select_mode = SELECT_KEY_DIV_OID_BLOCK_E_SMALL_IID_SMALL_OID
                else:
                    select_mode = SELECT_KEY_DIV_OID_BLOCK_E_BIG_IID_SMALL_OID

        elif scalar.e_num % self.ele_num_per_block == 0:
            if scalar.ids_num <= self.ids_once_num:
                if scalar.e_num <= self.e_max_by_stride:
                    if scalar.num_segments_core_num > 1:
                        select_mode = SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID_ALIGN
                    else:
                        select_mode = SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID_ALIGN
                else:
                    select_mode = SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID_ALIGN
            else:
                if scalar.e_num <= self.e_max_by_stride:
                    if scalar.num_segments_core_num > 1:
                        select_mode = SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID_ALIGN
                    else:
                        select_mode = SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID_ALIGN
                else:
                    select_mode = SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID_ALIGN
        else:
            if scalar.ids_num <= self.ids_once_num:
                if scalar.e_num <= self.e_max_by_stride:
                    if scalar.num_segments_core_num > 1:
                        select_mode = SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID
                    else:
                        select_mode = SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID
                else:
                    select_mode = SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID
            else:
                if scalar.e_num <= self.e_max_by_stride:
                    if scalar.num_segments_core_num > 1:
                        select_mode = SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID
                    else:
                        select_mode = SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID
                else:
                    select_mode = SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID

        return select_mode


class TikTemplate(object):
    """
        Function: basic function template
    """

    def __init__(self, block_index, tik_inst, dtype, obj_gm_tensor, obj_ub_tensor, obj_tiling, instruction):
        self.block_index = block_index
        self.dtype = dtype
        self.ele_num_per_block = BYTE_BLOCK // _get_dtype_byte(self.dtype)
        self.tik_inst = tik_inst
        self.obj_gm_tensor = obj_gm_tensor
        self.obj_ub_tensor = obj_ub_tensor
        self.instruction = instruction
        self.obj_tiling = obj_tiling
        self.obj_scalar = None

    def _basic_module_get_input_and_cal(self, num_segments_part_index, num_segments_part,
                                        e_mov_index_gm2ub, ids_indexa, ids_block_num,
                                        e_lenBurst, block_align, e_level, last_e, big_iid):
        if last_e:
            repeat_time_part = self.obj_scalar.repeat_time_last_part
        else:
            repeat_time_part = self.obj_scalar.repeat_time_front_part

        if e_level != BIG_E and block_align:
            input_offset_gm = (ids_indexa * self.obj_scalar.ids_param.front) * \
                              self.obj_scalar.e_num + \
                              e_mov_index_gm2ub * self.obj_scalar.e_out_param.front
            src_stride = self.obj_scalar.e_num // self.ele_num_per_block - e_lenBurst
            dst_stride = self.obj_scalar.e_num_part_ub_num // self.ele_num_per_block - e_lenBurst
            _tik_mov_input_gm2ub_continue(self.tik_inst, self.obj_gm_tensor.input_gm, self.obj_ub_tensor.input_ub,
                                          input_offset_gm, 0, ids_block_num, e_lenBurst, src_stride, dst_stride)
        else:
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, ids_block_num, name="ids_indexb") as ids_indexb:
                    # id in segment
                    input_offset_gm = (ids_indexa * self.obj_scalar.ids_param.front + ids_indexb) * \
                                      self.obj_scalar.e_num + \
                                      e_mov_index_gm2ub * self.obj_scalar.e_out_param.front
                    if not block_align and last_e and (e_level == BIG_E or e_level == SMALL_E):
                        input_offset_gm = input_offset_gm - self.obj_scalar.align_scalar
                    input_offset_ub = ids_indexb * self.obj_scalar.e_num_part_ub_num
                    _tik_mov_input_gm2ub_continue(self.tik_inst, self.obj_gm_tensor.input_gm,
                                                  self.obj_ub_tensor.input_ub,
                                                  input_offset_gm, input_offset_ub, 1, e_lenBurst)

        with self.tik_inst.for_range(0, ids_block_num, name="ids_indexb") as ids_indexb:
            if big_iid:
                self.obj_scalar.id_val_scalar.set_as(self.obj_ub_tensor.ids_ub[ids_indexb])
            else:
                self.obj_scalar.id_val_scalar.set_as(
                    self.obj_ub_tensor.ids_ub[ids_indexa * self.obj_scalar.ids_param.front + ids_indexb])
            id_val_index = self.obj_scalar.id_val_scalar - \
                           num_segments_part_index * self.obj_scalar.num_segments_param.front

            with self.tik_inst.if_scope(tik.all(id_val_index >= 0, id_val_index < num_segments_part)):
                _tik_unsorted_segment_instructions(self.tik_inst,
                                                   self.obj_ub_tensor.input_ub[
                                                       ids_indexb * self.obj_scalar.e_num_part_ub_num],
                                                   self.obj_ub_tensor.output_ub[
                                                       id_val_index * self.obj_scalar.e_num_part_ub_num],
                                                   repeat_time_part, self.obj_tiling.mask, self.instruction)

    def _basic_module_ids_cal(self, num_segments_part_index, num_segments_part,
                              e_mov_index_gm2ub, block_align, e_level, last_e, big_iid):
        if last_e:
            e_lenBurst = self.obj_scalar.e_lenBurst_last
        else:
            e_lenBurst = self.obj_scalar.e_lenBurst_front
        # init output_ub
        _tik_init_ub_tensor_multi(self.tik_inst, self.obj_ub_tensor.output_ub,
                                  _ceil_div(num_segments_part * self.obj_scalar.e_num_part_ub_num,
                                            self.obj_tiling.mask),
                                  self.obj_tiling.mask,
                                  self.instruction)

        with self.tik_inst.for_range(0, self.obj_scalar.ids_param.times - 1,
                                     name="ids_indexa") as ids_indexa:
            # visit ids
            if big_iid:
                ids_offset_gm = ids_indexa * self.obj_scalar.ids_param.front
                ids_burst_len = _ceil_div(self.obj_scalar.ids_param.front, self.obj_tiling.ids_per_block)
                _tik_mov_ids_gm2ub(self.tik_inst, self.obj_gm_tensor.ids_gm, self.obj_ub_tensor.ids_ub, ids_offset_gm,
                                   0, 1, ids_burst_len)

            self._basic_module_get_input_and_cal(num_segments_part_index, num_segments_part, e_mov_index_gm2ub,
                                                 ids_indexa,
                                                 self.obj_scalar.ids_param.front,
                                                 e_lenBurst, block_align, e_level=e_level, last_e=last_e,
                                                 big_iid=big_iid)
        # visit ids
        if big_iid:
            ids_offset_gm = (self.obj_scalar.ids_param.times - 1) * \
                            self.obj_scalar.ids_param.front
            ids_burst_len = _ceil_div(self.obj_scalar.ids_param.last, self.obj_tiling.ids_per_block)
            _tik_mov_ids_gm2ub(self.tik_inst, self.obj_gm_tensor.ids_gm, self.obj_ub_tensor.ids_ub, ids_offset_gm,
                               0, 1, ids_burst_len)
        self._basic_module_get_input_and_cal(num_segments_part_index, num_segments_part, e_mov_index_gm2ub,
                                             self.obj_scalar.ids_param.times - 1,
                                             self.obj_scalar.ids_param.last,
                                             e_lenBurst, block_align, e_level=e_level, last_e=last_e,
                                             big_iid=big_iid)

    def _basic_module_ub2gm(self, num_segments_part_index, num_segments_part,
                            e_mov_index_gm2ub, block_align, e_level, last_e):
        if last_e:
            e_lenBurst = self.obj_scalar.e_lenBurst_last
        else:
            e_lenBurst = self.obj_scalar.e_lenBurst_front

        if e_level == SMALL_E and block_align:
            output_nburst = num_segments_part
            output_offset_gm = e_mov_index_gm2ub * self.obj_scalar.e_out_param.front + \
                               num_segments_part_index * self.obj_scalar.num_segments_param.front * \
                               self.obj_scalar.e_num
            dst_stride = self.obj_scalar.e_num // self.ele_num_per_block - e_lenBurst
            src_stride = self.obj_scalar.e_num_part_ub_num // self.ele_num_per_block - e_lenBurst
            _tik_mov_output_ub2gm_continue(self.tik_inst, self.obj_gm_tensor.output_gm, self.obj_ub_tensor.output_ub,
                                           output_offset_gm, 0, output_nburst, e_lenBurst, src_stride, dst_stride)
        else:
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, num_segments_part, name="segment_index") as segment_index:
                    output_offset_gm = e_mov_index_gm2ub * self.obj_scalar.e_out_param.front + (
                            num_segments_part_index * self.obj_scalar.num_segments_param.front + segment_index) * \
                                       self.obj_scalar.e_num
                    if not block_align and last_e:
                        output_offset_gm = output_offset_gm - self.obj_scalar.align_scalar

                    _tik_mov_output_ub2gm_continue(self.tik_inst, self.obj_gm_tensor.output_gm,
                                                   self.obj_ub_tensor.output_ub,
                                                   output_offset_gm, segment_index * self.obj_scalar.e_num_part_ub_num,
                                                   1, e_lenBurst)

    def _basic_module_block_e_ub2gm(self, num_segments_part_index, num_segments_part, e_mov_index_gm2ub):
        num_segments_part_front = num_segments_part * self.obj_scalar.e_out_param.last - self.ele_num_per_block
        align_ub = self.tik_inst.Tensor(self.dtype, (self.ele_num_per_block,), name="align_ub",
                                        scope=tik.scope_ubuf)
        with self.tik_inst.for_range(0, num_segments_part, name="segment_index") as segment_index:
            with self.tik_inst.for_range(0, self.obj_scalar.e_out_param.last, name="ele_i") as ele_i:
                out_index = segment_index * self.obj_scalar.e_out_param.last + ele_i
                self.obj_ub_tensor.output_ub[out_index].set_as(
                    self.obj_ub_tensor.output_ub[segment_index * self.obj_scalar.e_num_part_ub_num + ele_i])
                with self.tik_inst.if_scope(out_index >= num_segments_part_front):
                    align_ub[out_index - num_segments_part_front].set_as(
                        self.obj_ub_tensor.output_ub[segment_index * self.obj_scalar.e_num_part_ub_num + ele_i])

        output_offset_gm = e_mov_index_gm2ub * self.obj_scalar.e_out_param.front + \
                           num_segments_part_index * self.obj_scalar.num_segments_param.front * self.obj_scalar.e_num
        output_lenburst = num_segments_part * self.obj_scalar.e_out_param.last // self.ele_num_per_block
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.if_scope(output_lenburst > 0):
                if isinstance(output_lenburst, int) and output_lenburst < 1:
                    output_lenburst = 1
                    num_segments_part_front = 0
                _tik_mov_output_ub2gm_continue(self.tik_inst, self.obj_gm_tensor.output_gm,
                                               self.obj_ub_tensor.output_ub, output_offset_gm, 0, 1, output_lenburst)
                _tik_mov_output_ub2gm_continue(self.tik_inst, self.obj_gm_tensor.output_gm, align_ub,
                                               output_offset_gm + num_segments_part_front, 0, 1, 1)
            with self.tik_inst.else_scope():
                _tik_mov_output_ub2gm_continue(self.tik_inst, self.obj_gm_tensor.output_gm,
                                               self.obj_ub_tensor.output_ub, output_offset_gm, 0, 1, 1)

    def _basic_module_one_e_ub2gm(self, num_segments_part_index, num_segments_part, e_mov_index_gm2ub):
        e_lenBurst = self.obj_scalar.e_lenBurst_last
        align_ub = self.tik_inst.Tensor(self.dtype, (self.ele_num_per_block,), name="align_ub",
                                        scope=tik.scope_ubuf)
        for ele_i in range(self.ele_num_per_block):
            align_ub[ele_i].set_as(
                self.obj_ub_tensor.output_ub[(num_segments_part - 1) * self.obj_scalar.e_num_part_ub_num +
                                             self.obj_scalar.e_out_param.last -
                                             self.ele_num_per_block + ele_i])

        with self.tik_inst.for_range(0, num_segments_part - 1, name="segment_index") as segment_index:
            output_offset_gm = e_mov_index_gm2ub * self.obj_scalar.e_out_param.front + (
                    num_segments_part_index * self.obj_scalar.num_segments_param.front + segment_index) \
                               * self.obj_scalar.e_num
            _tik_mov_output_ub2gm_continue(self.tik_inst, self.obj_gm_tensor.output_gm,
                                           self.obj_ub_tensor.output_ub,
                                           output_offset_gm, segment_index * self.obj_scalar.e_num_part_ub_num,
                                           1, e_lenBurst)
        output_offset_gm = e_mov_index_gm2ub * self.obj_scalar.e_out_param.front + (
                num_segments_part_index * self.obj_scalar.num_segments_param.front + num_segments_part - 1) \
                           * self.obj_scalar.e_num
        _tik_mov_output_ub2gm_continue(self.tik_inst, self.obj_gm_tensor.output_gm,
                                       self.obj_ub_tensor.output_ub,
                                       output_offset_gm, (num_segments_part - 1) * self.obj_scalar.e_num_part_ub_num,
                                       1, e_lenBurst - 1)
        output_offset_gm = output_offset_gm + self.obj_scalar.e_out_param.last - self.ele_num_per_block

        _tik_mov_output_ub2gm_continue(self.tik_inst, self.obj_gm_tensor.output_gm, align_ub, output_offset_gm,
                                       0, 1, 1)

    def _basic_module_each_div_e(self, num_segments_block_index, e_mov_index_gm2ub, last_e,
                                 block_align, e_level, big_iid, div_oid):

        def _call_func(num_segments_part_index, num_segments_part, e_mov_index_gm2ub, last_e):
            if e_level == ONE_BLOCK_E:
                self._basic_module_ids_cal(num_segments_part_index, num_segments_part, e_mov_index_gm2ub,
                                           block_align=False, e_level=e_level, last_e=True, big_iid=big_iid)
                self._basic_module_block_e_ub2gm(num_segments_part_index, num_segments_part, e_mov_index_gm2ub)
            elif e_level == ONE_DIV_E:
                self._basic_module_ids_cal(num_segments_part_index, num_segments_part, e_mov_index_gm2ub,
                                           block_align=False, e_level=e_level, last_e=True, big_iid=big_iid)
                self._basic_module_one_e_ub2gm(num_segments_part_index, num_segments_part, e_mov_index_gm2ub)
            else:
                self._basic_module_ids_cal(num_segments_part_index, num_segments_part, e_mov_index_gm2ub,
                                           block_align=block_align, e_level=e_level, last_e=last_e, big_iid=big_iid)
                self._basic_module_ub2gm(num_segments_part_index, num_segments_part, e_mov_index_gm2ub,
                                         block_align=block_align, e_level=e_level, last_e=last_e)

        if div_oid:
            with self.tik_inst.if_scope(
                    num_segments_block_index < self.obj_scalar.num_segments_loop_param.times - 1):
                with self.tik_inst.for_range(0, self.obj_scalar.num_segments_loop_param.front,
                                             name="num_segments_part_index") as num_segments_part_index:
                    _call_func(
                        num_segments_block_index * self.obj_scalar.num_segments_loop_param.front +
                        num_segments_part_index,
                        self.obj_scalar.num_segments_param.front,
                        e_mov_index_gm2ub, last_e=last_e)

            with self.tik_inst.if_scope(
                    num_segments_block_index == self.obj_scalar.num_segments_loop_param.times - 1):
                with self.tik_inst.for_range(0, self.obj_scalar.num_segments_loop_param.last - 1,
                                             name="num_segments_part_index") as num_segments_part_index:
                    _call_func(
                        (self.obj_scalar.num_segments_loop_param.times - 1) *
                        self.obj_scalar.num_segments_loop_param.front + num_segments_part_index,
                        self.obj_scalar.num_segments_param.front,
                        e_mov_index_gm2ub, last_e=last_e)

                _call_func(self.obj_scalar.num_segments_param.times - 1,
                           self.obj_scalar.num_segments_param.last,
                           e_mov_index_gm2ub, last_e=last_e)
        else:
            with self.tik_inst.for_range(0, self.obj_scalar.num_segments_param.times - 1,
                                         name="num_segments_part_index") as num_segments_part_index:
                _call_func(num_segments_part_index,
                           self.obj_scalar.num_segments_param.front,
                           e_mov_index_gm2ub,
                           last_e=last_e)

            _call_func(self.obj_scalar.num_segments_param.times - 1,
                       self.obj_scalar.num_segments_param.last,
                       e_mov_index_gm2ub,
                       last_e=last_e)

    def tik_template_interface(self, obj_scalar, block_align, e_level, big_iid, div_oid):
        """
        tik_template_interface

        Parameters
        ----------
        block_align: block_align
        e_level: e_level
        big_iid: big_iid
        div_oid: div_oid

        Returns
        -------
        None
        """
        self.obj_scalar = obj_scalar
        if e_level == ONE_DIV_E or e_level == ONE_BLOCK_E:
            num_segments_block_index = self.block_index
            if not big_iid:
                _tik_mov_ids_gm2ub(self.tik_inst, self.obj_gm_tensor.ids_gm, self.obj_ub_tensor.ids_ub, 0,
                                   0, 1, self.obj_scalar.ids_last_burst_len)

            self._basic_module_each_div_e(num_segments_block_index, 0, last_e=True,
                                          block_align=block_align, e_level=e_level, big_iid=big_iid, div_oid=True)
            return

        if div_oid:
            e_block_index = self.block_index % self.obj_scalar.e_out_loop_param.times
            num_segments_block_index = self.block_index // self.obj_scalar.e_out_loop_param.times
        else:
            e_block_index = self.block_index
            num_segments_block_index = 0

        with self.tik_inst.if_scope(e_block_index < self.obj_scalar.e_out_loop_param.times - 1):
            if not big_iid:
                _tik_mov_ids_gm2ub(self.tik_inst, self.obj_gm_tensor.ids_gm, self.obj_ub_tensor.ids_ub, 0,
                                   0, 1, self.obj_scalar.ids_last_burst_len)
            with self.tik_inst.for_range(0, self.obj_scalar.e_out_loop_param.front,
                                         name="e_mov_index") as e_mov_index_gm2ub:
                self._basic_module_each_div_e(num_segments_block_index,
                                              e_block_index * self.obj_scalar.e_out_loop_param.front +
                                              e_mov_index_gm2ub, last_e=False,
                                              block_align=block_align, e_level=e_level, big_iid=big_iid,
                                              div_oid=div_oid)

        with self.tik_inst.if_scope(e_block_index == self.obj_scalar.e_out_loop_param.times - 1):
            if not big_iid:
                _tik_mov_ids_gm2ub(self.tik_inst, self.obj_gm_tensor.ids_gm, self.obj_ub_tensor.ids_ub, 0,
                                   0, 1, self.obj_scalar.ids_last_burst_len)
            with self.tik_inst.for_range(0, self.obj_scalar.e_out_loop_param.last - 1,
                                         name="e_mov_index") as e_mov_index_gm2ub:
                self._basic_module_each_div_e(num_segments_block_index,
                                              e_block_index * self.obj_scalar.e_out_loop_param.front +
                                              e_mov_index_gm2ub, last_e=False,
                                              block_align=block_align, e_level=e_level, big_iid=big_iid,
                                              div_oid=div_oid)

            self._basic_module_each_div_e(num_segments_block_index,
                                          self.obj_scalar.e_out_param.times - 1,
                                          last_e=True,
                                          block_align=block_align, e_level=e_level, big_iid=big_iid, div_oid=div_oid)


class UnsortedSegment(object):
    """
        Function: use to store concat base parameters
    """

    def __init__(self, x_dict, segment_ids_dict, num_segments_dict, y_dict, kernel_name, instruction):
        """
        constructor of class UnsortedSegment

        Parameters
        ----------
        x_dict: dict
            shape and dtype of x
        segment_ids_dict: dict
            shape and dtype of segment_ids
        num_segments_dict: dict
            shape and dtype of num_segments
        y_dict: dict
            shape and dtype of y
        kernel_name: str
            kernel_name, default value is "UnsortedSegment"
        instruction: str
            instruction: "segment_min", "segment_max", "segment_sum", "segment_prod"

        Returns
        -------
        None
        """
        # get dtype
        self.input_dtype = x_dict.get("dtype", None)
        self.input_dtype = self.input_dtype.lower()
        self.ids_dtype = segment_ids_dict.get("dtype", None)
        self.ids_dtype = self.ids_dtype.lower()
        self.num_segments_dtype = num_segments_dict.get("dtype", None)
        self.num_segments_dtype = self.num_segments_dtype.lower()
        self.output_dtype = self.input_dtype
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.core_num = _tik_get_core_num()
        self.ub_tensor_num = 3
        self.instruction = instruction

        self.obj_tiling = UnsortedSegmentTiling(self, x_dict, segment_ids_dict, y_dict)

        class GmTensor(object):
            """
            Function: use to store concat base parameters
            Modify : 2020-12-9
            """

            def __init__(self, tik_instance, input_dtype, ids_dtype,
                         num_segments_dtype, obj_scalar):
                """
                constructor of class GmTensor

                Parameters
                ----------
                tik_instance: tik_instance
                input_dtype: x dtype
                ids_dtype: ids dtype
                num_segments_dtype: num_segments dtype

                Returns
                -------
                None
                """
                self.input_gm = tik_instance.Tensor(input_dtype, (MAX_INT32,), name="input_gm", scope=tik.scope_gm)
                self.ids_gm = tik_instance.Tensor(ids_dtype, (MAX_INT32,), name="ids_gm", scope=tik.scope_gm)

                self.num_segments_gm = tik_instance.Tensor(num_segments_dtype, (MIN_TENSOR_ELE_NUM,),
                                                           name="num_segments_gm", scope=tik.scope_gm)

                self.output_gm = tik_instance.Tensor(input_dtype, (MAX_INT32,), name="output_gm", scope=tik.scope_gm)

                self.tiling_gm = tik_instance.Tensor(TILING_PARAM_DTYPE, (TILING_PARAMS_NUM,), name="tiling_gm",
                                                     scope=tik.scope_gm)

        class UbTensor(object):
            """
            Function: use to store concat base parameters
            Modify : 2020-12-9
            """

            def __init__(self, param):
                """
                constructor of class UbTensor

                Parameters
                ----------
                None

                Returns
                -------
                None
                """
                self.num_segments_ub = param.tik_instance.Tensor(param.num_segments_dtype, (MIN_TENSOR_ELE_NUM,),
                                                                 name="num_segments_ub", scope=tik.scope_ubuf)
                self.input_ub = None
                self.ids_ub = None
                self.output_ub = None
                self.tiling_ub = None

            def set_ub(self, param, ids_num, input_num, output_num):
                """
                Function: declar ub tensor.
                """
                self.input_ub = param.tik_instance.Tensor(param.input_dtype, (input_num,), name="input_ub",
                                                          scope=tik.scope_ubuf)
                self.ids_ub = param.tik_instance.Tensor(param.ids_dtype, (ids_num,), name="ids_ub",
                                                        scope=tik.scope_ubuf)
                self.output_ub = param.tik_instance.Tensor(param.output_dtype, (output_num,), name="output_ub",
                                                           scope=tik.scope_ubuf)

        self.obj_gm_tensor = GmTensor(self.tik_instance, self.input_dtype,
                                      self.ids_dtype, self.num_segments_dtype, self.obj_tiling.obj_scalar)
        self.obj_ub_tensor = UbTensor(self)

    def unsorted_segment(self, dynamic_mode):
        """
        main process of unsorted_segment

        Parameters
        ----------
        None

        Returns:
        -------
        None
        """
        obj_scalar = self.obj_tiling.scalars[0]
        with self.tik_instance.new_stmt_scope():
            # num_segments
            if not isinstance(obj_scalar.num_segments, int):
                self.tik_instance.data_move(self.obj_ub_tensor.num_segments_ub, self.obj_gm_tensor.num_segments_gm,
                                            0, 1, 1, 0, 0)
                obj_scalar.num_segments.set_as(self.obj_ub_tensor.num_segments_ub[0])

        with self.tik_instance.new_stmt_scope():
            scalar_list = [
                obj_scalar.select_key,
                obj_scalar.e_out_loop_param.times,
                obj_scalar.e_out_loop_param.front,
                obj_scalar.e_out_loop_param.last,
                obj_scalar.ids_param.times,
                obj_scalar.ids_param.front,
                obj_scalar.ids_param.last,
                obj_scalar.e_out_param.times,
                obj_scalar.e_out_param.front,
                obj_scalar.e_out_param.last,
                obj_scalar.num_segments_param.times,
                obj_scalar.num_segments_param.front,
                obj_scalar.num_segments_param.last,
                obj_scalar.num_segments_loop_param.times,
                obj_scalar.num_segments_loop_param.front,
                obj_scalar.num_segments_loop_param.last,
                obj_scalar.e_num_part_ub_num,
                obj_scalar.e_num,
                obj_scalar.repeat_time_front_part,
                obj_scalar.repeat_time_last_part,
                obj_scalar.align_scalar,
                obj_scalar.e_lenBurst_front,
                obj_scalar.e_lenBurst_last,
                obj_scalar.ids_last_burst_len]

            scalar_in_list = False
            for ele in scalar_list:
                if not (isinstance(ele, int) or isinstance(ele, dict)):
                    scalar_in_list = True

            if scalar_in_list:
                self.obj_ub_tensor.tiling_ub = self.tik_instance.Tensor(
                    TILING_PARAM_DTYPE, (TILING_PARAMS_NUM,), name="tiling_ub",
                    scope=tik.scope_ubuf)
                # mov tiling params from gm to ub
                self.tik_instance.data_move(self.obj_ub_tensor.tiling_ub,
                                            self.obj_gm_tensor.tiling_gm, 0, 1,
                                            TILING_PARAMS_NUM * BYTE_INT32 // \
                                            BYTE_BLOCK, 0, 0)

            index = 0
            for ele in scalar_list:
                if not (isinstance(ele, int) or isinstance(ele, dict)):
                    ele.set_as(self.obj_ub_tensor.tiling_ub[index])
                index = index + 1

        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_index:

            tp = TikTemplate(block_index, self.tik_instance, self.input_dtype, self.obj_gm_tensor, self.obj_ub_tensor,
                             self.obj_tiling, self.instruction)
            if isinstance(self.obj_tiling.obj_scalar.select_key, dict):
                scalar_select_key = self.obj_tiling.obj_scalar.select_key["select_key"]
            else:
                scalar_select_key = self.obj_tiling.obj_scalar.select_key

            ids_once_num = self.obj_tiling.ids_once_num

            def _func_interface_div_by_e(key):
                if scalar_select_key == key["select_key"] if isinstance(scalar_select_key, int) else True:
                    with self.tik_instance.new_stmt_scope():
                        with self.tik_instance.if_scope(scalar_select_key == key["select_key"]):
                            scalar = self.obj_tiling.scalars[key["ub_div_id"]]
                            self.obj_ub_tensor.set_ub(self, ids_once_num, scalar.input_once_num, scalar.output_once_num)
                            tp.tik_template_interface(scalar,
                                                      block_align=key["block_align"],
                                                      e_level=key["e_level"],
                                                      big_iid=key["big_iid"],
                                                      div_oid=key["div_oid"])

            for key in self.obj_tiling.select_keys_for_compile:
                _func_interface_div_by_e(key)

        if dynamic_mode:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=[self.obj_gm_tensor.input_gm,
                                               self.obj_gm_tensor.ids_gm,
                                               self.obj_gm_tensor.num_segments_gm],
                                       outputs=[self.obj_gm_tensor.output_gm],
                                       flowtable=[self.obj_gm_tensor.tiling_gm],
                                       config={'enable_const_fold': True, "out_of_bound_sync_check": True})
        else:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=[self.obj_gm_tensor.input_gm,
                                               self.obj_gm_tensor.ids_gm],
                                       outputs=[self.obj_gm_tensor.output_gm],
                                       config={'enable_const_fold': True, "out_of_bound_sync_check": True})


def _tik_get_ub_size(is_double_buffer=True):
    """
    get ub size

    Parameters
    ----------
    is_double_buffer: is_double_buffer flag

    Returns
    -------
    ub_size
    """
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 512
    if is_double_buffer:
        return ub_size // 2
    return ub_size


def _tik_get_core_num():
    """
    get core num

    Parameters
    ----------
    None

    Returns
    -------
    core num
    """
    return tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)


def _tik_init_ub_tensor_once(tik_inst, ub_tensor, repeat_time, mask, instruction):
    """
    init ub tensor once

    Parameters
    ----------
    tik_inst: tik instance
    ub_tensor: ub_tensor
    repeat_time: repeat time
    mask: mask
    instructions: segment_min, segment_max, segment_prod, segment_sum

    Returns
    -------
    None
    """
    if isinstance(repeat_time, int) and (repeat_time <= 0 or repeat_time > 255):
        return

    if instruction == 'segment_min':
        tik_inst.vector_dup(mask, ub_tensor, _get_dtype_max_val(ub_tensor.dtype), repeat_time, 1, 8)
    elif instruction == 'segment_max':
        tik_inst.vector_dup(mask, ub_tensor, _get_dtype_min_val(ub_tensor.dtype), repeat_time, 1, 8)
    elif instruction == 'segment_prod':
        tik_inst.vector_dup(mask, ub_tensor, 1, repeat_time, 1, 8)
    elif instruction == 'segment_sum':
        tik_inst.vector_dup(mask, ub_tensor, 0, repeat_time, 1, 8)
    else:
        raise RuntimeError('operation %s not support yes' % instruction)


def _tik_init_ub_tensor_multi(tik_inst, ub_tensor, repeat_time, mask, instruction):
    """
    init ub tensor multi

    Parameters
    ----------
    tik_inst: tik instance
    ub_tensor: ub_tensor
    repeat_time: repeat time
    mask: mask
    instructions: segment_min, segment_max, segment_prod, segment_sum

    Returns
    -------
    None
    """
    for_times = repeat_time // 255
    with tik_inst.new_stmt_scope(disable_sync=True):
        with tik_inst.for_range(0, for_times, name="init_ub_index") as init_ub_index:
            _tik_init_ub_tensor_once(tik_inst, ub_tensor[init_ub_index * 255 * mask], 255, mask, instruction)

        with tik_inst.if_scope(repeat_time % 255 > 0):
            _tik_init_ub_tensor_once(tik_inst, ub_tensor[for_times * 255 * mask], repeat_time % 255, mask, instruction)


def _tik_unsorted_segment_instructions(tik_inst, input_ub, output_ub, repeat_time, mask, instructions):
    """
    tik_vadd

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input ub tensor
    output_ub: output ub tensor
    repeat_time: repeat time
    mask: mask
    instructions: segment_min, segment_max, segment_prod, segment_sum

    Returns
    -------
    None
    """
    tik_funcs = {'segment_min': tik_inst.vmin, 'segment_max': tik_inst.vmax, 'segment_prod': tik_inst.vmul,
                 'segment_sum': tik_inst.vadd}
    tik_funcs[instructions](mask, output_ub, output_ub, input_ub, repeat_time, 1, 1, 1, 8, 8, 8)


def _tik_mov_output_ub2gm_continue(tik_inst,
                                   output_gm,
                                   output_ub,
                                   output_offset_gm,
                                   output_offset_ub,
                                   output_nburst,
                                   output_lenburst, src_stride=0, dst_stride=0):
    """
    tik_mov_output_ub2gm_continue

    Parameters
    ----------
    tik_inst: tik instance
    output_gm: output gm tensor
    output_ub: output ub tensor
    output_offset_gm: output offset gm
    output_offset_ub: output offset ub
    output_nburst: n_burst
    output_lenburst: burst_len
    src_stride: src_stride
    dst_stride: dst_stride

    Returns
    -------
    None
    """
    if isinstance(src_stride, int) and (src_stride < 0 or src_stride > 65535):
        return
    if isinstance(dst_stride, int) and (dst_stride < 0 or dst_stride > 65535):
        return
    tik_inst.data_move(output_gm[output_offset_gm], output_ub[output_offset_ub], 0, output_nburst, output_lenburst,
                       src_stride, dst_stride)


def _tik_mov_input_gm2ub_continue(tik_inst,
                                  input_gm,
                                  input_ub,
                                  input_offset_gm,
                                  input_offset_ub,
                                  input_n_burst,
                                  input_burst_len, src_stride=0, dst_stride=0):
    """
    tik_mov_input_gm2ub_continue

    Parameters
    ----------
    tik_inst: tik instance
    input_gm: input gm tensor
    input_ub: input ub tensor
    input_offset_gm: input offset gm
    input_offset_ub: input offset ub
    input_n_burst: n_burst
    input_burst_len: burst_len
    src_stride: src_stride
    dst_stride: dst_stride

    Returns
    -------
    None
    """
    if isinstance(src_stride, int) and (src_stride < 0 or src_stride > 65535):
        return
    if isinstance(dst_stride, int) and (dst_stride < 0 or dst_stride > 65535):
        return
    tik_inst.data_move(input_ub[input_offset_ub], input_gm[input_offset_gm], 0, input_n_burst, input_burst_len,
                       src_stride, dst_stride)


def _tik_mov_ids_gm2ub(tik_inst,
                       ids_gm,
                       ids_ub,
                       ids_offset_gm,
                       ids_offset_ub,
                       ids_n_burst,
                       ids_burst_len):
    """
    tik_mov_ids_gm2ub

    Parameters
    ----------
    tik_inst: tik instance
    ids_gm: ids_gm tensor
    ids_ub: ids_ub tensor
    ids_offset_gm: ids_offset_gm
    ids_offset_ub: ids_offset_ub
    ids_n_burst: ids_n_burst
    ids_burst_len: ids_burst_len

    Returns
    -------
    None
    """
    tik_inst.data_move(ids_ub[ids_offset_ub], ids_gm[ids_offset_gm], 0, ids_n_burst, ids_burst_len, 0, 0)


def unsorted_segment(x_dict, segment_ids_dict, num_segments_dict, y_dict,
                     kernel_name="UnsortedSegment", instruction='segment_min',
                     dynamic_mode=True):
    """
    unsorted_segment entry interface

    Parameters
    ----------
    x_dict: input params shape, dtype and range
    segment_ids_dict: segment_ids shape, dtype and range
    num_segments_dict: num_segments shape, dtype and range
    y_dict: output shape, dtype and range
    kernel_name: kernel name of UnsortedSegment op
    instruction: "segment_min", "segment_max", "segment_sum", "segment_prod"
    dynamic_mode: dynamic mode

    Returns
    -------
    compile info
    """
    if dynamic_mode:
        def _cmp_as_list(a, b):
            if len(a) != len(b):
                return False
            for i, ele in enumerate(a):
                if list(ele) != list(b[i]):
                    return False
            return True

        if not _cmp_as_list(x_dict['range'][:len(segment_ids_dict['range'])], segment_ids_dict['range']):
            error_manager_vector.raise_err_specific_reson(kernel_name, "range of segment_ids_dict cannot fit x_dict!")
        if not _cmp_as_list(x_dict['range'][len(segment_ids_dict['range']):], y_dict['range'][1:]):
            error_manager_vector.raise_err_specific_reson(kernel_name, "range of y_dict cannot fit x_dict!")

    obj = UnsortedSegment(x_dict, segment_ids_dict, num_segments_dict, y_dict, kernel_name, instruction)
    obj.unsorted_segment(dynamic_mode)
    # add compile info
    if dynamic_mode:
        tbe_context.get_context().add_compile_info("vars",
                                                   {"ub_size": obj.obj_tiling.ub_size, "core_num": obj.core_num,
                                                    "dtype": obj.obj_gm_tensor.input_gm.dtype,
                                                    "ub_tensor_num": obj.ub_tensor_num})
