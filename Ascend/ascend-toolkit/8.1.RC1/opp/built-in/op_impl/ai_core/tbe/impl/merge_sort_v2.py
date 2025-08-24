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
merge_sort
"""
from impl.util.platform_adapter import PlatformApi
from impl.util.platform_adapter import error_manager_vector
from impl import constant_util as constant


def check_soc_version_support(soc_version, soc_version_all):
    for version_support in soc_version_all:
        if soc_version == version_support:
            return True
    return False


def ceil_div(dividend, divisor):
    """
    ceiling division
    """
    result = (dividend + divisor - 1) // divisor
    return result


def get_align_num(data_num_total, data_num_align, ceil=True):
    """
    Calculate the nearest number to data_num_total that can be divisible by data_num_align
    """
    if ceil:
        result = ceil_div(data_num_total, data_num_align) * data_num_align
    else:
        result = data_num_total // data_num_align * data_num_align
    return result


def get_loop_info(data_num_total, data_num_each_loop):
    """
    Calculate the number of cycles and the last data amount required for each cycle
    with a total data amount of data_num_total and a data amount of each cycle of data_num_each_loop
    """
    loop_times = ceil_div(data_num_total, data_num_each_loop)
    last_loop_num = data_num_total - data_num_each_loop * (loop_times - 1)
    return loop_times, last_loop_num


def emit_vmuls(tik_inst, dst, src, cnt, dtype):
    """
    emit vmuls
    """
    if dtype == "float32":
        mask = 64
    else:
        mask = 128
    repeat = cnt // mask
    repeat_remain = cnt % mask
    times = (repeat + constant.MAX_REPEAT_TIMES - 1) // constant.MAX_REPEAT_TIMES
    if repeat > 0:
        with tik_inst.for_range(0, times, name="vmuls_i0") as i:
            src0_scalar = tik_inst.Scalar(dtype="int64", name="src0_scalar",
                                          init_value=repeat - i * constant.MAX_REPEAT_TIMES)
            src1_scalar = tik_inst.Scalar(dtype="int64", name="src1_scalar",
                                          init_value=constant.MAX_REPEAT_TIMES)
            times_len = tik_inst.Scalar(dtype="int64", name="times_len")
            tik_inst.scalar_min(times_len, src0_scalar, src1_scalar)
            tik_inst.vmuls(mask, dst[i * mask * constant.MAX_REPEAT_TIMES],
                           src[i * mask * constant.MAX_REPEAT_TIMES],
                           -1, times_len, 1, 1, 8, 8)
    if repeat_remain > 0:
        tik_inst.vmuls(repeat_remain, dst[repeat * mask], src[repeat * mask], -1, 1, 1, 1, 8, 8)


class BaseConstant(object):
    """
    Constant data required for merge sort
    """

    def __init__(self, score_type="float32", index_type="uint32", ub_size=None, kernel_name="merge_sort_const_value"):
        self.repeat_num_max_cmd = 255
        self.channel_num_max_sort = 4
        self.proposal_size = 8
        self.block_size = 32
        self.repeat_size = 256
        self.proposal_num_repeat = 32
        self.proposal_num_channel_max_ub = 65535
        self.proposal_num_block = self.block_size // self.proposal_size
        self.ub_size_repeat_sort = self.proposal_num_repeat * self.proposal_size
        self.ub_size_min_align = self.ub_size_repeat_sort * self.channel_num_max_sort * 2

        self.kernel_name = kernel_name

        self.score_type, self.index_type = self._init_type(score_type, index_type)
        self.score_size, self.score_num_block, self.score_num_repeat = self._init_cmd_data_num(self.score_type)
        self.index_size, self.index_num_block, self.index_num_repeat = self._init_cmd_data_num(self.index_type)
        self.ub_size_sum = self._init_ub_size(ub_size)
        self.neg_inf = self._init_neg_inf()
        self.element_num_proposal = self.proposal_size // self.score_size
        self.proposal_num_max_ub = self.ub_size_sum // 2 // self.proposal_size

    def _init_type(self, score_type, index_type):
        """
        Initialize the data type of the score and the data type of the index.
        Check whether the data type of score and index meet the requirements.
        """
        score_type_supported = ("float16", "float32")
        index_type_supported = ("uint32",)
        assert score_type in score_type_supported, \
            error_manager_vector.raise_err_input_dtype_not_supported(self.kernel_name,
                                                                     "score",
                                                                     str(score_type_supported),
                                                                     score_type)
        assert index_type in index_type_supported, \
            error_manager_vector.raise_err_input_dtype_not_supported(self.kernel_name,
                                                                     "index",
                                                                     str(index_type_supported),
                                                                     index_type)
        return score_type, index_type

    def _init_cmd_data_num(self, data_type):
        """
        Calculate the amount of data consumed by each command according to the data_type
        """
        data_size_all = {"float16": 2, "float32": 4, "uint32": 4}
        data_size = data_size_all.get(data_type)
        data_num_block = self.block_size // data_size
        data_num_repeat = self.repeat_size // data_size
        return data_size, data_num_block, data_num_repeat

    def _init_ub_size(self, ub_size_input):
        """
        Initialize ub size.
        Check whether ub_size_input meets the requirements.
        """
        ub_size_ori = PlatformApi.get_soc_spec(PlatformApi.UB_SIZE)
        if ub_size_input is None:
            ub_size = ub_size_ori
        else:
            excepted_value = "between {} and {}".format(self.ub_size_min_align, ub_size_ori)
            assert self.ub_size_min_align <= ub_size_input <= ub_size_ori, \
                error_manager_vector.raise_err_input_value_invalid(self.kernel_name,
                                                                   "ub_size",
                                                                   excepted_value,
                                                                   ub_size_input)
            ub_size = ub_size_input
        ub_align_size = get_align_num(ub_size, self.ub_size_min_align, ceil=False)
        return ub_align_size

    def _init_neg_inf(self):
        """
        A constant representing negative infinitesimal is determined according to the data type of the score
        """
        const_neg_inf = {
            "float16": -(2**16 - 1.0),
            "float32": -3.40282346638528860E38
        }
        return const_neg_inf.get(self.score_type)


class ProposalSortUB(object):
    """
    Sort proposal in proposal_num_repeat order in UB space.
    """

    def __init__(self, tik, tik_inst, const_value):
        self._tik = tik
        self._tik_inst = tik_inst
        self._pro_num_repeat = const_value.proposal_num_repeat
        self._repeat_num_max_cmd = const_value.repeat_num_max_cmd

    def proposal_sort_ub_start(self, proposal_ub, pro_num, get_score_index_tensor, fun_args):
        """
        start sort
        Parameters
        ----------
        proposal_ub:
            TVM ub tensor. Dtype: float16 or float32. Shape: (pro_num, pro_element)
        pro_num:
            Int. Number of proposal to sort.
        get_score_index_tensor:
            Function. Function to get score tensor and index tensor.
        fun_args:
            Dict. Input parameters of get_score_index_tensor
        """
        with self._tik_inst.new_stmt_scope():
            score_tensor_ub, index_tensor_ub = get_score_index_tensor(fun_args)
            vsort_repeat_num = ceil_div(pro_num, self._pro_num_repeat)
            repeat_loop_times, last_loop_repeat_num = get_loop_info(vsort_repeat_num, self._repeat_num_max_cmd)
            for loop_index in range(repeat_loop_times):
                data_index = loop_index * self._repeat_num_max_cmd * self._pro_num_repeat
                if loop_index != repeat_loop_times - 1:
                    self._tik_inst.vsort32(proposal_ub[data_index, 0], score_tensor_ub[data_index],
                                           index_tensor_ub[data_index], self._repeat_num_max_cmd)
                else:
                    self._tik_inst.vsort32(proposal_ub[data_index, 0], score_tensor_ub[data_index],
                                           index_tensor_ub[data_index], last_loop_repeat_num)


class ProposalMergeUB(object):
    """
    Merge partially ordered proposal in UB space.
    """

    def __init__(self, tik, tik_inst, const_value):
        self._tik = tik
        self._tik_inst = tik_inst

        self._c_num_max_sort = const_value.channel_num_max_sort
        self._pro_num_block = const_value.proposal_num_block

    def proposal_merge_ub_start(self, proposal_ub_0, proposal_ub_1, pro_num_dst, pro_num_src):
        """
        start merge proposal
        Parameters
        ----------
        proposal_ub_0:
            TVM ub tensor. Partially ordered proposal. Dtype: float16 or float32. Shape: (pro_num, pro_element)
        proposal_ub_1:
            TVM ub tensor. The data type and shape are the same as proposal_ub_0
        pro_num_dst:
            int. Number of proposal to sort.
        pro_num_src
            int. Number of proposal already sorted.
        Returns
        -------
        proposal_ub_0:
            sorted tensor
        proposal_ub_1:
            work tensor
        """
        while pro_num_src < pro_num_dst:
            pro_num_src_next = pro_num_src * self._c_num_max_sort
            repeat_times = pro_num_dst // pro_num_src_next
            if repeat_times > 0:
                pro_tensor_src_ub_list = tuple(proposal_ub_0[c_index * pro_num_src, 0]
                                               for c_index in range(self._c_num_max_sort))
                pro_num_src_list = tuple(pro_num_src for _ in range(self._c_num_max_sort))
                self._tik_inst.vmrgsort(proposal_ub_1, pro_tensor_src_ub_list, pro_num_src_list, False, repeat_times)
            pro_num_merged = pro_num_src_next * repeat_times
            pro_num_not_merged = pro_num_dst - pro_num_merged
            if pro_num_not_merged > 0:
                if pro_num_not_merged <= pro_num_src:
                    block_num_move = pro_num_not_merged // self._pro_num_block
                    self._tik_inst.data_move(proposal_ub_1[pro_num_merged, 0],
                                             proposal_ub_0[pro_num_merged, 0],
                                             0, 1, block_num_move, 0, 0)
                else:
                    channel_not_merge = pro_num_not_merged // pro_num_src
                    pro_tensor_rem_src_ub_list = tuple(proposal_ub_0[pro_num_merged + c_index * pro_num_src, 0]
                                                       for c_index in range(channel_not_merge))
                    pro_num_rem_src_ub_list = tuple(pro_num_src for _ in range(channel_not_merge))
                    pro_num_last = pro_num_not_merged - channel_not_merge * pro_num_src
                    if pro_num_last != 0:
                        pro_tensor_rem_src_ub_list += (proposal_ub_0[pro_num_merged + channel_not_merge * pro_num_src,
                                                                     0],)
                        pro_num_rem_src_ub_list += (pro_num_last,)
                    self._tik_inst.vmrgsort(proposal_ub_1[pro_num_merged, 0],
                                            pro_tensor_rem_src_ub_list,
                                            pro_num_rem_src_ub_list, False, 1)
            pro_num_src *= self._c_num_max_sort
            proposal_ub_0, proposal_ub_1 = proposal_ub_1, proposal_ub_0
        return proposal_ub_0, proposal_ub_1


class ProposalMergeGM(object):
    """
    Merge partially ordered proposal in gm space.
    """

    def __init__(self, tik, tik_inst, const_value):
        self._tik = tik
        self._tik_inst = tik_inst
        self._tensor_name = "proposal_merge_gm"

        self._score_type = const_value.score_type
        self._int_type = "int64"
        self._pro_num_max_ub = const_value.proposal_num_max_ub
        self._ele_num_pro = const_value.element_num_proposal
        self._pro_num_block = const_value.proposal_num_block
        self._pro_num_repeat = const_value.proposal_num_repeat
        self._neg_inf = const_value.neg_inf
        self._pro_num_c_max_ub = const_value.proposal_num_channel_max_ub

    def proposal_merge_gm_start(self, tensor_dst_gm, tensor_src_gm, batch_index_dst_gm, batch_index_src_gm,
                                pro_index_start_dst_gm, pro_index_start_src_gm, pro_num_dst_gm, pro_num_list_src_gm):
        """
        start merge proposal
        Parameters
        ----------
        tensor_dst_gm:
            TVM gm tensor. Dtype: float16 or float32. Shape: (batch, pro_num, pro_element)
        tensor_src_gm:
            TVM gm tensor. The data type and shape are the same as tensor_dst_gm
        batch_index_dst_gm:
            Int/Scalar(int). Batch index of tensor_dst_gm.
        batch_index_src_gm:
            Int/Scalar(int). Batch index of tensor_src_gm.
        pro_index_start_dst_gm:
            Int/Scalar(int). proposal index of tensor_dst_gm.
        pro_index_start_src_gm:
            Int/Scalar(int). proposal index of tensor_src_gm.
        pro_num_dst_gm:
            int. Number of proposal to sort.
        pro_num_list_src_gm:
            List[int]. Number of proposal already sorted. 1 <= len <= channel_num_max_sort
        """
        pro_num_dst_gm = min(pro_num_dst_gm, sum(pro_num_list_src_gm))
        with self._tik_inst.new_stmt_scope():
            if len(pro_num_list_src_gm) == 1:
                self._src_to_dst(tensor_dst_gm, tensor_src_gm, batch_index_dst_gm, batch_index_src_gm,
                                 pro_index_start_dst_gm, pro_index_start_src_gm, pro_num_dst_gm)
            else:
                self._merge_proposal_channels_gm(tensor_dst_gm, tensor_src_gm, batch_index_dst_gm, batch_index_src_gm,
                                                 pro_index_start_dst_gm, pro_index_start_src_gm, pro_num_dst_gm,
                                                 pro_num_list_src_gm)

    def _src_to_dst(self, tensor_dst_gm, tensor_src_gm, batch_index_dst_gm, batch_index_src_gm,
                    pro_index_start_dst_gm, pro_index_start_src_gm, pro_num_dst_gm):
        """
        move data form tensor_src_gm to tensor_dst_gm
        """
        pro_num_each_loop = self._pro_num_max_ub * 2
        tensor_shape_ub = (pro_num_each_loop, self._ele_num_pro)
        tensor_name_ub = "pro_ub_{}".format(self._tensor_name)
        pro_ub = self._tik_inst.Tensor(self._score_type, tensor_shape_ub, self._tik.scope_ubuf, tensor_name_ub)

        loop_times, pro_num_last_loop = get_loop_info(pro_num_dst_gm, pro_num_each_loop)
        each_loop_block_num = pro_num_each_loop // self._pro_num_block
        last_loop_block_num = pro_num_last_loop // self._pro_num_block
        with self._tik_inst.for_range(0, loop_times) as loop_index:
            pro_index_loop_src_gm = pro_index_start_src_gm + loop_index * pro_num_each_loop
            pro_index_loop_dst_gm = pro_index_start_dst_gm + loop_index * pro_num_each_loop
            with self._tik_inst.if_scope(loop_index != loop_times - 1):
                self._tik_inst.data_move(pro_ub, tensor_src_gm[batch_index_src_gm, pro_index_loop_src_gm, 0],
                                         0, 1, each_loop_block_num, 0, 0)
                self._tik_inst.data_move(tensor_dst_gm[batch_index_dst_gm, pro_index_loop_dst_gm, 0], pro_ub,
                                         0, 1, each_loop_block_num, 0, 0)
            with self._tik_inst.else_scope():
                self._tik_inst.data_move(pro_ub, tensor_src_gm[batch_index_src_gm, pro_index_loop_src_gm, 0],
                                         0, 1, last_loop_block_num, 0, 0)
                self._tik_inst.data_move(tensor_dst_gm[batch_index_dst_gm, pro_index_loop_dst_gm, 0], pro_ub,
                                         0, 1, last_loop_block_num, 0, 0)

    def _merge_proposal_channels_gm(self, tensor_dst_gm, tensor_src_gm,
                                    batch_index_dst_gm, batch_index_src_gm,
                                    pro_index_start_dst_gm, pro_index_start_src_gm,
                                    pro_num_dst_gm, pro_num_list_src_gm):
        """
        Merge sort tensor_src_gm according to the input information
        """
        c_num = len(pro_num_list_src_gm)
        pro_num_c_ub = min(self._pro_num_c_max_ub, self._pro_num_max_ub // c_num)
        pro_num_c_ub = get_align_num(pro_num_c_ub, self._pro_num_repeat, ceil=False)
        tensor_shape_ub = (c_num, pro_num_c_ub, self._ele_num_pro)
        tensor_name_ub_0 = "pro_ub_0_{}".format(self._tensor_name)
        tensor_name_ub_1 = "pro_ub_1_{}".format(self._tensor_name)
        tensor_src_ub = self._tik_inst.Tensor(self._score_type, tensor_shape_ub, self._tik.scope_ubuf, tensor_name_ub_0)
        tensor_dst_ub = self._tik_inst.Tensor(self._score_type, tensor_shape_ub, self._tik.scope_ubuf, tensor_name_ub_1)

        # Loop instead of recursion
        loop_times = pro_num_dst_gm // pro_num_c_ub + c_num

        src_ub_tuple = tuple(tensor_src_ub[list_idx, 0, 0] for list_idx in range(c_num))
        scalar_all, scalar_list_all = self._init_scalar_all(tensor_src_ub, pro_num_c_ub, pro_index_start_src_gm,
                                                            pro_num_dst_gm, pro_num_list_src_gm)
        with self._tik_inst.for_range(0, loop_times):
            with self._tik_inst.if_scope(scalar_all.get("selected_num_sum") < pro_num_dst_gm):
                self._move_gm_to_ub(tensor_src_gm, src_ub_tuple, batch_index_src_gm, scalar_all, scalar_list_all)

                self._tik_inst.vmrgsort(
                    tensor_dst_ub, src_ub_tuple, scalar_list_all.get("pro_num_src_ub"),
                    True, 1, scalar_list_all.get("pro_num_dst_ub"))
                self._update_scalar(scalar_all, scalar_list_all)
                self._move_ub_to_gm(tensor_dst_gm, tensor_dst_ub, batch_index_dst_gm,
                                    pro_index_start_dst_gm, scalar_all)

    def _init_scalar_all(self, tensor_src_ub, pro_num_c_ub, pro_index_start_src_gm, pro_num_dst_gm,
                         pro_num_list_src_gm):
        """
        init scalar
        """
        scalar_all = {
            "selected_num_sum": self._tik_inst.Scalar(self._int_type, init_value=0),
            "selected_num": self._tik_inst.Scalar(self._int_type, init_value=0),
            "rem_num_sum": self._tik_inst.Scalar(self._int_type, init_value=pro_num_dst_gm),
            "pro_num_c_ub": self._tik_inst.Scalar(self._int_type, init_value=pro_num_c_ub),
        }
        c_num = tensor_src_ub.shape[0]

        scalar_tuple_all = {
            "pro_idx_src_gm": tuple(self._tik_inst.Scalar(self._int_type, init_value=0) for _ in range(c_num)),
            "rem_num_src_gm": tuple(self._tik_inst.Scalar(self._int_type, init_value=pro_num_list_src_gm[c_index])
                                    for c_index in range(c_num)),
            "pro_num_src_ub": tuple(self._tik_inst.Scalar(self._int_type, init_value=0) for _ in range(c_num)),
            "pro_num_dst_ub": tuple(self._tik_inst.Scalar(self._int_type, init_value=0) for _ in range(c_num)),
        }
        pro_index_c_src_gm = pro_index_start_src_gm
        for c_index, pro_num_src_gm in enumerate(pro_num_list_src_gm):
            scalar_tuple_all.get("pro_idx_src_gm")[c_index].set_as(pro_index_c_src_gm)
            pro_index_c_src_gm += pro_num_src_gm
        return scalar_all, scalar_tuple_all

    def _move_gm_to_ub(self, tensor_src_gm, src_ub_tuple, batch_index_src_gm, scalar_all, scalar_list_all):
        """
        move proposal form gm to ub
        """
        c_num = len(src_ub_tuple)

        pro_idx_src_gm_ss = scalar_list_all.get("pro_idx_src_gm")
        rem_num_src_gm_ss = scalar_list_all.get("rem_num_src_gm")
        pro_num_src_ub_ss = scalar_list_all.get("pro_num_src_ub")

        for c_idx in range(c_num):
            tensor_ub = src_ub_tuple[c_idx]
            rem_num_src_gm_s = rem_num_src_gm_ss[c_idx]
            pro_num_src_ub_s = pro_num_src_ub_ss[c_idx]
            with self._tik_inst.if_scope(rem_num_src_gm_s > 0):
                pro_idx_src_gm_s = pro_idx_src_gm_ss[c_idx]
                pro_num_c_ub = scalar_all.get("pro_num_c_ub")
                self._tik_inst.scalar_min(pro_num_src_ub_s, rem_num_src_gm_s, pro_num_c_ub)
                block_num_move_in = ceil_div(pro_num_src_ub_s, self._pro_num_block)
                self._tik_inst.data_move(tensor_ub, tensor_src_gm[batch_index_src_gm, pro_idx_src_gm_s, 0],
                                         0, 1, block_num_move_in, 0, 0)
            with self._tik_inst.else_scope():
                pro_num_src_ub_s.set_as(pro_num_c_ub)
                vector_dup_mask = self._pro_num_repeat * self._ele_num_pro
                self._tik_inst.vector_dup(vector_dup_mask, tensor_ub, self._neg_inf, 1, 1, 8)

    def _update_scalar(self, scalar_all, scalar_list_all):
        """
        updata scalar
        """
        pro_idx_src_gm_ss = scalar_list_all.get("pro_idx_src_gm")
        rem_num_src_gm_ss = scalar_list_all.get("rem_num_src_gm")
        pro_num_dst_ub_ss = scalar_list_all.get("pro_num_dst_ub")
        selected_num_s = scalar_all.get("selected_num")
        selected_num_s.set_as(0)
        c_num = len(pro_idx_src_gm_ss)
        for c_index in range(c_num):
            pro_idx_src_gm_s = pro_idx_src_gm_ss[c_index]
            rem_num_src_gm_s = rem_num_src_gm_ss[c_index]
            pro_num_dst_ub_s = pro_num_dst_ub_ss[c_index]
            selected_num_s.set_as(selected_num_s + pro_num_dst_ub_s)
            pro_idx_src_gm_s.set_as(pro_idx_src_gm_s + pro_num_dst_ub_s)
            rem_num_src_gm_s.set_as(rem_num_src_gm_s - pro_num_dst_ub_s)

    def _move_ub_to_gm(self, tensor_dst_gm, tensor_dst_ub, batch_index_dst_gm,
                       pro_index_start_dst_gm, scalar_all):
        """
        move proposal form ub to gm
        """
        selected_num_sum_s = scalar_all.get("selected_num_sum")
        selected_num_s = scalar_all.get("selected_num")
        rem_num_sum_s = scalar_all.get("rem_num_sum")
        self._tik_inst.scalar_min(selected_num_s, selected_num_s, rem_num_sum_s)

        with self._tik_inst.if_scope(selected_num_s > 0):
            pro_index_dst_gm = pro_index_start_dst_gm + selected_num_sum_s
            block_num_move_out = ceil_div(selected_num_s, self._pro_num_block)
            self._tik_inst.data_move(
                tensor_dst_gm[batch_index_dst_gm, pro_index_dst_gm, 0], tensor_dst_ub, 0, 1, block_num_move_out, 0, 0
            )
            selected_num_sum_s.set_as(selected_num_sum_s + selected_num_s)
            rem_num_sum_s.set_as(rem_num_sum_s - selected_num_s)


class MergeSort(object):
    """
    merge sort
    """

    def __init__(self, tik, tik_inst, const_value):
        self._tik = tik
        self._tik_inst = tik_inst
        self._tensor_name = "merge_sort"
        self._score_type = const_value.score_type

        self._const_value = const_value

        self._pro_num_align = const_value.proposal_num_repeat
        self._pro_num_max_ub = const_value.proposal_num_max_ub
        self._ele_num_pro = const_value.element_num_proposal
        self._pro_num_block = const_value.proposal_num_block
        self._c_num_max_sort = const_value.channel_num_max_sort

        self.proposal_sort_ub = ProposalSortUB(self._tik, self._tik_inst, self._const_value).proposal_sort_ub_start
        self.proposal_merge_ub = ProposalMergeUB(self._tik, self._tik_inst, self._const_value).proposal_merge_ub_start
        self.proposal_merge_gm = ProposalMergeGM(self._tik, self._tik_inst, self._const_value).proposal_merge_gm_start

    def merge_sort_start(self, pro_dst_gm, pro_src_gm, batch_index_dst_gm, batch_index_src_gm,
                         score_num_dst, score_num_src, get_score_index_tensor, fun_args):
        """

        Parameters
        ----------
        pro_dst_gm:
            TVM gm tensor. Dtype: float16 or float32. Shape: (batch, pro_num, pro_element)
        pro_src_gm:
            TVM gm tensor. The data type and shape are the same as tensor_dst_gm
        batch_index_dst_gm:
            Int/Scalar(int). Batch index of tensor_dst_gm.
        batch_index_src_gm:
            Int/Scalar(int). Batch index of tensor_src_gm.
        score_num_dst:
            Int. Input score num.
        score_num_src:
            Int. result score num.
        get_score_index_tensor:
            Function. Function to get score tensor and index tensor.
        fun_args
            Dict. Input parameters of get_score_index_tensor
        """
        pro_num_dst = get_align_num(score_num_dst, self._pro_num_align)
        pro_num_src = get_align_num(score_num_src, self._pro_num_align)
        pro_num_dst = min(pro_num_dst, pro_num_src)

        score_num_loop_src_ub = self._pro_num_max_ub
        pro_num_loop_src_ub = self._pro_num_max_ub
        pro_num_loop_dst_ub = min(pro_num_loop_src_ub, pro_num_dst)

        loop_times, score_num_last_loop_src_ub = get_loop_info(score_num_src, score_num_loop_src_ub)
        pro_num_last_loop_src_ub = get_align_num(score_num_last_loop_src_ub, self._pro_num_align)
        pro_num_last_loop_dst_ub = min(pro_num_last_loop_src_ub, pro_num_dst)
        merge_loop_info_all = self._init_merge_loop_info(pro_num_dst, loop_times,
                                                         pro_num_loop_dst_ub, pro_num_last_loop_dst_ub)
        if len(merge_loop_info_all) % 2 == 1:
            pro_dst_gm, pro_src_gm = pro_src_gm, pro_dst_gm
            batch_index_dst_gm, batch_index_src_gm = batch_index_src_gm, batch_index_dst_gm
        self._proposal_sort_start(pro_dst_gm, batch_index_dst_gm, get_score_index_tensor, fun_args, loop_times,
                                  pro_num_loop_dst_ub, pro_num_last_loop_dst_ub,
                                  score_num_loop_src_ub, score_num_last_loop_src_ub)

        pro_dst_gm, pro_src_gm = pro_src_gm, pro_dst_gm
        batch_index_dst_gm, batch_index_src_gm = batch_index_src_gm, batch_index_dst_gm
        self._proposal_merge_start(pro_dst_gm, pro_src_gm,
                                   batch_index_dst_gm, batch_index_src_gm,
                                   merge_loop_info_all)

    def _init_merge_loop_info(self, pro_num_dst, merge_channel, sorted_num, sorted_num_last):
        """
        Calculate the information for each recursion. Loop instead of recursion
        """
        merge_loop_info_all = []
        while merge_channel > 1:
            loop_times_merge, merge_channel_last_times = get_loop_info(merge_channel, self._c_num_max_sort)
            pro_num_list_src_gm = (sorted_num,) * self._c_num_max_sort
            pro_num_list_last_src_gm = (sorted_num,) * (merge_channel_last_times - 1) + (sorted_num_last,)

            sorted_num_next_loop = min(sum(pro_num_list_src_gm), pro_num_dst)
            sorted_num_last_next_loop = min(sum(pro_num_list_last_src_gm), pro_num_dst)
            merge_loop_info_all.append([loop_times_merge, sorted_num,
                                        sorted_num_next_loop, sorted_num_last_next_loop,
                                        pro_num_list_src_gm, pro_num_list_last_src_gm])
            merge_channel = loop_times_merge
            sorted_num = sorted_num_next_loop
            sorted_num_last = sorted_num_last_next_loop
        return merge_loop_info_all

    def _proposal_sort_start(self, pro_dst_gm, batch_index_dst_gm, get_score_index_tensor, fun_args, loop_times,
                             pro_num_loop_dst_ub, pro_num_last_loop_dst_ub,
                             score_num_loop_src_ub, score_num_last_loop_src_ub):
        """
        proposal sort im ub
        """
        with self._tik_inst.for_range(0, loop_times) as loop_index:
            pro_index_src_ub = loop_index * score_num_loop_src_ub
            pro_index_dst_ub = loop_index * pro_num_loop_dst_ub
            with self._tik_inst.if_scope(loop_index != loop_times - 1):
                self._proposal_sort_loop_start(pro_dst_gm, batch_index_dst_gm,
                                               pro_index_dst_ub, pro_index_src_ub, pro_num_loop_dst_ub,
                                               score_num_loop_src_ub, get_score_index_tensor, fun_args)
            with self._tik_inst.else_scope():
                self._proposal_sort_loop_start(pro_dst_gm, batch_index_dst_gm,
                                               pro_index_dst_ub, pro_index_src_ub, pro_num_last_loop_dst_ub,
                                               score_num_last_loop_src_ub, get_score_index_tensor, fun_args)

    def _proposal_sort_loop_start(self, pro_dst_gm,
                                  batch_index_dst_gm,
                                  pro_index_dst_ub, pro_index_src_ub,
                                  pro_num_dst, score_num_src,
                                  get_score_index_tensor, fun_args):
        """
        sort proposal loop in ub
        """
        fun_args["score_num_loop"] = score_num_src
        fun_args["score_index_loop"] = pro_index_src_ub

        pro_num_src = get_align_num(score_num_src, self._pro_num_align)
        fun_args["proposal_num_loop"] = pro_num_src

        tensor_name_0_ub = "tensor_name_0_ub_{}".format(self._tensor_name)
        tensor_name_1_ub = "tensor_name_1_ub_{}".format(self._tensor_name)

        tensor_shape_ub = (pro_num_src, self._ele_num_pro)
        proposal_ub_0 = self._tik_inst.Tensor(self._score_type, tensor_shape_ub, self._tik.scope_ubuf, tensor_name_0_ub)
        self.proposal_sort_ub(proposal_ub_0, pro_num_src, get_score_index_tensor, fun_args)
        proposal_ub_1 = self._tik_inst.Tensor(self._score_type, tensor_shape_ub, self._tik.scope_ubuf, tensor_name_1_ub)

        proposal_ub_0, proposal_ub_1 = self.proposal_merge_ub(proposal_ub_0, proposal_ub_1, pro_num_src,
                                                              self._pro_num_align)

        block_move_out = pro_num_dst // self._pro_num_block
        self._tik_inst.data_move(pro_dst_gm[batch_index_dst_gm, pro_index_dst_ub, 0],
                                 proposal_ub_0, 0, 1, block_move_out, 0, 0)

    def _proposal_merge_start(self, pro_dst_gm, pro_src_gm,
                              batch_index_dst_gm, batch_index_src_gm,
                              merge_loop_info_all):
        """
        proposal merge in gm
        """
        for loop_info in merge_loop_info_all:
            (merge_times, sorted_num, sorted_num_next_loop, sorted_num_last_next_loop,
             pro_num_list_src_gm, pro_num_list_last_src_gm) = loop_info
            with self._tik_inst.for_range(0, merge_times) as merge_index:
                pro_index_dst_gm = merge_index * sorted_num_next_loop
                pro_index_src_gm = merge_index * self._c_num_max_sort * sorted_num
                with self._tik_inst.if_scope(merge_index != merge_times - 1):
                    self.proposal_merge_gm(pro_dst_gm, pro_src_gm, batch_index_dst_gm, batch_index_src_gm,
                                           pro_index_dst_gm, pro_index_src_gm, sorted_num_next_loop,
                                           pro_num_list_src_gm)
                with self._tik_inst.else_scope():
                    self.proposal_merge_gm(pro_dst_gm, pro_src_gm, batch_index_dst_gm, batch_index_src_gm,
                                           pro_index_dst_gm, pro_index_src_gm,
                                           sorted_num_last_next_loop, pro_num_list_last_src_gm)
            pro_dst_gm, pro_src_gm = pro_src_gm, pro_dst_gm
            batch_index_dst_gm, batch_index_src_gm = batch_index_src_gm, batch_index_dst_gm
