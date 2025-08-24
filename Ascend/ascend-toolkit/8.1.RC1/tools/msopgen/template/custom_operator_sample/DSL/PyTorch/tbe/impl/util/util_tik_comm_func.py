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
util_tik_comm_func
"""
import functools
from tbe import tik
from .. import common_util
from .. import constant_util
from . import util_common
from .platform_adapter import tik as tik_adapter
from .platform_adapter import tbe_platform as tbe_platform_adapter


# define a scalar, value = 2**(-126), minimun num of float32 2**(-126)
SCALAR_MIN_FP32 = 2**(-126)
# define a scalar, value = 2**(50)
SCALAR_MUL_FP32 = 2**50
# define a scalar, value = 2**(26)
SCALAR_MUL2_FP32 = 2**26
# repeat max num
MAX_REPEAT_NUM = 255
# max int64
MAX_INT64 = 2 ** 64 - 1


# pylint: disable=too-many-instance-attributes
class OpBase:
    """
    Class: class that OpBase
    """
    def __init__(self):
        self.tik_instance = tik_adapter.Tik()
        self.core_nums = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.CORE_NUM)
        self.ub_size_bytes = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.UB_SIZE)
        self.input_gm_list = []
        self.output_gm_list = []
        self.tiling_gm = None
        self.unknown_max_shape = (MAX_INT64,)
        self.kernel_name = None
        self.opt_config = {"out_of_bound_sync_check": True,
                           "enable_const_fold": True}
        self.tiling_key = None
        self.mode_compute = dict()

    def op_init_gm(self, input_dict_list, output_dict_list, tiling_info=None, is_fused_1d=False):
        """
        op_init_gm

        Parameters
        ----------
        input_dict_list: list
            a list of input dict
        output_dict_list: list
            a list of output dict
        tiling_info: dict
            include key shape and dtype
        is_fused_1d: bool
            whether fused shape to id when apply gm

        Returns
        ------
        None
        """
        def gen_gm_scope(gm_dict, gm_type="input"):
            gm_dtype = gm_dict.get("dtype")
            if util_common.is_unknown([gm_dict]):
                gm_shape = self.unknown_max_shape
            else:
                gm_shape = gm_dict.get("shape")
                if is_fused_1d:
                    total_num = functools.reduce(lambda x, y: x*y, gm_shape)
                    gm_shape = [total_num]

            # get gm name from dict
            if "param_name" not in gm_dict.keys():
                gm_name = gm_type + "_gm_" + str(i)
            else:
                gm_name = gm_dict.get("param_name")

            # get is_atomic_add from dict
            is_atomic_add = False
            if "is_atomic_add" in gm_dict.keys():
                is_atomic_add = gm_dict.get("is_atomic_add")
            gen_gm = self.tik_instance.Tensor(gm_dtype, gm_shape,
                                              name=gm_name, scope=tik.scope_gm,
                                              is_atomic_add=is_atomic_add)
            return gen_gm

        for i, input_dict in enumerate(input_dict_list):
            if input_dict is None:
                continue
            input_gm = gen_gm_scope(input_dict)
            self.input_gm_list.append(input_gm)

        for i, output_dict in enumerate(output_dict_list):
            if output_dict is None:
                continue
            output_gm = gen_gm_scope(output_dict, gm_type="output")
            self.output_gm_list.append(output_gm)

        if tiling_info is not None:
            tiling_dtype = tiling_info.get("dtype")
            tiling_shape = tiling_info.get("shape")
            self.tiling_gm = self.tik_instance.Tensor(tiling_dtype, tiling_shape,
                                                      name="tiling_gm", scope=tik.scope_gm)

    def op_build_cce(self):
        """
        op_build_cce
        """
        flowtable = [self.tiling_gm] if self.tiling_gm is not None else None
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_gm_list,
                                   flowtable=flowtable,
                                   outputs=self.output_gm_list,
                                   config=self.opt_config)

    def regist_compute(self, tiling_key, tiling_func, *var_tuple, **var_key_value):
        """
        regist_compute

        Parameters
        ----------
        tiling_key: int
            tiling_key info
        tiling_func: class
            tiling_key function

        Returns
        ------
        None
        """
        compute_classify = "default"
        if "compute_classify" in var_key_value.keys():
            compute_classify = var_key_value[compute_classify]
        if compute_classify not in self.mode_compute.keys():
            self.mode_compute[compute_classify] = dict()
        self.mode_compute[compute_classify][tiling_key] = [tiling_func, var_tuple, var_key_value]

    def run_compute(self, tiling_key, compute_classify=None):
        """
        run_compute
        """
        for classify_key, compute_info in self.mode_compute.items():
            if compute_classify is not None and classify_key != compute_classify:
                continue
            for key, key_func in compute_info.items():
                with self.tik_instance.if_scope(tiling_key == key):
                    with self.tik_instance.new_stmt_scope():
                        key_func[0](*key_func[1], **key_func[2])

    # pylint: disable=unnecessary-pass
    def tiling_args(self):
        """
        read tiling args, should over write
        """
        pass

    # pylint: disable=unnecessary-pass
    def core_scedule_args(self, core_idx):
        """
        calcu core para base tiling
        if need should over write, else do nothing
        """
        pass

    def op_run_compute(self):
        """
        op_run_base run all the regist_compute base tiling_key
        if can not run all regist_compute at the same time, need should over write
        """
        with self.tik_instance.for_range(0, self.core_nums, block_num=self.core_nums) as core_index:
            self.tiling_args()
            self.core_scedule_args(core_index)
            self.run_compute(self.tiling_key)


def ub_offset(input_ub):
    """
    get ub offset
    when ub.shape is 1D tensor offset = 0
    when ub.shape is not 1D tensor change offset = 1D
    ex:
       ub.shape = [2,2,2]
       ub1 = ub[1,:,:]
       ub_offset(ub1) = 2*2 = 4 for ub
    """
    ub_shape = input_ub.shape
    if len(ub_shape) in (0, 1):
        return 0

    return input_ub.offset


# pylint: disable=too-many-branches,too-many-statements,too-many-locals
# pylint: disable=too-many-arguments
def tik_func_vector(tik_instance, _ub, value, dup_len):
    """
    tik_func_vector

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    _ub: ub
        vector ub
    value: value
        vector value
    dup_len: int
        vector data len

    Returns
    -------
    None
    """
    do_dtype = _ub.dtype
    byte_num_one = common_util.get_data_size(do_dtype)
    block_num = constant_util.BLOCK_SIZE // byte_num_one
    vector_num = block_num*constant_util.REPEAT_STRIDE_EIGHT
    repeat = dup_len // vector_num
    repeat_tail = dup_len % vector_num
    offset = 0
    while repeat > MAX_REPEAT_NUM:
        tik_instance.vector_dup(vector_num, _ub[offset], value, MAX_REPEAT_NUM, 1, 8)
        repeat = repeat - MAX_REPEAT_NUM
        offset = offset + vector_num*MAX_REPEAT_NUM
    if repeat > 0:
        tik_instance.vector_dup(vector_num, _ub[offset], value, repeat, 1, 8)
        offset = offset + vector_num*repeat
    if repeat_tail > 0:
        tik_instance.vector_dup(repeat_tail, _ub[offset], value, 1, 1, 8)


def tik_fuc_vrec_newton(tik_instance, vrec_ub, origin_ub, do_len, newton_iteration=2, block_num=16,
                        vrec_blk=1, origin_blk=1, vrec_rep=8, origin_rep=8):
    """
    only do newton for vrec result

    Parameters
    ----------
    tik_instance: class
        tik_instance
    vrec_ub: ub
        the result of vrec
    origin_ub: ub
        the origin input for vrec
    do_len: int
        vrec num
    newton_iteration: int
        do newton iteration, default: 2
    block_num: int
        num in one block, default: 16
    vrec_blk: int
        the block stride of vrec_ub, default: 1
    origin_blk: int
        the block stride of origin_ub, default: 1
    vrec_rep: int
        the repeat stride of vrec_ub, default: 8
    origin_rep: int
        the repeat stride of origin_ub, default: 8

    Returns
    -------
    None
    """
    with tik_instance.new_stmt_scope():
        vrec_newton_1 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_1", scope=tik.scope_ubuf)
        vrec_newton_2 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_2", scope=tik.scope_ubuf)

        def _one_newton():
            tik_func_vcomple(tik_instance, "vmul", vrec_newton_1, vrec_ub, origin_ub, do_len,
                             src0_blk=vrec_blk, src0_rep=vrec_rep,
                             src1_blk=origin_blk, src1_rep=origin_rep)
            tik_func_vmuls(tik_instance, vrec_newton_2, vrec_newton_1, -1, do_len)
            tik_func_vadds(tik_instance, vrec_newton_1, vrec_newton_2, 2, do_len)
            tik_func_vcomple(tik_instance, "vmul", vrec_ub, vrec_newton_1, vrec_ub, do_len,
                             dst_blk=vrec_blk, dst_rep=vrec_rep,
                             src1_blk=vrec_blk, src1_rep=vrec_rep)

        for _ in range(newton_iteration):
            _one_newton()


def tik_func_vrec(tik_instance, dst_ub, src_ub, do_len, newton_iteration=2,
                  dst_blk=1, src_blk=1, dst_rep=8, src_rep=8):
    """
    do vrce for input src_ub
    and will do newton_iteration with para newton_iteration

    Parameters
    ----------
    tik_instance: class
        tik_instance
    dst_ub: ub
        the result of vrec
    src_ub: ub
        the origin input for vrec
    do_len: int
        vrec num
    newton_iteration: int
        do newton iteration, default: 2
    dst_blk: int
        the block stride of dst_ub, default: 1
    src_blk: int
        the block stride of src_ub, default: 1
    dst_rep: int
        the repeat stride of dst_ub, default: 8
    src_rep: int
        the repeat stride of src_ub, default: 8

    Returns
    -------
    None
    """
    vmuls_type = dst_ub.dtype
    byte_num_one = common_util.get_data_size(vmuls_type)
    block_num = constant_util.BLOCK_SIZE // byte_num_one
    vector_num = block_num * constant_util.REPEAT_STRIDE_EIGHT
    repeat = do_len // vector_num
    repeat_tail = do_len % vector_num
    dst_offset = ub_offset(dst_ub)
    src_offset = ub_offset(src_ub)
    tik_api = tik_instance.vrec
    while repeat > MAX_REPEAT_NUM:
        tik_api(vector_num, dst_ub[dst_offset], src_ub[src_offset],
                MAX_REPEAT_NUM, dst_blk, src_blk, dst_rep, src_rep)
        tik_fuc_vrec_newton(tik_instance, dst_ub[dst_offset], src_ub[src_offset], MAX_REPEAT_NUM * vector_num,
                            newton_iteration=newton_iteration, block_num=block_num,
                            vrec_blk=dst_blk, origin_blk=src_blk, vrec_rep=dst_rep, origin_rep=src_rep)
        repeat = repeat - MAX_REPEAT_NUM
        dst_offset = dst_offset + block_num * MAX_REPEAT_NUM * dst_rep
        src_offset = src_offset + block_num * MAX_REPEAT_NUM * src_rep
    if repeat > 0:
        tik_api(vector_num, dst_ub[dst_offset], src_ub[src_offset],
                repeat, dst_blk, src_blk, dst_rep, src_rep)
        tik_fuc_vrec_newton(tik_instance, dst_ub[dst_offset], src_ub[src_offset], repeat * vector_num,
                            newton_iteration=newton_iteration, block_num=block_num,
                            vrec_blk=dst_blk, origin_blk=src_blk, vrec_rep=dst_rep, origin_rep=src_rep)
        dst_offset = dst_offset + block_num * repeat * dst_rep
        src_offset = src_offset + block_num * repeat * src_rep
    if repeat_tail > 0:
        tik_api(repeat_tail, dst_ub[dst_offset], src_ub[src_offset],
                1, dst_blk, src_blk, dst_rep, src_rep)
        tik_fuc_vrec_newton(tik_instance, dst_ub[dst_offset], src_ub[src_offset], repeat_tail,
                            newton_iteration=newton_iteration, block_num=block_num,
                            vrec_blk=dst_blk, origin_blk=src_blk, vrec_rep=dst_rep, origin_rep=src_rep)


def tik_func_vcomple(tik_instance, function, out_dst, src0, src1, copy_num,
                     dst_blk=1, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8,
                     src1_rep=8):
    """
    tik_func_vcomple
    """
    do_dtype = out_dst.dtype
    byte_num_one = common_util.get_data_size(do_dtype)
    block_num = constant_util.BLOCK_SIZE // byte_num_one
    vector_num = block_num*constant_util.REPEAT_STRIDE_EIGHT
    repeat_time = copy_num // vector_num
    repeat_tail = copy_num % vector_num
    tik_fun = None
    ori_offset_dst = ub_offset(out_dst)
    ori_offset_src0 = ub_offset(src0)
    ori_offset_src1 = ub_offset(src1)
    if function == "vmin":
        tik_fun = tik_instance.vmin
    elif function == "vmax":
        tik_fun = tik_instance.vmax
    elif function == "vmul":
        tik_fun = tik_instance.vmul
    elif function == "vadd":
        tik_fun = tik_instance.vadd
    elif function == "vsub":
        tik_fun = tik_instance.vsub
    elif function == "vdiv":
        tik_fun = tik_instance.vdiv

    if function != "vdiv" or tbe_platform_adapter.api_check_support("tik.vdiv"):
        while repeat_time > MAX_REPEAT_NUM:
            tik_fun(vector_num,
                    out_dst[ori_offset_dst],
                    src0[ori_offset_src0],
                    src1[ori_offset_src1],
                    255,
                    dst_blk, src0_blk, src1_blk,
                    dst_rep, src0_rep, src1_rep)
            repeat_time = repeat_time - MAX_REPEAT_NUM
            ori_offset_dst = ori_offset_dst + MAX_REPEAT_NUM * block_num * dst_rep
            ori_offset_src0 = ori_offset_src0 + MAX_REPEAT_NUM * block_num * src0_rep
            ori_offset_src1 = ori_offset_src1 + MAX_REPEAT_NUM * block_num * src1_rep

        if repeat_time > 0:
            tik_fun(vector_num,
                    out_dst[ori_offset_dst],
                    src0[ori_offset_src0],
                    src1[ori_offset_src1],
                    repeat_time,
                    dst_blk, src0_blk, src1_blk,
                    dst_rep, src0_rep, src1_rep)
            ori_offset_dst = ori_offset_dst + repeat_time * block_num * dst_rep
            ori_offset_src0 = ori_offset_src0 + repeat_time * block_num * src0_rep
            ori_offset_src1 = ori_offset_src1 + repeat_time * block_num * src1_rep

        if repeat_tail > 0:
            tik_fun(repeat_tail,
                    out_dst[ori_offset_dst],
                    src0[ori_offset_src0],
                    src1[ori_offset_src1],
                    1,
                    dst_blk, src0_blk, src1_blk,
                    dst_rep, src0_rep, src1_rep)
    else:
        # div func and do not support vdiv, will use src0 * vrec(src1) to do div
        with tik_instance.new_stmt_scope():
            vrec_data = tik_instance.Tensor(src0.dtype, (((copy_num + block_num - 1) // block_num) * block_num,),
                                            name="vrec_data", scope=tik.scope_ubuf)
            tik_func_vrec(tik_instance, vrec_data, src1, copy_num,
                          src_blk=src1_blk, src_rep=src1_rep)
            tik_func_vcomple(tik_instance, "vmul", out_dst, src0, vrec_data, copy_num,
                             dst_blk=dst_blk, src0_blk=src0_blk,
                             dst_rep=dst_rep, src0_rep=src0_rep)


def _tik_func_single_input_with_scalar(tik_api, dst_ub, src_ub, value, do_len,
                                       dst_blk=1, src_blk=1, dst_rep=8, src_rep=8):
    """
    _tik_func_single
    """
    vmuls_type = dst_ub.dtype
    byte_num_one = common_util.get_data_size(vmuls_type)
    block_num = constant_util.BLOCK_SIZE // byte_num_one
    vector_num = block_num * constant_util.REPEAT_STRIDE_EIGHT
    repeat = do_len // vector_num
    repeat_tail = do_len % vector_num
    dst_offset = ub_offset(dst_ub)
    src_offset = ub_offset(src_ub)
    while repeat > MAX_REPEAT_NUM:
        tik_api(vector_num, dst_ub[dst_offset], src_ub[src_offset], value,
                MAX_REPEAT_NUM, dst_blk, src_blk, dst_rep, src_rep)
        repeat = repeat - MAX_REPEAT_NUM
        dst_offset = dst_offset + block_num * MAX_REPEAT_NUM * dst_rep
        src_offset = src_offset + block_num * MAX_REPEAT_NUM * src_rep
    if repeat > 0:
        tik_api(vector_num, dst_ub[dst_offset], src_ub[src_offset], value,
                repeat, dst_blk, src_blk, dst_rep, src_rep)
        dst_offset = dst_offset + block_num * repeat * dst_rep
        src_offset = src_offset + block_num * repeat * src_rep
    if repeat_tail > 0:
        tik_api(repeat_tail, dst_ub[dst_offset], src_ub[src_offset], value,
                1, dst_blk, src_blk, dst_rep, src_rep)


def tik_func_vmuls(tik_instance, dst_ub, src_ub, value, do_len,
                   dst_blk=1, src_blk=1, dst_rep=8, src_rep=8):
    """
    tik_func_vadds
    """
    _tik_func_single_input_with_scalar(tik_instance.vmuls, dst_ub, src_ub, value, do_len,
                                       dst_blk, src_blk, dst_rep, src_rep)


def tik_func_vadds(tik_instance, dst_ub, src_ub, value, do_len,
                   dst_blk=1, src_blk=1, dst_rep=8, src_rep=8):
    """
    tik_func_vadds
    """
    _tik_func_single_input_with_scalar(tik_instance.vadds, dst_ub, src_ub, value, do_len,
                                       dst_blk, src_blk, dst_rep, src_rep)


def tik_func_vconcat(tik_instance, proposals_ub, _ub, trans_repeat, mode):
    """
    tik_func_vconcat
    """
    tik_instance.vconcat(proposals_ub, _ub, trans_repeat, mode)


def tik_func_vextract(tik_instance, proposals_ub, _ub, trans_repeat, mode):
    """
    tik_func_vextract
    """
    tik_instance.vextract(_ub, proposals_ub, trans_repeat, mode)


def tik_func_vconv(tik_instance, dst_ub, src_ub, do_len, mode="", mini_mid_ub=None):
    """
    tik_func_vconv
    """
    src_dtype = src_ub.dtype
    dst_dtype = dst_ub.dtype

    def do_vconv(dst_repeat_stride, src_repeat_stride, deq_scale=None, block_num=64):
        ori_dst_offset = ub_offset(dst_ub)
        ori_src_offset = ub_offset(src_ub)
        repeat = do_len // block_num
        repeat_tail = do_len % block_num
        while repeat > MAX_REPEAT_NUM:
            tik_instance.vconv(block_num, mode, dst_ub[ori_dst_offset], src_ub[ori_src_offset],
                               MAX_REPEAT_NUM, 1, 1, dst_repeat_stride, src_repeat_stride, deqscale=deq_scale)
            repeat = repeat - MAX_REPEAT_NUM
            ori_dst_offset = ori_dst_offset + block_num*MAX_REPEAT_NUM
            ori_src_offset = ori_src_offset + block_num*MAX_REPEAT_NUM
        if repeat > 0:
            tik_instance.vconv(block_num, mode, dst_ub[ori_dst_offset], src_ub[ori_src_offset],
                               repeat, 1, 1, dst_repeat_stride, src_repeat_stride, deqscale=deq_scale)
            ori_dst_offset = ori_dst_offset + block_num*repeat
            ori_src_offset = ori_src_offset + block_num*repeat
        if repeat_tail > 0:
            tik_instance.vconv(repeat_tail, mode, dst_ub[ori_dst_offset], src_ub[ori_src_offset],
                               1, 1, 1, dst_repeat_stride, src_repeat_stride, deqscale=deq_scale)

    if src_dtype in ("float32",) and dst_dtype in ("int32",):
        cast_flag = tbe_platform_adapter.api_check_support("tik.vconv", "f322s32r")
        if not cast_flag:
            with tik_instance.new_stmt_scope():
                tmp_fp16_ub = tik_instance.Tensor(
                    "float16", (((do_len + 15) // 16) * 16,),
                    name="tmp_fp16_ub", scope=tik.scope_ubuf)
                tik_func_vconv(tik_instance, tmp_fp16_ub, src_ub, do_len)
                tik_func_vconv(tik_instance, dst_ub, tmp_fp16_ub, do_len, mode)
                if mode == "floor":
                    # when the product not support f322s32, will cast to fp16 and to int32, will get error
                    # ex: f32 value is 1.99998, cast int32 is 2, this step will reduce the error
                    # step 1 int32 cast to fp32_new   2.0
                    # step 2 int32_sub_fp32_value = f32_old - fp32_new
                    # step 3 int32_sub_fp32_value = 0 when int32_sub_fp32_value >= 0
                    #        int32_sub_fp32_value = 1 when int32_sub_fp32_value < 0
                    # step 4 int32 - int32_sub_fp32_value
                    if mini_mid_ub is None:
                        tmp_fp32_ub = tik_instance.Tensor(
                            "float32", (((do_len + 15) // 16) * 16,),
                            name="tmp_fp32_ub", scope=tik.scope_ubuf)
                    else:
                        tmp_fp32_ub = mini_mid_ub
                    tmp_fp32_ub_error = tik_instance.Tensor(
                        "float32", (((do_len + 15) // 16) * 16,),
                        name="tmp_fp32_ub_error", scope=tik.scope_ubuf)
                    tik_func_vconv(tik_instance, tmp_fp16_ub, dst_ub, do_len)
                    tik_func_vconv(tik_instance, tmp_fp32_ub, tmp_fp16_ub, do_len)
                    tik_func_vcomple(tik_instance, "vsub", tmp_fp32_ub_error,
                                     tmp_fp32_ub, src_ub, do_len)
                    tmp_zero = tik_instance.Tensor("float32", (8,), name="tmp_zero", scope=tik.scope_ubuf)
                    tmp_min_fp32 = tik_instance.Tensor("float32", (8,), name="tmp_minest_fp32", scope=tik.scope_ubuf)
                    tik_instance.vmuls(8, tmp_zero, tmp_zero, 0.0, 1, 1, 1, 8, 8)
                    tik_instance.vector_dup(8, tmp_min_fp32, SCALAR_MIN_FP32, 1, 1, 1)
                    tik_func_vcomple(tik_instance, "vmax", tmp_fp32_ub_error,
                                     tmp_zero, tmp_fp32_ub_error, do_len, src0_rep=0, src0_blk=0)
                    tik_func_vcomple(tik_instance, "vmin", tmp_fp32_ub_error,
                                     tmp_min_fp32, tmp_fp32_ub_error, do_len, src0_rep=0, src0_blk=0)
                    tik_func_vmuls(tik_instance, tmp_fp32_ub_error,
                                   tmp_fp32_ub_error, SCALAR_MUL_FP32, do_len)
                    tik_func_vmuls(tik_instance, tmp_fp32_ub_error,
                                   tmp_fp32_ub_error, SCALAR_MUL_FP32, do_len)
                    tik_func_vmuls(tik_instance, tmp_fp32_ub_error,
                                   tmp_fp32_ub_error, SCALAR_MUL2_FP32, do_len)
                    tik_func_vcomple(tik_instance, "vsub", tmp_fp32_ub,
                                     tmp_fp32_ub, tmp_fp32_ub_error, do_len)
                    tik_func_vconv(tik_instance, tmp_fp16_ub, tmp_fp32_ub, do_len)
                    tik_func_vconv(tik_instance, dst_ub, tmp_fp16_ub, do_len, "round")
        else:
            do_vconv(8, 8)

    elif src_dtype in ("float32",) and dst_dtype in ("float16",):
        do_vconv(4, 8)

    elif src_dtype in ("float16",) and dst_dtype in ("int32",):
        do_vconv(8, 4)

    elif src_dtype in ("int32",) and dst_dtype in ("float16",):
        do_vconv(4, 8, 1.0)

    elif src_dtype in ("float16",) and dst_dtype in ("float32",):
        do_vconv(8, 4)

    elif src_dtype in ("int32",) and dst_dtype in ("float32",):
        cast_flag = tbe_platform_adapter.api_check_support("tik.vconv", "s322f32")
        if not cast_flag:
            with tik_instance.new_stmt_scope():
                tmp_fp16_ub = tik_instance.Tensor(
                    "float16", (((do_len + 15) // 16) * 16,),
                    name="tmp_fp16_ub", scope=tik.scope_ubuf)
                tik_func_vconv(tik_instance, tmp_fp16_ub, src_ub, do_len)
                tik_func_vconv(tik_instance, dst_ub, tmp_fp16_ub, do_len)
        else:
            do_vconv(8, 8)


def ceil_div(int1, int2):
    """
    ceil for (int1 / int2)
    :param int1: Scalar variable or an immediate
    :param int2: Scalar variable or an immediate
    :return: ceil for (int1 / int2)
    """
    return (int1 + int2 - 1) // int2


def ub2ub(tik_instance: tik.Tik, dst: tik.Tensor, src: tik.Tensor, count, tail_overlap=True):
    """
    move data from ub to ub
    :param tik_instance: tik instance
    :param dst: dst ub
    :param src: src ub
    :param count: count to move
    :param tail_overlap: when count is not 32 bytes align, set to allow write overlap the tail count of dst from src.
            For example, to move 5 count fof float32 data, which is not 32 bytes align,
            the tail count is 3 (32 // sizeof(float32) - (5 % (32 // sizeof(float32)))),
            if tail_overlap is `True`, will write more 3 count data at dst to make better performance
    :return: None
    """
    if dst.scope != tik.scope_ubuf or src.scope != tik.scope_ubuf:
        raise RuntimeError("dst and src must be UB, but dst is {} and src is {}.".format(dst.scope, src.scope))

    if dst.dtype != src.dtype:
        raise RuntimeError("dst.dtype[{}] != src.dtype[{}].".format(dst.dtype, src.dtype))

    dtype_size = common_util.get_data_size(src.dtype)
    block_element = constant_util.BLOCK_SIZE // dtype_size
    if tail_overlap:
        burst = ceil_div(count, block_element)
        tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)
    else:
        burst = count // block_element
        with tik_instance.if_scope(burst != 0):
            tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)
        new_index = block_element * burst
        with tik_instance.for_range(new_index, count) as index:
            dst[index] = src[index]


def ub2gm(tik_instance: tik.Tik, dst: tik.Tensor, src: tik.Tensor, count, burst=None):
    """
    move data from ub to gm
    :param tik_instance: tik instance
    :param dst: dst gm
    :param src: src ub
    :param count: count to move
    :param burst: burst to move, if is None, burst=ceil(count / block_element), by default None
    :return: None
    """
    if dst.scope != tik.scope_gm:
        raise RuntimeError("dst must be global, but dst is {}.".format(dst.scope))

    if src.scope != tik.scope_ubuf:
        raise RuntimeError("src must be UB, but src is {}.".format(src.scope))

    if dst.dtype != src.dtype:
        raise RuntimeError("dst.dtype[{}] != src.dtype[{}].".format(dst.dtype, src.dtype))

    dtype_size = common_util.get_data_size(src.dtype)
    block_element = constant_util.BLOCK_SIZE // dtype_size
    if burst is None:
        burst = ceil_div(count, block_element)
    tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)


def gm2ub(tik_instance: tik.Tik, dst: tik.Tensor, src: tik.Tensor, count, burst=None):
    """
    move data from gm to ub
    :param tik_instance: tik instance
    :param dst: dst ub
    :param src: src gm
    :param count: count to move
    :param burst: burst to move, if is None, burst=ceil(count / block_element), by default None
    :return: None
    """
    if dst.scope != tik.scope_ubuf:
        raise RuntimeError("dst must be UB, but dst is {}.".format(dst.scope))

    if src.scope != tik.scope_gm:
        raise RuntimeError("src must be global, but src is {}.".format(src.scope))

    if dst.dtype != src.dtype:
        raise RuntimeError("dst.dtype[{}] != src.dtype[{}].".format(dst.dtype, src.dtype))

    dtype_size = common_util.get_data_size(src.dtype)
    block_element = constant_util.BLOCK_SIZE // dtype_size
    if burst is None:
        burst = ceil_div(count, block_element)
    tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)


def ceil_align(count, base):
    """
    Get the ceil number `count` align of `base`

    Parameters
    ----------
    count :
    base :

    Returns
    -------
        `count` ceil align of `base`
    """
    return ceil_div(count, base) * base


def floor_align(count, base):
    """
    Get the floor number `count` align of `base`

    Parameters
    ----------
    count :
    base :

    Returns
    -------
        `count` floor align of `base`
    """
    return count // base * base
