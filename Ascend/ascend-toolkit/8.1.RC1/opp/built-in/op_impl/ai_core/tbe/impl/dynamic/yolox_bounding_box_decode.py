# Copyright 2022 Huawei Technologies Co., Ltd
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
dynamic yolox_bounding_box_decode
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from impl import constant_util as constant
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    NUMBER_FOUR = 4
    # the number of blocks skipped per repeat
    STRIDE_EIGHT = 8
    # the number of blocks skipped per repeat
    STRIDE_FOUR = 4
    # the number of blocks skipped per repeat
    STRIDE_ONE = 1
    # the number of blocks per transposition
    LIST_NUMBER = 16
    # the number of transposes per repeat
    NUMBER_TWO = 2
    # max int64
    MAX_INT64 = 2 ** 63 - 1
    TILING_SCALAR_DTYPE = "int64"
    # tiling param num
    TILING_ARG_NUM = 3
    # the dim of priors shape
    PRIORS_DIMS = 2
    # the dim of bboxes shape
    BBOXES_DIMS = 3
    # the number of the last dim
    DIM_FOUR = 4


# 'pylint: disable=useless-object-inheritance,too-many-instance-attributes
class YoloxBoundingBoxDecode(object):
    """
       Function: use to store YoloxBoundingBoxDecode base parameters
    """

    # 'pylint: disable=too-many-arguments,invalid-name
    def __init__(self, priors, bboxes, kernel_name):
        """
        Init BoundingBoxDecode base parameters

        Parameters
        ----------
        priors : dict
            shape and dtype of input priors
        bboxes : dict
            shape and dtype of input bboxes
        kernel_name : str
            kernel name, default value is "yolox_bounding_box_decode"

        Returns
        -------
        None
        """
   
        byte_size = 8
        
        # loop split unit, split by axis 2, which is the number of boxes
        self.tik_instance = tik.Tik()
        self.split_unit = self.tik_instance.Scalar("int32", name="split_unit", init_value=512)

        self.prois_dtype = priors.get("dtype").lower()
        self.bboxes_dtype = bboxes.get("dtype").lower()
        self.kernel_name = kernel_name

        self.bboxes_dtype_bytes_size = get_bit_len(self.bboxes_dtype) // byte_size
        self.bboxes_data_each_block = constant.BLOCK_SIZE // self.bboxes_dtype_bytes_size 
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        # init gm data
        self.bboxes_gm = self.tik_instance.Tensor(self.bboxes_dtype, [Constant.MAX_INT64],
                                                        name="bboxes_gm", scope=tik.scope_gm)
        self.priors_gm = self.tik_instance.Tensor(self.prois_dtype, [Constant.MAX_INT64],
                                                        name="priors_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.bboxes_dtype, [Constant.MAX_INT64],
                                                        name="output_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, (Constant.TILING_ARG_NUM,),
                                                        name="tiling_gm", scope=tik.scope_gm)                                          
    
        # init tiling data
        self.batch_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="batch_num")
        self.bboxes_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="bboxes_num")
        self.core_num_var = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="bboxes_num", 
                                                     init_value=self.core_num)
       

    def batch_computational_logic(self, p1_ub, p2_ub, b1_ub, b2_ub, box_num, output_addr):

        """
        the computational logic of every batch

        Parameters
        ----------
        p1_ub: the front 2 slices of prior info by every splitting loop, like prior[loop1:loop2, :2]
        p2_ub: the end 2 slices of prior info by every splitting loop, like prior[loop1:loop2, :2]
        b1_ub: the front 2 slices of prior info by every batch and splitting loop, like bboxes[ibatch, loop1:loop2, :2]
        b2_ub: the end 2 slices of prior info by every batch and splitting loop, like bboxes[ibatch, loop1:loop2, 2:]
        box_num: the boxes number of current loop
        output_addr: the start written address of output_gm

        the math formula:
        xys = bboxes[..., :2] * priors[:, 2:] + priors[:, :2]
        whs = bboxes[..., :2].exp() * priors[:, 2:]
        whs = whs / 2
        tl_x = xys[..., 0] - whs[..., 0]
        tl_y = xys[..., 1] - whs[..., 1]
        br_x = xys[..., 0] + whs[..., 0]
        br_y = xys[..., 1] + whs[..., 1]
        res = torch.stack([t1_x, t1_y, br_x, br_y], -1)
        
        Returns
        -------
        """
        
        stride = 8
        half_stride = 4
        mask_64 = 64
        mask_128 = 128
        value_num = box_num * 4
        MASK_NUM = mask_128 if self.bboxes_dtype == 'float16' else mask_64
   
        xys = self.tik_instance.Tensor(self.bboxes_dtype, (box_num, Constant.DIM_FOUR), 
                                                    name='xys', scope=tik.scope_ubuf)
        self.tik_instance.vec_mul(MASK_NUM, xys, b1_ub, p2_ub, value_num // MASK_NUM, 
                                                    stride, stride, stride)
        self.tik_instance.vec_add(MASK_NUM, xys, p1_ub, xys, value_num // MASK_NUM, 
                                                    stride, stride, stride)

        whs = self.tik_instance.Tensor(self.bboxes_dtype, (box_num, Constant.DIM_FOUR), 
                                                    name='whs', scope=tik.scope_ubuf)
        if tbe_platform.api_check_support("tik.vec_exp", self.bboxes_dtype):
            self.tik_instance.vec_exp(MASK_NUM, whs, b2_ub, value_num // MASK_NUM, stride, stride)
        else:
            b2_ub_16 = self.tik_instance.Tensor('float16', (box_num, Constant.DIM_FOUR), 
                                                    name='b2_ub_16', scope=tik.scope_ubuf)
            self.tik_instance.vconv(mask_64, '', b2_ub_16, b2_ub, value_num // mask_64, 
                                                    1, 1, half_stride, stride)
            self.tik_instance.vec_exp(mask_128, b2_ub_16, b2_ub_16, value_num // mask_128, 
                                                    stride, stride)
            self.tik_instance.vconv(mask_64, '', whs, b2_ub_16, value_num // mask_64, 
                                                    1, 1, stride, half_stride)

        self.tik_instance.vec_mul(MASK_NUM, whs, p2_ub, whs, value_num // MASK_NUM, 
                                                    stride, stride, stride)
        self.tik_instance.vec_muls(MASK_NUM, whs, whs, 0.5, value_num // MASK_NUM, 
                                                    stride, stride)

        top_left = self.tik_instance.Tensor(self.bboxes_dtype, (box_num, Constant.DIM_FOUR), 
                                                    name='top_left', scope=tik.scope_ubuf)
        self.tik_instance.vec_sub(MASK_NUM, top_left, xys, whs, value_num // MASK_NUM, 
                                                    stride, stride, stride)

        bottom_right = self.tik_instance.Tensor(self.bboxes_dtype, (box_num, Constant.DIM_FOUR), 
                                                    name='bottom_right', scope=tik.scope_ubuf)
        self.tik_instance.vec_add(MASK_NUM, bottom_right, xys, whs, value_num // MASK_NUM, 
                                                    stride, stride, stride)

        with self.tik_instance.for_range(0, box_num) as i:
            top_left[i * 4 + 2].set_as(bottom_right[i * 4])
            top_left[i * 4 + 3].set_as(bottom_right[i * 4 + 1])

        self.tik_instance.data_move(self.output_gm[output_addr], top_left, 0, 1, 
                                                    value_num // self.bboxes_data_each_block, 0, 0)


    def tiling_args(self, tiling_ub):
        """
        get runtime tiling params from tiling

        Parameters
        ----------

        Returns
        -------
        None
        """

        # read tiling int64 scalar
        self.batch_num.set_as(tiling_ub[0])
        self.bboxes_num.set_as(tiling_ub[1])
        self.core_num_var.set_as(tiling_ub[2])


    def reset_split_unit(self):
        """
        reset the split unit

        Parameters
        ----------

        Returns
        -------
        None
        """

        with self.tik_instance.if_scope(self.bboxes_num < self.split_unit):
            self.split_unit.set_as(32)

    def slice_fallback_address_compute(self, box_num, loop_addr, batch_addr):
        """
        compute the result of the last loop of each batch by fallback address 

        Parameters
        ----------
        box_num: the box number of the last loop drop
        loop_addr: the begin address of the gm proior info 
        batch_addr: the begin address of the batch gm info
        
        Returns
        -------
        None
        """
        data_num = self.split_unit * 4
        back_num = (self.split_unit - box_num) * 4

        back_loop_addr = loop_addr - back_num
        back_batch_addr = batch_addr - back_num

        b1_ub = self.tik_instance.Tensor(self.bboxes_dtype, (self.split_unit, 4), 
                                                name='b1_ub', scope=tik.scope_ubuf)
        b2_ub = self.tik_instance.Tensor(self.bboxes_dtype, (self.split_unit, 4), 
                                                name='b2_ub', scope=tik.scope_ubuf)
        
        self.tik_instance.data_move(b1_ub, self.bboxes_gm[back_batch_addr], 0, 1, 
                                                data_num//self.bboxes_data_each_block, 0, 0)
        self.tik_instance.data_move(b2_ub, self.bboxes_gm[back_batch_addr + 2], 0, 1, 
                                                data_num//self.bboxes_data_each_block, 0, 0)
        
        p1_ub = self.tik_instance.Tensor(self.bboxes_dtype, (self.split_unit, 4), 
                                                name='p1_ub', scope=tik.scope_ubuf)
        p2_ub = self.tik_instance.Tensor(self.bboxes_dtype, (self.split_unit, 4), 
                                                name='p2_ub', scope=tik.scope_ubuf)  

        self.tik_instance.data_move(p1_ub, self.priors_gm[back_loop_addr], 0, 1, 
                                                data_num//self.bboxes_data_each_block, 0, 0)
        self.tik_instance.data_move(p2_ub, self.priors_gm[back_loop_addr + 2], 0, 1, 
                                                data_num//self.bboxes_data_each_block, 0, 0)
        
        self.batch_computational_logic(p1_ub, p2_ub, b1_ub, b2_ub, self.split_unit, back_batch_addr)


    def slice_normal_compute(self, loop_addr, batch_addr):
        """
        compute the result of the common loop of each batch

        Parameters
        ----------
        loop_addr: the begin address of the gm proior info 
        batch_addr: the begin address of the batch gm info
        
        Returns
        -------
        None
        """
        data_num = self.split_unit * 4
        b1_ub = self.tik_instance.Tensor(self.bboxes_dtype, (self.split_unit, 4), 
                                                name='b1_ub', scope=tik.scope_ubuf)
        b2_ub = self.tik_instance.Tensor(self.bboxes_dtype, (self.split_unit, 4), 
                                                name='b2_ub', scope=tik.scope_ubuf)

        self.tik_instance.data_move(b1_ub, self.bboxes_gm[batch_addr], 0, 1, 
                                                data_num//self.bboxes_data_each_block, 0, 0)
        self.tik_instance.data_move(b2_ub, self.bboxes_gm[batch_addr+2], 0, 1, 
                                                data_num//self.bboxes_data_each_block, 0, 0)

        p1_ub = self.tik_instance.Tensor(self.bboxes_dtype, (self.split_unit, 4), 
                                                name='p1_ub', scope=tik.scope_ubuf)
        p2_ub = self.tik_instance.Tensor(self.bboxes_dtype, (self.split_unit, 4), 
                                                name='p2_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(p1_ub, self.priors_gm[loop_addr], 0, 1, 
                                                data_num//self.bboxes_data_each_block, 0, 0)
        self.tik_instance.data_move(p2_ub, self.priors_gm[loop_addr+2], 0, 1, 
                                                data_num//self.bboxes_data_each_block, 0, 0)

        self.batch_computational_logic(p1_ub, p2_ub, b1_ub, b2_ub, self.split_unit, batch_addr)

              
    def batch_calculate(self, batch_id):
        """
        calculate the decode bbox of every batch

        Parameters
        ----------
        batch_id: batch index , eg:(N, C, 4), here is Range(N)
        
        Returns
        -------
        None
        """

        n_loop = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name='n_loop')
        n_loop.set_as(self.bboxes_num // self.split_unit)
        with self.tik_instance.if_scope(self.bboxes_num % self.split_unit > 0):
            n_loop.set_as(self.bboxes_num // self.split_unit + 1)

        with self.tik_instance.for_range(0, n_loop) as loop_id:
            data_num = self.split_unit * 4
            loop_addr = loop_id * data_num
            batch_addr = batch_id * (self.bboxes_num * 4)  + loop_addr

            with self.tik_instance.if_scope(tik.all(loop_id == n_loop - 1, self.bboxes_num % self.split_unit > 0)):
                self.slice_fallback_address_compute(self.bboxes_num % self.split_unit, loop_addr, batch_addr)
            
            with self.tik_instance.else_scope():
                self.slice_normal_compute(loop_addr, batch_addr)


    def get_tik_instance(self):
        """
        the entry of yolox_bounding_box_decode calculation

        Parameters
        ----------

        Returns
        -------
        None
        """

        # set tiling info
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, (Constant.TILING_ARG_NUM,), 
                                                     name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
            self.tiling_args(tiling_ub)

        # reset the split unit
        self.reset_split_unit()
        block_batches = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name='block_batches')
        tail_batches = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name='tail_batches')
        tail_batches.set_as(self.batch_num % self.core_num_var)
        i_batch = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name='i_batch', init_value=0)

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as block_id:
            with self.tik_instance.if_scope(block_id < tail_batches):
                block_batches.set_as(self.batch_num // self.core_num_var + 1)
                i_batch.set_as(block_id * block_batches)
            
            with self.tik_instance.else_scope():
                block_batches.set_as(self.batch_num // self.core_num_var)
                i_batch.set_as(tail_batches * (block_batches + 1) + (block_id  - tail_batches) * block_batches)
                
            with self.tik_instance.for_range(0, block_batches) as batch_id:
                self.batch_calculate(i_batch + batch_id)

        opt_config = {"out_of_bound_sync_check":True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info("vars",
                                                   {"core_num": self.core_num,
                                                    "bboxes_data_each_block": self.bboxes_data_each_block})
                                                    
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.priors_gm, self.bboxes_gm],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm], config=opt_config)
        return self.tik_instance


# 'pylint: disable=too-many-arguments, too-many-instance-attributes
# 'pylint: disable=unused-argument, too-many-locals, too-many-lines
@register_operator("YoloxBoundingBoxDecode")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def yolox_bounding_box_decode(priors,
                        bboxes,
                        decoded_bboxes,
                        kernel_name="yolox_bounding_box_decode"):
    """
    bboxes decode interface

    Parameters
    ----------
    priors : dict
        shape and dtype of input priors
    bboxes : dict
        shape and dtype of input bboxes
    decoded_bboxes : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "yolox_bounding_box_decode"

    Returns
    -------
    None
    """
    priors_dtype = priors.get("dtype").lower()
    bboxes_dtype = bboxes.get("dtype").lower()
    para_check.check_dtype(priors_dtype, ["float16", "float32"])
    para_check.check_dtype(bboxes_dtype, ["float16", "float32"])
    if priors_dtype != bboxes_dtype:
        error_detail = "dtype of priors and bboxes_input should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "priors", "bboxes_input", error_detail)

    priors_shape = priors.get('shape')
    bboxes_shape = bboxes.get('shape')
    if len(priors_shape) != Constant.PRIORS_DIMS or len(bboxes_shape) != Constant.BBOXES_DIMS:
        error_detail = "shape of priors and bboxes_input should match"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "priors", "bboxes_input", error_detail)

    bboxes_instance = YoloxBoundingBoxDecode(priors, bboxes, kernel_name)
    instance = bboxes_instance.get_tik_instance()
    return instance
