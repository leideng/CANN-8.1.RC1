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
dynamic_lstm_grad_cell
"""
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import add_compile_info
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    INT32_MAX_NUM = 2 * 32 - 1
    TILLING_ARG_NUM = 8
    T_STATE_NUM = 1
    INT64 = 'int64'
    INT32 = 'int32'
    FORWARD = 'UNIDIRECTIONAL'
    TILLING_PARA_INDEX_MAP = {
        't_size': 0,
        'ubSize': 1,
        'hiddenSize': 2,
        'batchSize': 3,
        'useCoreNum': 4,
        'fuseSize': 5,
    }


# 'pylint: disable=too-many-instance-attributes
class LstmCellGradInput:
    """
    Class: use to store LstmCellGradInput input parameters
    Modify : 2019-12-28
    """

    # 'pylint: disable=too-many-arguments,unused-argument
    def __init__(self, mask, init_c, cell_state, dht_out, dht, dct, input_gate, forget_gate,
                 update_gate, output_gate, tanh_ct, gate_order, direction, kernel_name):
        """
        init LstmCellGradInput base parameters

        Parameters
        ----------
        cell_state: dict
            cell state at the last moment
        dht_out: dict
            output state gradient at time t
        dht: dict
            hidden state gradient at time t
        dct: dict
            cell state gradient at time t

        input_gate: dict
            forward it buffer value at time t
        forget_gate: dict
            forward ft buffer value at time t
        update_gate: dict
            forward jt buffer value at time t
        output_gate: dict
            forward ot buffer value at time t
        tanh_ct: dict
            forward tanh_ct buffer value at time t
        kernel_name: str
            op kernel name

        Returns
        -------
        None
        """
        self.dht_out_shape = dht_out.get("shape")
        self.dht_out_dtype = dht_out.get("dtype")
        if mask is None:
            self.mask_shape = None
        else:
            self.mask_shape = mask.get('shape')

        self.dht_shape = dht.get("shape")
        self.dht_dtype = dht.get("dtype")
        self.dct_shape = dct.get("shape")
        self.dct_dtype = dct.get("dtype")
        self.it_shape = input_gate.get("shape")
        self.it_dtype = input_gate.get("dtype")
        self.ft_shape = forget_gate.get("shape")
        self.ft_dtype = forget_gate.get("dtype")
        self.jt_shape = update_gate.get("shape")
        self.jt_dtype = update_gate.get("dtype")
        self.ot_shape = output_gate.get("shape")
        self.ot_dtype = output_gate.get("dtype")
        self.tanh_ct_shape = tanh_ct.get("shape")
        self.tanh_ct_dtype = tanh_ct.get("dtype")
        self.c_shape = cell_state.get("shape")
        self.c_dtype = cell_state.get("dtype")
        self.format = input_gate.get("format")

        self.t_size = None
        self.ub_size = None
        self.batch_size = None
        self.hidden_size = None
        self.use_core_num = None
        self.fuse_size = None

        self.t_state = None
        self.dgate_shape = None
        self.gm_init_c = None
        self.gm_c = None
        self.gm_dht_out = None
        self.gm_dht = None
        self.gm_dct = None
        self.gm_it = None
        self.gm_jt = None
        self.gm_ft = None
        self.gm_ot = None
        self.gm_tanh_ct = None
        self.gm_t_state = None
        self.gm_mask = None
        self.tilling_gm = None
        self.gm_dct1 = None
        self.gm_dgate = None
        self.gm_dgate1 = None
        self.gm_dgate2 = None
        self.gm_dgate3 = None

        self.dgate_dtype = output_gate.get('dtype')
        self.direction = direction
        self.gate_order = gate_order
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.aicore_num = self.tik_instance.d_profiling.get_aicore_num()

        self.check_input_param()
        self.init_gm_tensor()

    def check_input_param(self):
        """
        Check the input parameter

        Parameters
        ----------
        None

        Returns:
        None
        """
        if self.gate_order not in ['ijfo', 'ifjo']:
            raise RuntimeError('gate_order illegal')
        shape_list = (self.c_shape, self.it_shape, self.jt_shape, self.ft_shape, self.ot_shape, self.tanh_ct_shape)
        no_t_shape = self.c_shape[1:]
        if self.format == "FRACTAL_NZ":
            for shape in shape_list:
                para_check.check_shape(shape, min_rank=4, max_rank=5, param_name="dht_out")
                shape = shape if len(shape) == 4 else shape[1:]
                if shape != no_t_shape:
                    raise RuntimeError("the input shapes are not same")

        check_list = ("float16", "float32")
        dtype_list = (self.c_dtype, self.dht_dtype, self.dht_out_dtype, self.dct_dtype,
                      self.it_dtype, self.jt_dtype, self.ft_dtype,
                      self.ot_dtype, self.tanh_ct_dtype)

        for dtype in dtype_list:
            para_check.check_dtype(dtype.lower(), check_list, param_name="dht_out")
            if dtype != self.c_dtype:
                raise RuntimeError("the input dtypes are not same")

    def init_gm_tensor(self):
        """
        Declare tensor on gm

        Parameters
        ----------
        None

        Returns:
        None
        """
        self.gm_dht_out = self.tik_instance.Tensor(self.dht_out_dtype, (Constant.INT32_MAX_NUM,),
                                                   name="gm_dht_out", scope=tik.scope_gm)
        self.gm_dht = self.tik_instance.Tensor(
            self.dht_dtype, (Constant.INT32_MAX_NUM,), name="gm_dht", scope=tik.scope_gm)
        self.gm_dct = self.tik_instance.Tensor(
            self.dct_dtype, (Constant.INT32_MAX_NUM,), name="gm_dct", scope=tik.scope_gm)
        self.gm_it = self.tik_instance.Tensor(
            self.it_dtype, (Constant.INT32_MAX_NUM,), name="gm_it", scope=tik.scope_gm)
        self.gm_ft = self.tik_instance.Tensor(
            self.ft_dtype, (Constant.INT32_MAX_NUM,), name="gm_ft", scope=tik.scope_gm)
        self.gm_jt = self.tik_instance.Tensor(
            self.jt_dtype, (Constant.INT32_MAX_NUM,), name="gm_jt", scope=tik.scope_gm)
        self.gm_ot = self.tik_instance.Tensor(
            self.ot_dtype, (Constant.INT32_MAX_NUM,), name="gm_ot", scope=tik.scope_gm)
        self.gm_tanh_ct = self.tik_instance.Tensor(
            self.tanh_ct_dtype,
            (Constant.INT32_MAX_NUM,),
            name="gm_tanh_ct",
            scope=tik.scope_gm)
        self.gm_c = self.tik_instance.Tensor(
            self.c_dtype, (Constant.INT32_MAX_NUM,), name="gm_c", scope=tik.scope_gm)
        self.gm_init_c = self.tik_instance.Tensor(
            self.c_dtype, (Constant.INT32_MAX_NUM,), name="gm_init_c", scope=tik.scope_gm)
        if self.mask_shape is not None:
            self.gm_mask = self.tik_instance.Tensor(
                self.c_dtype, (Constant.INT32_MAX_NUM,), name="gm_mask", scope=tik.scope_gm)
        self.gm_t_state = self.tik_instance.Tensor(
            Constant.INT32, (Constant.T_STATE_NUM,), name="gm_t_state", scope=tik.scope_gm)
        self.tilling_gm = self.tik_instance.Tensor(
            Constant.INT64, (Constant.TILLING_ARG_NUM,), name="tilling_gm", scope=tik.scope_gm)
        # output gm
        self.gm_dct1 = self.tik_instance.Tensor(
            self.c_dtype, (Constant.INT32_MAX_NUM,), name="gm_dct1", scope=tik.scope_gm)

        self.gm_dgate = self.tik_instance.Tensor(
            self.dgate_dtype,
            (Constant.INT32_MAX_NUM,),
            name="gm_dgate",
            scope=tik.scope_gm)
        if self.format == "ND":
            self.gm_dgate1 = self.tik_instance.Tensor(self.dgate_dtype, (Constant.INT32_MAX_NUM,),
                                                      name="gm_dgate1", scope=tik.scope_gm)
            self.gm_dgate2 = self.tik_instance.Tensor(self.dgate_dtype, (Constant.INT32_MAX_NUM,),
                                                      name="gm_dgate2", scope=tik.scope_gm)
            self.gm_dgate3 = self.tik_instance.Tensor(self.dgate_dtype, (Constant.INT32_MAX_NUM,),
                                                      name="gm_dgate3", scope=tik.scope_gm)


class LstmCellGrad(LstmCellGradInput):
    """
    Class: use to store LstmCellGrad input parameters
    Modify : 2019-12-28
    """

    # 'pylint: disable=too-many-arguments,too-many-locals
    def __init__(self, mask, init_c, cell_state, dht_out, dht, dct, input_gate, forget_gate,
                 update_gate, output_gate, tanh_ct, gate_order, direction, kernel_name):
        """
        init LstmCellGradInput base parameters

        Parameters
        ----------
        cell_state: dict
            cell state at the last moment
        dht_out: dict
            output state gradient at time t
        dht: dict
            hidden state gradient at time t
        dct: dict
            cell state gradient at time t

        input_gate: dict
            forward it buffer value at time t
        forget_gate: dict
            forward ft buffer value at time t
        update_gate: dict
            forward jt buffer value at time t
        output_gate: dict
            forward ot buffer value at time t
        tanh_ct: dict
            forward tanh_ct buffer value at time t
        kernel_name: str
            op kernel name

        Returns
        -------
        None
        """
        # 'pylint: disable=super-with-arguments
        super(LstmCellGrad, self).__init__(mask, init_c, cell_state, dht_out, dht, dct, input_gate,
                                           forget_gate, update_gate, output_gate, tanh_ct, gate_order,
                                           direction, kernel_name)
        self.each_core_size = None
        self.each_ub_loop = None
        self.each_ub_tail_size = None
        self.each_core_offset = None

        self.ub_pice_num = 15
        if self.mask_shape is not None:
            # ub tensor count
            self.ub_pice_num = self.ub_pice_num + 1

        # get vector compute parameters
        dtype_bytes_size = get_bit_len(self.dht_dtype) // 8
        int64_bytes_size = 8
        int32_bytes_size = 4
        block_size = 32
        self.v_mask_max = 128 // (dtype_bytes_size // 2)
        self.v_repeat_max = 255
        self.v_ele_each_block = 32 // dtype_bytes_size

        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        t_state_size = (Constant.T_STATE_NUM * int32_bytes_size + block_size - 1) // block_size * block_size
        tiling_size = (Constant.TILLING_ARG_NUM * int64_bytes_size + block_size - 1) // block_size * block_size
        ub_one_size = block_size * 2
        ub_max_ele_num = (self.ub_size_bytes - (t_state_size + tiling_size + ub_one_size)) // dtype_bytes_size
        align = self.v_ele_each_block
        self.max_block_ele_num = ub_max_ele_num // self.ub_pice_num // 2 // align * align
        self.max_mem_size = dtype_bytes_size * self.max_block_ele_num

        self.ub_dht_out = None
        self.ub_dht = None
        self.ub_ot = None
        self.ub_dot = None
        self.ub_tanh_ct = None
        self.ub_dc = None
        self.ub_dct = None
        self.ub_it = None
        self.ub_jt = None
        self.ub_djt = None
        self.ub_dit = None
        self.ub_dft = None
        self.ub_c = None
        self.ub_ft = None
        self.ub_dct1 = None
        self.ub_mask = None
        self.ub_t_state = None
        self.ub_t_size = None
        self.ub_one = None

    def get_tik_instance(self):
        """
        Return tik instance for tik debug

        Parameters
        ----------
        None

        Returns:
        tik_instance:
            tik instance
        """
        return self.tik_instance

    def get_tilling_params(self, core_idx):
        """
        set tilling params
        """

        tilling_ub = self.tik_instance.Tensor(Constant.INT64,
                                              (Constant.TILLING_ARG_NUM,), name="tilling_ub", scope=tik.scope_ubuf)
        burst = (Constant.TILLING_ARG_NUM * 8 + 31) // 32
        self.tik_instance.data_move(tilling_ub, self.tilling_gm, 0, 1, burst, 0, 0)

        self.t_size = self.tik_instance.Scalar(Constant.INT64, name='t_size')
        self.t_size.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('t_size')])
        self.ub_size = self.tik_instance.Scalar(Constant.INT64, name='ub_size')
        self.ub_size.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('ubSize')])
        self.hidden_size = self.tik_instance.Scalar(Constant.INT64, name='hidden_size')
        self.hidden_size.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('hiddenSize')])
        self.batch_size = self.tik_instance.Scalar(Constant.INT64, name='batch_size')
        self.batch_size.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('batchSize')])
        self.use_core_num = self.tik_instance.Scalar(Constant.INT64, name='use_core_num')
        self.use_core_num.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('useCoreNum')])
        self.fuse_size = self.tik_instance.Scalar(Constant.INT64, name='fuse_size')
        self.fuse_size.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('fuseSize')])

        self.each_core_size = self.tik_instance.Scalar(Constant.INT64, name='each_core_size',
                                                       init_value=self.fuse_size // self.use_core_num)
        self.each_core_offset = self.tik_instance.Scalar(Constant.INT64, name='each_core_offset',
                                                         init_value=self.each_core_size * core_idx)
        core_tail = self.fuse_size % self.use_core_num
        with self.tik_instance.if_scope(core_tail > 0):
            with self.tik_instance.if_scope(core_idx < core_tail):
                self.each_core_size.set_as(self.each_core_size + 1)
                self.each_core_offset.set_as(self.each_core_size * core_idx)
            with self.tik_instance.else_scope():
                self.each_core_offset.set_as(self.each_core_size * core_idx + core_tail)

        self.each_ub_loop = self.each_core_size // self.ub_size
        self.each_ub_tail_size = self.each_core_size % self.ub_size

    def init_ub(self):
        """
        Declare tensor on UB buffer

        Parameters
        ----------
        None

        Returns:
        None
        """
        if self.mask_shape is not None:
            self.ub_mask = self.tik_instance.Tensor(
                self.dht_dtype, (self.ub_size,),
                name='ub_mask',
                scope=tik.scope_ubuf, max_mem_size=self.max_mem_size
            )
        self.ub_dht = self.tik_instance.Tensor(
            self.dht_dtype, (self.ub_size,),
            name="ub_dht",
            scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_dht_out = self.tik_instance.Tensor(
            self.dht_out_dtype, (self.ub_size,),
            name="ub_dht_out",
            scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_ot = self.tik_instance.Tensor(
            self.ot_dtype, (self.ub_size,), name="ub_ot", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_dot = self.tik_instance.Tensor(
            self.ot_dtype, (self.ub_size,), name="ub_dot", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_tanh_ct = self.tik_instance.Tensor(
            self.tanh_ct_dtype, (self.ub_size,),
            name="ub_tanh_ct",
            scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        self.ub_dc = self.tik_instance.Tensor(
            self.dct_dtype, (self.ub_size,), name="ub_dc", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_dct = self.tik_instance.Tensor(
            self.dct_dtype, (self.ub_size,),
            name="ub_dct",
            scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        self.ub_it = self.tik_instance.Tensor(
            self.it_dtype, (self.ub_size,), name="ub_it", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_jt = self.tik_instance.Tensor(
            self.jt_dtype, (self.ub_size,), name="ub_jt", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_djt = self.tik_instance.Tensor(
            self.jt_dtype, (self.ub_size,), name="ub_djt", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        self.ub_dit = self.tik_instance.Tensor(
            self.it_dtype, (self.ub_size,), name="ub_dit", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        self.ub_dft = self.tik_instance.Tensor(
            self.ft_dtype, (self.ub_size,), name="ub_dft", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_c = self.tik_instance.Tensor(
            self.c_dtype, (self.ub_size,), name="ub_c", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_ft = self.tik_instance.Tensor(
            self.ft_dtype, (self.ub_size,), name="ub_ft", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        self.ub_dct1 = self.tik_instance.Tensor(
            self.c_dtype, (self.ub_size,), name="ub_dct1", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_one = self.tik_instance.Tensor(
            self.ot_dtype, (self.v_ele_each_block,), name="ub_one", scope=tik.scope_ubuf)

    def vector_compute(self, index, mask, repeat):
        """
        Calculate the smallest data shard

        Parameters
        ----------
        src: int
            source address offset
        dst: int
            destination address offset
        mask: int
            vector compute mask
        repeat:
            vector compute repeat times
        Returns:
        None
        """
        # mask mul dy dh
        if self.mask_shape is not None:
            self.tik_instance.vmul(mask, self.ub_dht[index], self.ub_dht[index], self.ub_mask[index],
                                   repeat, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vmul(mask, self.ub_dht_out[index], self.ub_dht_out[index], self.ub_mask[index],
                                   repeat, 1, 1, 1, 8, 8, 8)
        # compute process for dot
        self.tik_instance.vadd(mask, self.ub_dht[index],
                               self.ub_dht_out[index], self.ub_dht[index],
                               repeat, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vec_dup(self.v_ele_each_block, self.ub_one, 1, 1, 8)

        # compute process for dot
        self.tik_instance.vmul(mask, self.ub_dht[index], self.ub_ot[index],
                               self.ub_dht[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dot[index], self.ub_tanh_ct[index],
                               self.ub_dht[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.ub_dct1[index], self.ub_one[0],
                               self.ub_ot[index], repeat, 1, 0, 1, 8, 0, 8)
        self.tik_instance.vmul(mask, self.ub_dot[index], self.ub_dot[index],
                               self.ub_dct1[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for dc
        self.tik_instance.vmul(mask, self.ub_dc[index], self.ub_tanh_ct[index],
                               self.ub_tanh_ct[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.ub_dc[index], self.ub_one[0],
                               self.ub_dc[index], repeat, 1, 0, 1, 8, 0, 8)
        self.tik_instance.vmul(mask, self.ub_dc[index], self.ub_dc[index],
                               self.ub_dht[index], repeat, 1, 1, 1, 8, 8, 8)
        if self.mask_shape is not None:
            self.tik_instance.vmul(mask, self.ub_tanh_ct[index], self.ub_dct[index], self.ub_mask[index],
                                   repeat, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vadd(mask, self.ub_dc[index], self.ub_dc[index],
                                   self.ub_tanh_ct[index], repeat, 1, 1, 1, 8, 8, 8)
        else:
            self.tik_instance.vadd(mask, self.ub_dc[index], self.ub_dc[index],
                                   self.ub_dct[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for dit
        self.tik_instance.vsub(mask, self.ub_dct1[index], self.ub_one[0],
                               self.ub_it[index], repeat, 1, 0, 1, 8, 0, 8)
        self.tik_instance.vmul(mask, self.ub_djt[index], self.ub_dc[index],
                               self.ub_it[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dit[index], self.ub_djt[index],
                               self.ub_jt[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dit[index], self.ub_dit[index],
                               self.ub_dct1[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for djt
        self.tik_instance.vmul(mask, self.ub_dct1[index], self.ub_jt[index],
                               self.ub_jt[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.ub_dct1[index], self.ub_one[0],
                               self.ub_dct1[index], repeat, 1, 0, 1, 8, 0, 8)
        self.tik_instance.vmul(mask, self.ub_djt[index], self.ub_djt[index],
                               self.ub_dct1[index], repeat, 1, 1, 1, 8, 8, 8)

        # fake
        # compute process for dft
        self.tik_instance.vsub(mask, self.ub_dct1[index], self.ub_one[0],
                               self.ub_ft[index], repeat, 1, 0, 1, 8, 0, 8)
        self.tik_instance.vmul(mask, self.ub_dft[index], self.ub_dc[index],
                               self.ub_c[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dft[index], self.ub_dft[index],
                               self.ub_ft[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dft[index], self.ub_dft[index],
                               self.ub_dct1[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for dct-1
        self.tik_instance.vmul(mask, self.ub_dct1[index], self.ub_dc[index],
                               self.ub_ft[index], repeat, 1, 1, 1, 8, 8, 8)
        if self.mask_shape is not None:
            self.tik_instance.vadd(mask, self.ub_dct1[index], self.ub_dct1[index],
                                   self.ub_dct[index], repeat, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vsub(mask, self.ub_dct1[index], self.ub_dct1[index],
                                   self.ub_tanh_ct[index], repeat, 1, 1, 1, 8, 8, 8)

    def compute_each_loop(self, ele_num):
        """
        Calculate each loop
        """
        # vector compute
        repeat_times = ele_num // self.v_mask_max
        with self.tik_instance.if_scope(repeat_times > 0):
            self.vector_compute(0, self.v_mask_max, repeat_times)

        tile_mask = ele_num % self.v_mask_max
        with self.tik_instance.if_scope(tile_mask > 0):
            compute_index = repeat_times * self.v_mask_max
            self.vector_compute(compute_index, tile_mask, 1)

    def calc_t_offset(self):
        """
        Calculate t_offset

        Parameters
        ----------

        Returns:
        t_offset
        """
        self.ub_t_state = self.tik_instance.Tensor(
            Constant.INT32, (4,),
            name="ub_t_state",
            scope=tik.scope_ubuf
        )
        self.tik_instance.data_move(self.ub_t_state, self.gm_t_state, 0, 1, 1, 0, 0)
        self.t_state = self.tik_instance.Scalar(Constant.INT32, name='t_state', init_value=self.ub_t_state[0])
        if self.direction == Constant.FORWARD:
            t_offset = self.t_size - self.t_state - 1
        else:
            t_offset = self.t_state
        return t_offset

    def input_data_move_in(self, start_index, t_offset, c_t_offset, ele_num):
        """
        Move the input data to ub
        """
        # move in vector data
        v_burst_lens = ele_num // self.v_ele_each_block
        if self.mask_shape is not None:
            self.tik_instance.data_move(self.ub_mask, self.gm_mask[t_offset], 0, 1, v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_dht_out,
                                    self.gm_dht_out[t_offset], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_dht, self.gm_dht[start_index], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_ot, self.gm_ot[t_offset], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_tanh_ct,
                                    self.gm_tanh_ct[t_offset], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_it, self.gm_it[t_offset], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_jt, self.gm_jt[t_offset], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_ft, self.gm_ft[t_offset], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_dct, self.gm_dct[start_index], 0, 1,
                                    v_burst_lens, 0, 0)
        with self.tik_instance.if_scope(self.t_state == self.t_size - 1):
            self.tik_instance.data_move(self.ub_c, self.gm_init_c[start_index], 0, 1, v_burst_lens, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.ub_c, self.gm_c[c_t_offset], 0, 1, v_burst_lens, 0, 0)

    def compute_process(self, start_index, t_offset, ele_num):
        """
        Calculate process on each core
        """
        cur_t_offset = t_offset - 1 if self.direction == Constant.FORWARD else t_offset + 1
        self.input_data_move_in(start_index,
                                t_offset * self.fuse_size + start_index,
                                cur_t_offset * self.fuse_size + start_index,
                                ele_num)
        self.compute_each_loop(ele_num)
        self.move_vector_data_out(start_index, ele_num)

    def compute_each_core_mode0(self, t_offset):
        """
        Calculate the data on each core
        """
        self.init_ub()
        block_size = self.v_ele_each_block
        start_index = self.each_core_offset
        num = self.each_ub_tail_size // block_size
        mod = self.each_ub_tail_size % block_size
        self.compute_process(start_index, t_offset, num * block_size)
        with self.tik_instance.if_scope(mod > 0):
            self.compute_process(start_index + self.each_ub_tail_size - self.v_ele_each_block,
                                 t_offset, self.v_ele_each_block)

    def compute_each_core_mode1(self, t_offset):
        """
        Calculate the data on each core
        """
        self.init_ub()
        block_size = self.v_ele_each_block
        start_index = self.each_core_offset
        self.compute_process(start_index, t_offset, self.ub_size)
        with self.tik_instance.if_scope(self.each_ub_tail_size > 0):
            align_size = (self.each_ub_tail_size + block_size - 1) // block_size * block_size
            pre_offset = align_size - self.each_ub_tail_size
            self.compute_process(start_index + self.ub_size - pre_offset, t_offset, align_size)

    def compute_each_core_mode2(self, t_offset):
        """
        Calculate the data on each core
        """
        loop_offset = self.each_core_offset
        block_size = self.v_ele_each_block
        with self.tik_instance.for_range(0, self.each_ub_loop, thread_num=2) as index:
            self.init_ub()
            start_index = loop_offset + index * self.ub_size
            self.compute_process(start_index, t_offset, self.ub_size)
        with self.tik_instance.if_scope(self.each_ub_tail_size > 0):
            self.init_ub()
            align_size = (self.each_ub_tail_size + block_size - 1) // block_size * block_size
            pre_offset = align_size - self.each_ub_tail_size
            start_index = loop_offset + self.each_ub_loop * self.ub_size - pre_offset
            self.compute_process(start_index, t_offset, align_size)

    def move_vector_data_out(self, index, ele_num):
        """
        Move the vector compute result to gm

        Parameters
        ----------
        index: int
            move out index
        ele_num: int
            the element number of result

        Returns:
        None
        """
        burst_len = ele_num // self.v_ele_each_block
        djt_src = self.ub_djt
        dit_src = self.ub_dit
        dot_src = self.ub_dot
        dft_src = self.ub_dft
        dgate_burst_len = burst_len
        gate_data_map = {'i': dit_src, 'j': djt_src, 'f': dft_src, 'o': dot_src}
        if self.format == "ND":
            self.tik_instance.data_move(self.gm_dgate[index], gate_data_map.get(self.gate_order[0]), 0, 1,
                                        dgate_burst_len, 0, 0)
            self.tik_instance.data_move(self.gm_dgate1[index], gate_data_map.get(self.gate_order[1]), 0,
                                        1, dgate_burst_len, 0, 0)
            self.tik_instance.data_move(self.gm_dgate2[index], gate_data_map.get(self.gate_order[2]),
                                        0, 1, dgate_burst_len, 0, 0)
            self.tik_instance.data_move(self.gm_dgate3[index], gate_data_map.get(self.gate_order[3]),
                                        0, 1, dgate_burst_len, 0, 0)
        else:
            offset = self.batch_size * self.hidden_size
            self.tik_instance.data_move(self.gm_dgate[index], gate_data_map.get(self.gate_order[0]), 0, 1,
                                        dgate_burst_len, 0, 0)
            self.tik_instance.data_move(self.gm_dgate[index + offset], gate_data_map.get(self.gate_order[1]), 0,
                                        1, dgate_burst_len, 0, 0)
            self.tik_instance.data_move(self.gm_dgate[index + offset * 2], gate_data_map.get(self.gate_order[2]),
                                        0, 1, dgate_burst_len, 0, 0)
            self.tik_instance.data_move(self.gm_dgate[index + offset * 3], gate_data_map.get(self.gate_order[3]),
                                        0, 1, dgate_burst_len, 0, 0)

        self.tik_instance.data_move(self.gm_dct1[index], self.ub_dct1, 0, 1,
                                    burst_len, 0, 0)

    def compute(self):
        """
        Calculate the data

        Parameters
        ----------
        None

        Returns:
        None
        """
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as index0:
            self.get_tilling_params(index0)
            t_offset = self.calc_t_offset()
            with self.tik_instance.if_scope(index0 < self.use_core_num):
                with self.tik_instance.if_scope(self.each_ub_loop == 0):
                    with self.tik_instance.new_stmt_scope():
                        self.compute_each_core_mode0(t_offset)
                with self.tik_instance.if_scope(self.each_ub_loop == 1):
                    with self.tik_instance.new_stmt_scope():
                        self.compute_each_core_mode1(t_offset)
                with self.tik_instance.if_scope(self.each_ub_loop >= 2):
                    with self.tik_instance.new_stmt_scope():
                        self.compute_each_core_mode2(t_offset)

        input_list = [self.gm_init_c, self.gm_c, self.gm_dht_out, self.gm_dht, self.gm_dct,
                      self.gm_it, self.gm_jt, self.gm_ft, self.gm_ot,
                      self.gm_tanh_ct, self.gm_t_state]
        if self.mask_shape is not None:
            input_list.append(self.gm_mask)

        add_compile_info("vars", {"device_aicore_num": self.aicore_num,
                                  "ub_size": tbe_platform.get_soc_spec(tbe_platform.UB_SIZE),
                                  "mask_input": 0 if self.mask_shape is None else 1})
        opt_config = {"enable_const_fold": True}

        output_list = [self.gm_dgate, self.gm_dct1]
        if self.format == "ND":
            output_list = [self.gm_dgate, self.gm_dgate1, self.gm_dgate2, self.gm_dgate3, self.gm_dct1]

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=input_list,
            outputs=output_list,
            flowtable=(self.tilling_gm,),
            enable_l2=False,
            config=opt_config)


# 'pylint: disable=unused-argument,too-many-arguments,invalid-name,too-many-locals
@register_operator("DynamicLSTMGradCell")
def dynamic_lstm_grad_cell(init_c, c, dy, dh, dc, i, j, f, o, tanhct, t_state, mask, dgate, dct1,
                           forget_bias=1, activation="tanh", direction="UNIDIRECTIONAL",
                           gate_order="ijfo", kernel_name="dynamic_lstm_grad_cell"):
    """
    Calculate the gradient of the four gates and the state of c at t-1

    Parameters
    ----------
    c: dict
        cell state at the last moment
    dht: dict
        hidden state gradient at time t
    dct: dict
        cell state gradient at time t
    it: dict
        forward it buffer value at time t
    jt: dict
        forward jt buffer value at time t
    ft: dict
        forward ft buffer value at time t
    ot: dict
        forward ot buffer value at time t
    tanh_ct: dict
        forward tanh_ct buffer value at time t
    forget_bias: int
        the bias of forget gate
    activation: str
        activation method
    kernel_name: str
        op kernel name

    Returns:
    None
    """
    dht_out = dy
    dht = dh
    dct = dc
    it = i
    jt = j
    ft = f
    ot = o

    lstm_cell_grad = LstmCellGrad(mask, init_c, c, dht_out, dht, dct, it, ft, jt, ot, tanhct, gate_order, direction,
                                  kernel_name)
    lstm_cell_grad.compute()

    return lstm_cell_grad
