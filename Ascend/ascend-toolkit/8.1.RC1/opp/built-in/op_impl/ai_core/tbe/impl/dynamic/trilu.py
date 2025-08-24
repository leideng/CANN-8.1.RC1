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


from impl.util.platform_adapter import tbe_context, tbe_platform, tik
from impl.util.util_tik_comm_func import ceil_div
from tbe.common.platform import UB_SIZE, get_soc_spec
from tbe.common.utils.errormgr.error_manager_vector import raise_err_specific_reson


class Constant:
    """
    Constant is a class for storing constant values.
    """

    BYTES_PER_BLOCK = 32
    BYTES_PER_KB = 1024
    # the 16 kb buffer is preserved for compiling optimization
    UB_PRESERVED = 16 * BYTES_PER_KB
    DTYPE_BYTES_DICT = {
        "uint8": 1,
        "int8": 1,
        "bool": 1,
        "uint16": 2,
        "int16": 2,
        "bfloat16": 2,
        "float16": 2,
        "uint32": 4,
        "int32": 4,
        "float32": 4,
        "float": 4,
        "uint64": 8,
        "int64": 8,
        "float64": 8,
        "double": 8,
    }
    MAX_REAPEAT_TIMES = 255
    MAX_INT32 = 2**31 - 1

    TILING_ARGUMENT_DTYPE = "int64"
    TILING_DATA_NUM = 9
    TILING_MODE_OUTPUT_ZERO = 0
    TILING_MODE_OUTPUT_INPUT = 1
    TILING_MODE_SMALL_MATRIX = 2
    TILING_MODE_SMALL_ROW = 3
    TILING_MODE_NORMAL = 4
    TILING_MODE_BIG_ROW = 5

    VECTOR_MASK_DICT = {1: 256, 2: 128, 4: 64}
    VECTOR_DUP_MAX_REPEAT_TIMES = 255


class Trilu:
    """
    Trilu is the basic class of tril and triu.
    """

    # pylint: disable=too-many-arguments, huawei-too-many-arguments
    def __init__(self, x, y, diagonal, upper, kernel_name):
        self.kernel_name = kernel_name
        self.upper = upper
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.available_aicore_num = tik.Dprofile().get_aicore_num()
        self.available_ub_size = (
            (get_soc_spec(UB_SIZE) - Constant.UB_PRESERVED) // Constant.BYTES_PER_BLOCK * Constant.BYTES_PER_BLOCK
        )

        self._get_dtype(x.get("dtype").lower())
        self.dsize = Constant.DTYPE_BYTES_DICT.get(self.dtype, 4)  # default float
        self.is_support_data_move_pad = tbe_platform.api_check_support("tik.data_move_pad", self.dtype)

        self.x_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="x_gm", scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="y_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(
            Constant.TILING_ARGUMENT_DTYPE, [Constant.TILING_DATA_NUM], name="tiling_gm", scope=tik.scope_gm
        )

        # tiling data
        self.tiling_mode = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="tiling_mode")
        self.matrix_num = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="matrix_num")
        self.row = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="row")
        self.col = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="col")
        self.diagonal = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="diagonal")
        self.task_num = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="task_num")
        self.elt_num_per_core = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="elt_num_per_core")
        self.mask = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="mask")
        self.used_core_num = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="used_core_num")
        self.elt_num_per_block = Constant.BYTES_PER_BLOCK // self.dsize
        self.elt_num_per_matrix = self.tik_instance.Scalar(
            dtype=Constant.TILING_ARGUMENT_DTYPE, name="elt_num_per_matrix"
        )
        self.total_row_num = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="total_row_num")
        self.total_elt_num = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="total_elt_num")
        self.row_num_per_core = self.tik_instance.Scalar(dtype=Constant.TILING_ARGUMENT_DTYPE, name="row_num_per_core")
        self.matrix_num_per_core = self.tik_instance.Scalar(
            dtype=Constant.TILING_ARGUMENT_DTYPE, name="matix_num_per_core"
        )
        self.mask_bitmap = self.tik_instance.ScalarArray("uint8", 64, name="mask_bitmap")
        self.len_mask_bitmap = self.tik_instance.Scalar("uint8", name="len_mask_bitmap", init_value=0)

        self._get_tiling_args()

    def task_dispatch(self, task_id):
        with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_OUTPUT_ZERO):
            self._execute_output_zero(task_id)
        with self.tik_instance.elif_scope(self.tiling_mode == Constant.TILING_MODE_OUTPUT_INPUT):
            self._execute_output_input(task_id)
        with self.tik_instance.elif_scope(self.tiling_mode == Constant.TILING_MODE_SMALL_MATRIX):
            self._execute_small_matrix(task_id)
        with self.tik_instance.elif_scope(self.tiling_mode == Constant.TILING_MODE_SMALL_ROW):
            self._execute_small_row(task_id)
        with self.tik_instance.elif_scope(self.tiling_mode == Constant.TILING_MODE_NORMAL):
            self._execute_normal(task_id)
        with self.tik_instance.else_scope():
            self._execute_big_row(task_id)

    def task_schedule(self):
        # load balance
        avg_task_per_core = self.tik_instance.Scalar(
            dtype=Constant.TILING_ARGUMENT_DTYPE,
            name="avg_task_per_core",
            init_value=self.task_num // self.used_core_num,
        )
        tail_task = self.tik_instance.Scalar(
            dtype=Constant.TILING_ARGUMENT_DTYPE,
            name="tail_task",
            init_value=self.task_num % self.used_core_num,
        )
        with self.tik_instance.for_range(0, self.used_core_num, block_num=self.used_core_num) as core_id:
            with self.tik_instance.for_range(0, avg_task_per_core) as i:
                self.task_dispatch(core_id * avg_task_per_core + i)
            with self.tik_instance.if_scope(core_id < tail_task):  # execute one more task
                self.task_dispatch(self.used_core_num * avg_task_per_core + core_id)

        ctx = tbe_context.get_context()
        if ctx:
            ctx.add_compile_info(
                "vars",
                {
                    "available_ub_size": self.available_ub_size,
                    "available_aicore_num": self.available_aicore_num,
                },
            )
        else:
            raise_err_specific_reson(self.kernel_name, "could not find the context")

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name, inputs=[self.x_gm], outputs=[self.y_gm], flowtable=[self.tiling_gm]
        )
        return self.tik_instance

    def _get_dtype(self, dtype: str):
        if dtype == "float64" or dtype == "double":
            self.dtype = "int64"
        elif dtype == "bool":
            self.dtype = "uint8"
        elif dtype == "bfloat16":
            self.dtype = "float16"
        else:
            self.dtype = dtype

    def _get_tiling_args(self):
        tiling_ub = self.tik_instance.Tensor(
            Constant.TILING_ARGUMENT_DTYPE, [Constant.TILING_DATA_NUM], name="tiling_ub", scope=tik.scope_ubuf
        )
        burst = ceil_div(Constant.TILING_DATA_NUM, Constant.BYTES_PER_BLOCK // 8)  # 8 for int64
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, burst, 0, 0)
        self.tiling_mode.set_as(tiling_ub[0])
        self.matrix_num.set_as(tiling_ub[1])
        self.row.set_as(tiling_ub[2])
        self.col.set_as(tiling_ub[3])
        self.diagonal.set_as(tiling_ub[4])
        self.task_num.set_as(tiling_ub[5])
        self.elt_num_per_core.set_as(tiling_ub[6])
        self.mask.set_as(tiling_ub[7])
        self.used_core_num.set_as(tiling_ub[8])

        self.elt_num_per_matrix.set_as(self.row * self.col)
        self.total_row_num.set_as(self.row * self.matrix_num)
        self.total_elt_num.set_as(self.total_row_num * self.col)
        self.row_num_per_core.set_as(self.elt_num_per_core // self.col)

        with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_SMALL_MATRIX):
            self.matrix_num_per_core.set_as(self.elt_num_per_core // self.elt_num_per_matrix)
            self.__init_mask_bitmap()

    def _execute_output_zero(self, core_id):
        """
        _execute_output_zero executes the case that y is all zero
        """
        start = core_id * self.elt_num_per_core
        ub = self.tik_instance.Tensor(self.dtype, [self.elt_num_per_core], name="output_zero_ub", scope=tik.scope_ubuf)
        self.__init_zero(ub, self.elt_num_per_core)

        move_num = self.tik_instance.Scalar(
            dtype=Constant.TILING_ARGUMENT_DTYPE, name="move_num", init_value=self.elt_num_per_core
        )
        with self.tik_instance.if_scope(core_id == self.task_num - 1):
            move_num.set_as(self.total_elt_num - start)

        self.__move(self.y_gm, ub, start, 0, move_num)

    def _execute_output_input(self, core_id):
        """
        _execute_output_input executes the case that y is the same as x
        """
        start = core_id * self.elt_num_per_core
        ub = self.tik_instance.Tensor(self.dtype, [self.elt_num_per_core], name="output_input_ub", scope=tik.scope_ubuf)
        move_num = self.tik_instance.Scalar(
            dtype=Constant.TILING_ARGUMENT_DTYPE, name="move_num", init_value=self.elt_num_per_core
        )
        with self.tik_instance.if_scope(core_id == self.task_num - 1):
            move_num.set_as(self.total_elt_num - start)

        self.__move(ub, self.x_gm, 0, start, move_num)
        self.__move(self.y_gm, ub, start, 0, move_num)

    def _execute_small_matrix(self, core_id):
        """
        _execute_small_matrix executes the case that the matrix is small(<64 elements),
        we use a bitmap to mark the zero elements
        """
        start = core_id * self.elt_num_per_core
        ub = self.tik_instance.Tensor(self.dtype, [self.elt_num_per_core], name="samll_matrix_ub", scope=tik.scope_ubuf)

        move_num = self.tik_instance.Scalar(
            dtype=Constant.TILING_ARGUMENT_DTYPE, name="move_num", init_value=self.elt_num_per_core
        )
        with self.tik_instance.if_scope(core_id == self.task_num - 1):
            move_num.set_as(self.total_elt_num - start)

        self.__move(ub, self.x_gm, 0, start, move_num)
        with self.tik_instance.for_range(0, self.matrix_num_per_core) as matrix_id:
            with self.tik_instance.for_range(0, self.len_mask_bitmap) as i:
                ub[matrix_id * self.elt_num_per_matrix + self.mask_bitmap[i]].set_as(0)
        self.__move(self.y_gm, ub, start, 0, move_num)

    def _execute_small_row(self, core_id):
        """
        _execute_small_row executes the case that the row is small(<32Bytes).
        We first move data to ub, then set_as 0 for every row according to the condition.
        """
        start = core_id * self.elt_num_per_core
        row_start = core_id * self.row_num_per_core
        ub = self.tik_instance.Tensor(self.dtype, [self.elt_num_per_core], name="small_row_ub", scope=tik.scope_ubuf)
        move_num = self.tik_instance.Scalar(
            dtype=Constant.TILING_ARGUMENT_DTYPE, name="move_num", init_value=self.elt_num_per_core
        )
        row_num = self.tik_instance.Scalar(
            dtype=Constant.TILING_ARGUMENT_DTYPE, name="row_num", init_value=self.row_num_per_core
        )
        with self.tik_instance.if_scope(core_id == self.task_num - 1):
            move_num.set_as(self.total_elt_num - start)
            row_num.set_as(self.total_row_num - row_start)
        self.__move(ub, self.x_gm, 0, start, move_num)

        with self.tik_instance.for_range(0, row_num) as offset_row:
            curr_row = row_start + offset_row
            row_id = curr_row % self.row
            if self.upper:
                zero_end = self.tik_instance.Scalar(name="zero_end", init_value=self.col)
                with self.tik_instance.if_scope(zero_end > row_id + self.diagonal):
                    zero_end.set_as(row_id + self.diagonal)
                with self.tik_instance.for_range(0, zero_end) as i:
                    ub[offset_row * self.col + i].set_as(0)
            else:
                zero_start = self.tik_instance.Scalar(name="zero_start", init_value=0)
                with self.tik_instance.if_scope(zero_start < row_id + self.diagonal + 1):
                    zero_start.set_as(row_id + self.diagonal + 1)
                with self.tik_instance.for_range(zero_start, self.col) as i:
                    ub[offset_row * self.col + i].set_as(0)
        self.__move(self.y_gm, ub, start, 0, move_num)

    def _execute_normal(self, core_id):
        """
        _execute_normal executes the case that the row is big enough(>=32Bytes and < Ub_Size).
        We first clear a ub, then move data from input or set_as 0 according to the condition.
        """
        start = core_id * self.elt_num_per_core
        row_start = core_id * self.row_num_per_core
        ub = self.tik_instance.Tensor(self.dtype, [self.elt_num_per_core], name="normal_ub", scope=tik.scope_ubuf)
        self.__init_zero(ub, self.elt_num_per_core)
        move_num = self.tik_instance.Scalar(
            dtype=Constant.TILING_ARGUMENT_DTYPE, name="move_num", init_value=self.elt_num_per_core
        )
        row_num = self.tik_instance.Scalar(
            dtype=Constant.TILING_ARGUMENT_DTYPE, name="row_num", init_value=self.row_num_per_core
        )
        with self.tik_instance.if_scope(core_id == self.task_num - 1):
            move_num.set_as(self.total_elt_num - start)
            row_num.set_as(self.total_row_num - row_start)

        with self.tik_instance.for_range(0, row_num) as offset_row:
            curr_row = row_start + offset_row
            row_id = curr_row % self.row
            move_row_num = self.tik_instance.Scalar(name="move_row_num", init_value=self.col)
            if self.upper:
                with self.tik_instance.if_scope(0 <= row_id + self.diagonal):
                    move_row_num.set_as(self.col - row_id - self.diagonal)
            else:
                with self.tik_instance.if_scope(row_id + self.diagonal + 1 < move_row_num):
                    move_row_num.set_as(row_id + self.diagonal + 1)
            with self.tik_instance.if_scope(move_row_num > 0):
                offset = offset_row * self.col + self.col - move_row_num if self.upper else offset_row * self.col
                aligned_offset = offset // self.elt_num_per_block * self.elt_num_per_block
                self.__move_row(
                    ub,
                    aligned_offset,
                    start + aligned_offset,
                    move_row_num + offset - aligned_offset,
                    offset - aligned_offset,
                )

        self.__move(self.y_gm, ub, start, 0, move_num)

    def _execute_big_row(self, core_id):
        """
        _execute_big_row executes the case that the row is big enough(>= Ub_Size, usually).
        """
        row_id = core_id % self.row
        start = self.tik_instance.Scalar(name="start", init_value=core_id * self.elt_num_per_core)
        elt_num_per_ub = self.tik_instance.Scalar(
            name="elt_num_per_ub",
            init_value=self.available_ub_size // Constant.BYTES_PER_BLOCK * self.elt_num_per_block,
        )
        with self.tik_instance.if_scope(elt_num_per_ub > self.elt_num_per_core):
            elt_num_per_ub.set_as(self.elt_num_per_core // self.elt_num_per_block * self.elt_num_per_block)
        ub = self.tik_instance.Tensor(self.dtype, [elt_num_per_ub], name="big_row_ub", scope=tik.scope_ubuf)
        loop = self.elt_num_per_core // elt_num_per_ub
        this_start = self.tik_instance.Scalar(name="this_start", init_value=0)
        this_end = self.tik_instance.Scalar(name="this_end", init_value=elt_num_per_ub)
        zero_start = self.tik_instance.Scalar(name="move_num", init_value=row_id + self.diagonal + 1)
        zero_end = self.tik_instance.Scalar(name="zero_end", init_value=row_id + self.diagonal)

        def move_big_row_upper():
            with self.tik_instance.if_scope(this_end <= zero_end):
                self.__init_zero(ub, elt_num_per_ub)
            with self.tik_instance.elif_scope(this_start >= zero_end):
                self.__move(ub, self.x_gm, 0, start, elt_num_per_ub)
            with self.tik_instance.else_scope():
                self.__init_zero(ub, elt_num_per_ub)
                move_start = zero_end - this_start
                aligned_move_start = move_start // self.elt_num_per_block * self.elt_num_per_block
                self.__move_row(
                    ub,
                    aligned_move_start,
                    start + aligned_move_start,
                    elt_num_per_ub - aligned_move_start,
                    move_start - aligned_move_start,
                )
            self.__move(self.y_gm, ub, start, 0, elt_num_per_ub)

        def move_big_row_lower():
            with self.tik_instance.if_scope(this_start >= zero_start):
                self.__init_zero(ub, elt_num_per_ub)
            with self.tik_instance.elif_scope(this_end <= zero_start):
                self.__move(ub, self.x_gm, 0, start, elt_num_per_ub)
            with self.tik_instance.else_scope():
                self.__init_zero(ub, elt_num_per_ub)
                self.__move_row(ub, 0, start, zero_start - this_start, 0)
            self.__move(self.y_gm, ub, start, 0, elt_num_per_ub)

        with self.tik_instance.for_range(0, loop):
            move_big_row_upper() if self.upper else move_big_row_lower()
            start.set_as(start + elt_num_per_ub)
            this_start.set_as(this_start + elt_num_per_ub)
            this_end.set_as(this_end + elt_num_per_ub)
        # address back to avoid data stampede
        start.set_as((core_id + 1) * self.elt_num_per_core - elt_num_per_ub)
        this_start.set_as(self.elt_num_per_core - elt_num_per_ub)
        this_end.set_as(self.elt_num_per_core)
        move_big_row_upper() if self.upper else move_big_row_lower()

    def __init_mask_bitmap(self):
        with self.tik_instance.for_range(0, 64) as i:
            with self.tik_instance.if_scope(self.mask == 0):
                self.tik_instance.tik_break()
            with self.tik_instance.if_scope(self.mask % 2 == 1):
                self.mask_bitmap[self.len_mask_bitmap].set_as(i)
                self.len_mask_bitmap.set_as(self.len_mask_bitmap + 1)
            self.mask.set_as(self.mask // 2)

    def __init_zero(self, ub, num):
        """
        __init_zero initializes ub with zero, normally we can use `vector_dup` to do this, however, the `vector_dup`
        does not work when data size comes to 1 byte or 8 bytes. To solve this, we use float16 or float32 to do the
        vector_dup.
        """
        size = self.dsize
        _num = num
        # hack: we directly change the dtype of ub to float16 or float32
        if size == 1:
            ub.dtype, size, _num = "float16", 2, _num // 2
        elif size == 8:
            ub.dtype, size, _num = "float32", 4, _num * 2
        self.__vec_dup_zero(ub, size, _num)
        # restore the dtype
        ub.dtype = self.dtype

    def __vec_dup_zero(self, ub, size, num):
        mask = Constant.VECTOR_MASK_DICT.get(size, 64)
        repeat = num // mask
        tail_num = num % mask
        loop = repeat // Constant.VECTOR_DUP_MAX_REPEAT_TIMES
        tail_repeat = repeat % Constant.VECTOR_DUP_MAX_REPEAT_TIMES
        with self.tik_instance.for_range(0, loop) as i:
            self.tik_instance.vector_dup(
                mask,
                ub[i * mask * Constant.VECTOR_DUP_MAX_REPEAT_TIMES],
                0,
                Constant.VECTOR_DUP_MAX_REPEAT_TIMES,
                1,
                8,
            )
        with self.tik_instance.if_scope(tail_repeat > 0):
            self.tik_instance.vector_dup(
                mask,
                ub[loop * mask * Constant.VECTOR_DUP_MAX_REPEAT_TIMES],
                0,
                tail_repeat,
                1,
                8,
            )
        with self.tik_instance.if_scope(tail_num > 0):
            self.tik_instance.vector_dup(tail_num, ub[num - tail_num], 0, 1, 0, 0)

    def __move_row_aligned(self, ub, ub_start, gm_start, move_num):
        """
        __move_row_aligned move aligned data from gm to ub, no need to resolve conflict
        Parameters:
            - ub: the ub to store data
            - ub_start: the start position of ub
            - gm_start: the start position of gm
            - move_num: the number of elements to move
        """
        burst = move_num // self.elt_num_per_block
        mask = move_num % self.elt_num_per_block
        with self.tik_instance.if_scope(burst > 0):
            self.tik_instance.data_move(ub[ub_start], self.x_gm[gm_start], 0, 1, burst, 0, 0)
        with self.tik_instance.if_scope(mask > 0):
            self.__move_pad(
                ub, ub_start + burst * self.elt_num_per_block, gm_start + burst * self.elt_num_per_block, mask
            )

    def __move_pad(self, ub, ub_start, gm_start, mask):
        if self.is_support_data_move_pad:
            self.tik_instance.data_move_pad(
                ub[ub_start],
                self.x_gm[gm_start],
                1,
                mask * self.dsize,
                0,
                0,
                self.elt_num_per_block - mask,
                0,
                padding_value=0,
            )
        else:
            self.tik_instance.data_move(ub[ub_start], self.x_gm[gm_start], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(mask, self.elt_num_per_block) as i:
                ub[ub_start + i].set_as(0)

    # 'pylint: disable=too-many-arguments, huawei-too-many-arguments
    def __move_row(self, ub, ub_start, gm_start, move_num, conflict_num):
        """
        __move_row moves a whole row from x_gm to given ub.
        Parameters:
            - ub: the ub to store the data
            - ub_start: the start position of ub
            - gm_start: the start position of x_gm
            - move_num: the number of elements to move
            - conflict_num: the number of elements to be overwritten
        """
        # backup the unaligned data
        ub_tmp_4_unaligned = self.tik_instance.Tensor(
            self.dtype, [self.elt_num_per_block], name="ub_tmp_4_unaligned", scope=tik.scope_ubuf
        )
        self.tik_instance.data_move(ub_tmp_4_unaligned, ub[ub_start], 0, 1, 1, 0, 0)
        # move the aligned data
        self.__move_row_aligned(ub, ub_start, gm_start, move_num)
        # restore the overwritten data
        with self.tik_instance.for_range(0, conflict_num) as i:
            ub[ub_start + i].set_as(ub_tmp_4_unaligned[i])

    # 'pylint: disable=too-many-arguments, huawei-too-many-arguments
    def __move(self, dst: tik.Tensor, src: tik.Tensor, dst_start, src_start, move_num):
        """
        __move moves src to dst, the ub's max size is 256KB(which is 256 * 1024 / 32 = 8192 blocks)
        Parameters:
            - dst: the destination
            - src: the source
            - dst_start: the start position of dst
            - src_start: the start position of src
            - move_num: the number of elements to move, must be 32B aligned
        """
        if not self.is_support_data_move_pad:
            burst = self.tik_instance.Scalar(
                Constant.TILING_ARGUMENT_DTYPE, name="burst", init_value=ceil_div(move_num, self.elt_num_per_block)
            )  # unit: 32Bytes
            self.tik_instance.data_move(dst[dst_start], src[src_start], 0, 1, burst, 0, 0)
        else:
            burst = self.tik_instance.Scalar(
                Constant.TILING_ARGUMENT_DTYPE, name="burst", init_value=move_num * self.dsize
            )  # unit: Byte
            self.tik_instance.data_move_pad(dst[dst_start], src[src_start], 1, burst, 0, 0)
