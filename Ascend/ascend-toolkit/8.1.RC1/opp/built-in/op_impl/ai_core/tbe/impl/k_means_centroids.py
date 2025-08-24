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
k_means_centroids
"""
import functools

from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform

FP16_TYPE = 2
FP32_TYPE = 4
UB_BLOCK_SIZE = 32
L0C_BLOCK_SIZE = 1024
SCALAR_MAX_FP32 = (2 ** 30 + 2 ** 29)
VECTOR_REPEAT_MAX = 255
FP16_MASK = 128
FP32_MASK = 64
INPUT_LENGTH = 2
VECTOR_LENGTH = 128
EXTEND_LENGTH = 2
VNCHWCONV_MIN_SIZE = 256
MAX_K_SIZE = 1792
INT64_SIZE = 2 ** 63 - 1
INDEX_DTYPE = "int32"
MASK_DTYPE = "uint64"
M_DIM_NET = 4096
N_DIM_NET = 1048576
K_DIM_NET_1 = 128
K_DIM_NET_2 = 32
K_DIM_NET_3 = 64


class KMeansCentroids:
    """
    class of k-means centroids operator
    """
    def __init__(self, para_dict):
        """
        init input, output, platform,
            tik instance, tiling and tensor

        Parameters:
        ----------
        para_dict: inputs and outputs parameters, dict
        """
        self.input_x1 = para_dict.get("x")
        self.input_x2 = para_dict.get("y")
        self.input_x3 = para_dict.get("sum_square_x")
        self.input_x4 = para_dict.get("sum_square_y")
        self.output_y1 = para_dict.get("segment_sum")
        self.output_y2 = para_dict.get("segment_count")
        self.output_y3 = para_dict.get("total_distance")
        self.aic_cnt = 0
        self.use_actual_distance = para_dict.get("use_actual_distance")
        self.kernel_name = para_dict.get("kernel_name")
        impl_mode = para_dict.get("impl_mode")
        self.high_perf = (impl_mode == "high_performance")
        self.vcmin_fp32_supported = tbe_platform.api_check_support("tik.vcmin", "float32")
        self._get_platform_info()
        self.tik_instance = tik.Tik()
        self.tiling = {}
        self._get_tiling()

        
        """ init loop, left and shape in different buffer
        """
        # matmul
        self.m_each_core = 16
        self.m_last_core = 0
        self.n_each_core = 16
        self.n_last_core = 0
        self.m_tiling_loop = 0
        self.m_tiling_left = 0
        self.m_last_tiling_loop = 0
        self.m_last_tiling_left = 0
        self.n_tiling_loop = 0
        self.n_tiling_left = 0
        self.m_tiling_ub_loop = 1
        self.n_tiling_ub_loop = 1
        self.n_tiling_cub_loop = 0
        self.n_tiling_cub_left = 0
        self.shape_x_ub = (16, 16)
        self.shape_x_ub_trans = (1, 16, 16)
        self.shape_y_ub = (16, 16)
        self.shape_y_ub_trans = (1, 16, 16)
        self.shape_x_l1 = (1, 16, 16)
        self.shape_y_l1 = (1, 16, 16)
        self.shape_x_l0a = (1, 1, 16, 16)
        self.shape_y_l0b = (1, 1, 16, 16)
        self.shape_z_l0c = (1, 16, 16)
        self.shape_z_ub = (1, 16, 16)
        self.shape_z_ub_extend = (1, 17, 16)
        self.shape_z_ub_nd = (16, 16)
        # argmin
        self.m_tiling = 16
        self.n_tiling = 16
        self.shape_input_3_ub = (16, 1)
        self.shape_input_4_ub = (1, 16)
        self.shape_broadcast_ub = (16, 16)
        self.shape_broadcast_ub_extend = (17, 16)
        self.shape_global_min_distance_ub = (16,)
        self.shape_total_distance = (1,)

        
        self._tiling_process()
        self._init_tensor()

    @staticmethod
    def _ceil(x1, x2):
        if x2 == 0:
            reason = "Division by zero."
            error_manager_cube.raise_err_message_cube("k_means_centroids", reason)
        return (x1 + x2 - 1) // x2

    @staticmethod
    def _elecnt_of_shape(shape):
        """ calculate reduce shape
        """
        return functools.reduce(lambda x, y: x * y, shape)

    @staticmethod
    def _get_factors(num):
        factor_list = []
        upper = num // 2 + 1
        for facotr in range(1, upper):
            if num % facotr == 0:
                factor_list.append(facotr)
        factor_list.append(num)
        return factor_list

    @staticmethod
    def _check_tiling_key(tiling_dict: dict, keys: list) -> None:
        """
        check tiling key

        Parameters
        ----------
        tiling_dict : tiling parameters, dict
        keys : target keys, tuple or list

        Returns
        -------
        None
        """
        for key in keys:
            if key not in tiling_dict:
                reason = "Key error, %s not in tiling." % key
                error_manager_cube.raise_err_message_cube("k_means_centroids", reason)

    def k_means_centroids_compute(self):
        """
        MAIN function of k-means centroids operator

        Parameters:
        -------------
        None

        Returns:
        -------------
        tik_instance: tik instance
        """
        with self.tik_instance.for_range(0, self.aic_cnt, block_num=self.aic_cnt) as blk_idx:
            self._compute_one_core(blk_idx)

        if self.use_actual_distance:
            inputs = [self.data_input_gm_1, self.data_input_gm_2,
                      self.data_input_gm_4, self.data_input_gm_3]
        else:
            inputs = [self.data_input_gm_1, self.data_input_gm_2, self.data_input_gm_4]

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=inputs,
            outputs=[self.data_output_gm_1, self.data_output_gm_2, self.data_output_gm_3],
        )

        return self.tik_instance

    def vconv(self, dst, src, shape_t, src_dtype="float32"):
        """
        transfer data type fp32 to fp16, vice versa

        Parameters:
        -------------
        dst: dst tensor, fp16
        src: src tensor, fp32
        shape_t: tensor shape, tuple
        src_dtype: default fp32

        Returns:
        -------------
        None
        """
        n = shape_t[1]
        repeat = (shape_t[0] * n) // FP32_MASK
        left = (shape_t[0] * n) % FP32_MASK
        repeat_loop = repeat // VECTOR_REPEAT_MAX
        repeat_left = repeat % VECTOR_REPEAT_MAX
        if src_dtype == "float32":
            dst_rep_stride = 4
            src_rep_stride = 8
        else:
            dst_rep_stride = 8
            src_rep_stride = 4
        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * VECTOR_REPEAT_MAX * FP32_MASK
                self.tik_instance.vconv(FP32_MASK, "", dst[offset // n, offset % n],
                                        src[offset // n, offset % n], VECTOR_REPEAT_MAX,
                                        1, 1, dst_rep_stride, src_rep_stride)

        if repeat_left > 0:
            offset = repeat_loop * VECTOR_REPEAT_MAX * FP32_MASK
            self.tik_instance.vconv(FP32_MASK, "", dst[offset // n, offset % n], src[offset // n, offset % n],
                                    repeat_left, 1, 1, dst_rep_stride, src_rep_stride)

        if left > 0:
            offset = repeat * FP32_MASK
            self.tik_instance.vconv(left, "", dst[offset // n, offset % n], src[offset // n, offset % n],
                                    1, 1, 1, dst_rep_stride, src_rep_stride)

    def nd_to_zn_3d(self, dst, src, shape_x_trans):
        """
        reshape tensor from format ND to Zn/Nz in ub buffer using vnchwconv

        Parameters:
        -------------
        dst: tensor (m1, k, m0), fp16
        src: tensor (m, k), fp16
        shape_x_trans: result shape, tuple

        Returns:
        -------------
        None
        """
        dst_high_half = False
        src_high_half = False
        m1 = shape_x_trans[0]
        k = shape_x_trans[1]
        repeat_times = self._elecnt_of_shape(shape_x_trans) // m1 // 256
        dst_rep_stride = 16
        src_rep_stride = 1
        if k <= 16:
            dst_rep_stride = 0
            src_rep_stride = 0
        with self.tik_instance.for_range(0, m1) as m1_idx:
            dst_list = [dst[m1_idx * repeat_times * 256 + 16 * i] for i in range(16)]
            src_list = [src[m1_idx * k * 16 + k * i] for i in range(16)]
            self.tik_instance.vnchwconv(dst_high_half, src_high_half, dst_list, src_list,
                                        repeat_times, dst_rep_stride, src_rep_stride)

    def zn_to_zz(self, dst, src, shape_x_l0a):
        """
        reshape tensor_a from format Zn to Zz using load2d

        Parameters:
        -------------
        dst: tensor (m1, k1, m0, k0), fp16
        src: tensor (m1, k, m0), fp16
        shape_x_l0a: result shape, tuple

        Returns:
        -------------
        None
        """
        index = 0
        repeat_times = self._elecnt_of_shape(shape_x_l0a) // 256
        dst_gap = 0
        src_stride = 1
        sid = 0
        if_transpose = True
        load2dv2_supported = tbe_platform.api_check_support("tik.load2dv2")
        if load2dv2_supported:
            self.tik_instance.load2dv2(dst, src, index, repeat_times,
                                       dst_gap, src_stride, sid, if_transpose)
        else:
            self.tik_instance.load2dv1(dst, src, index, repeat_times,
                                       src_stride, sid, if_transpose)

    def nz_to_zn(self, dst, src, shape_y_l0b):
        """
        reshape tensor_b from format Nz to Zn using load2d

        Parameters:
        -------------
        dst: tensor (k1, n1, n0, k0), fp16
        src: tensor (n1, k, n0), fp16
        shape_y_l0b: result shape, tuple

        Returns:
        -------------
        None
        """
        k1 = shape_y_l0b[0]
        repeat_times = self._elecnt_of_shape(shape_y_l0b) // k1 // 256
        dst_gap = 0
        src_stride = k1
        sid = 0
        if_transpose = True
        load2dv2_supported = tbe_platform.api_check_support("tik.load2dv2")
        if load2dv2_supported:
            with self.tik_instance.for_range(0, k1) as index:
                self.tik_instance.load2dv2(dst[index, :, :, :], src,
                                           index, repeat_times, dst_gap, src_stride, sid, if_transpose)
        else:
            with self.tik_instance.for_range(0, k1) as index:
                self.tik_instance.load2dv1(dst[index, :, :, :], src,
                                           index, repeat_times, src_stride, sid, if_transpose)

    def _get_platform_info(self):
        """ get platform information, such as CORE_NUM
        """
        self.aic_cnt = tbe_platform.get_soc_spec("CORE_NUM")

    def _get_tiling(self):
        """ native get_tiling api
        """
        shape_x1 = self.input_x1.get("shape")
        shape_x2 = self.input_x2.get("shape")
        m_dim, k_dim = shape_x1
        n_dim, _ = shape_x2
        tiling_para = {
            "l0_m": 16,
            "l0_n": 8,
            "aub_m": 16,
            "bub_n": 8,
            "cub_n": 4,
            "k1": k_dim // 16,
            "block_dim_m": 1
        }

        self.cube_unit = 16
        is_large_net_case = ((k_dim in (K_DIM_NET_1, K_DIM_NET_2, K_DIM_NET_3)) and
            (m_dim >= M_DIM_NET) and (n_dim >= N_DIM_NET))
        if is_large_net_case:
            tiling_para = self._default_tiling(k_dim, tiling_para)
        else:
            tiling_para = self._formula_tiling(m_dim, k_dim, n_dim)
        self._fillin_tiling(tiling_para)

    def _default_tiling(self, k_dim: int, tiling_para: dict) -> dict:
        """
        default tiling

        Parameters
        ----------
        k_dim : axis k of original shape, number
        tiling_para : tiling parameters, dict

        Returns
        -------
        tiling_para : modified tiling parameters, dict
        """
        if not self.vcmin_fp32_supported and not self.high_perf:
            if k_dim == K_DIM_NET_1:
                tiling_para = {
                    "l0_m": 16,
                    "l0_n": 14,
                    "aub_m": 16,
                    "bub_n": 14,
                    "cub_n": 7,
                    "k1": k_dim // 16,
                    "block_dim_m": 1
                }
            elif k_dim == K_DIM_NET_2:
                tiling_para = {
                    "l0_m": 4,
                    "l0_n": 30,
                    "aub_m": 4,
                    "bub_n": 30,
                    "cub_n": 30,
                    "k1": k_dim // 16,
                    "block_dim_m": 1
                }
        return tiling_para

    def _formula_tiling(self, m_dim, k_dim, n_dim):
        """
        tiling parameters calculated by formula

        Parameters
        ----------
        m_dim : axis m of original shape, number
        k_dim : axis k of original shape, number
        n_dim : axis n of original shape, number

        Returns
        -------
        tiling_para : modified tiling parameters, dict
        """
        m_div_16 = m_dim // self.cube_unit
        k_div_16 = k_dim // self.cube_unit
        n_div_16 = n_dim // self.cube_unit
        block_dim_m = m_div_16 if m_div_16 < self.aic_cnt else self.aic_cnt
        single_core_m = self._ceil(m_div_16, block_dim_m) * self.cube_unit
        mkn_single_core = single_core_m // self.cube_unit, k_div_16, n_div_16
        l0_mn_list = self._get_l0_tiling(mkn_single_core)
        aub_m, bub_n, cub_n = self._get_mkn_ub(l0_mn_list[0], l0_mn_list[1], k_div_16)
        tiling_para = {
            "l0_m": l0_mn_list[0],
            "l0_n": l0_mn_list[1],
            "aub_m": aub_m,
            "bub_n": bub_n,
            "cub_n": cub_n,
            "k1": k_div_16,
            "block_dim_m": 1
        }
        return tiling_para

    def _fillin_tiling(self, tiling_para: dict) -> None:
        """
        filling tiling parameters in a dict

        Parameters
        ----------
        tiling_para : partial tiling parameters, dict

        Returns
        -------
        None
        """
        l0_m = tiling_para.get('l0_m')
        l0_n = tiling_para.get('l0_n')
        aub_m = tiling_para.get('aub_m')
        bub_n = tiling_para.get('bub_n')
        cub_n = tiling_para.get('cub_n')
        k1 = tiling_para.get('k1')
        block_dim_m = tiling_para.get('block_dim_m')
        if l0_n > bub_n:
            l0_n = bub_n
            cub_n = min(cub_n, bub_n)
        self.tiling = {
            "block_dim": [1, 1, block_dim_m, 1],
            "AUB_shape": [k1 * self.cube_unit, aub_m * self.cube_unit, 1, 1],
            "BUB_shape": [k1 * self.cube_unit, bub_n * self.cube_unit, 1, 1],
            "AL1_shape": [k1 * self.cube_unit, 1, 1, 1],
            "BL1_shape": [k1 * self.cube_unit, 1, 1, 1],
            "AL0_matrix": [l0_m, k1, 16, 16, 1, 1],
            "BL0_matrix": [k1, l0_n, 16, 16, 1, 1],
            "CL0_matrix": [l0_n, l0_m, 16, 16, 1, 1],
            "CUB_matrix": [cub_n, l0_m, 16, 16, 1, 1],
            "manual_pingpong_buffer": {
                "AUB_pbuffer": 1,
                "BUB_pbuffer": 1,
                "AL1_pbuffer": 1,
                "BL1_pbuffer": 1,
                "AL0_pbuffer": 1,
                "BL0_pbuufer": 1,
                "CL0_pbuffer": 1,
                "CUB_pbuffer": 1,
                "UBG_pbuffer": 1
            }
        }

    def _get_l0_tiling(self, mkn_single_core):
        """
        get l0_buffer tiling

        Parameters
        ----------
        mkn_single_core : axis m, k and n in single ai-core, tuple

        Returns
        -------
        target_mn : l0_buffer tiling, list
        """
        l0a_buffer_size = tbe_platform.get_soc_spec("L0A_SIZE")
        l0c_buffer_size = tbe_platform.get_soc_spec("L0C_SIZE")
        l0a_m_max = l0a_buffer_size // FP16_TYPE // mkn_single_core[1] // self.cube_unit // self.cube_unit
        l0b_n_max = l0a_m_max
        min_cost = float('inf')
        target_mn = [1, 1]
        if not self.vcmin_fp32_supported and self.high_perf:
            # Set m_tiling to be smaller than or equal to 16*16,
            # because some tensors, such as vcmin_input_fp16, are set as a fixed value 256.
            l0a_m_max = min(l0a_m_max, 16)
        for l0_m in range(1, l0a_m_max + 1):
            l0c_n_max = l0c_buffer_size // FP32_TYPE // l0_m // self.cube_unit // self.cube_unit
            m_loop = self._ceil(mkn_single_core[0], l0_m)
            for l0_n in range(1, min(l0b_n_max, l0c_n_max) + 1):
                cur_cost = self._calc_cost(l0_m, l0_n, m_loop, mkn_single_core)
                if cur_cost < min_cost:
                    min_cost = cur_cost
                    target_mn = [l0_m, l0_n]
        return target_mn

    def _calc_cost(self, l0_m: int, l0_n: int, m_loop: int, mkn_single_core: tuple) -> int:
        """
        calculate cost of tiling optimization function

        Parameters
        ----------
        l0_m : axis m in l0_buffer, number
        l0_n : axis n in l0_buffer, number
        m_loop : loop of axis m in single ai-core, number
        mkn_single_core : axis m, k and n in single ai-core, tuple

        Returns
        -------
        cost, number
        """
        bandwidth = 22.17
        n_loop = self._ceil(mkn_single_core[2], l0_n)
        n_one_loop_cost = self._calc_n_one_loop_cost(l0_m, l0_n)
        a_load_size = l0_m * mkn_single_core[1] * self.cube_unit * self.cube_unit * FP32_TYPE
        b_load_size = l0_n * mkn_single_core[1] * self.cube_unit * self.cube_unit * FP32_TYPE
        a_load_time = a_load_size / bandwidth
        b_load_time = b_load_size / bandwidth
        return m_loop * (a_load_time + b_load_time + n_loop * n_one_loop_cost)

    def _get_mkn_ub(self, l0_m, l0_n, k_div_16):
        """
        get m, k, and n in ub_buffer

        Parameters
        ----------
        l0_m : axis m in l0_buffer, number
        l0_n : axis n in l0_buffer, number
        k_div_16 : axis k divided by 16 of original shape, number

        Returns
        -------
        aub_m : axis m in ub_buffer before CUBE unit, number
        bub_n : axis n in ub_buffer before CUBE unit, number
        cub_n : axis n in ub_buffer after CUBE unit, number
        """
        ub_buffer_size = tbe_platform.get_soc_spec("UB_SIZE")
        global_ub = l0_m * self.cube_unit * 2 * FP32_TYPE
        global_ub += 8 * 2 * FP32_TYPE
        bub_size = ub_buffer_size - global_ub
        l0_m_factor_lis = sorted(self._get_factors(l0_m), reverse=True)
        l0_n_factor_lis = sorted(self._get_factors(l0_n), reverse=True)
        aub_m = 1
        bub_n = 1

        for aub_m in l0_m_factor_lis:
            aub_tensor_count = 4
            tensor_size = aub_m * k_div_16 * aub_tensor_count * self.cube_unit * self.cube_unit * FP16_TYPE
            if tensor_size <= ub_buffer_size:
                break

        for bub_n in l0_n_factor_lis:
            bub_tensor_count = 4
            tensor_size = bub_n * k_div_16 * bub_tensor_count * self.cube_unit * self.cube_unit * FP16_TYPE
            if tensor_size <= bub_size:
                break

        cub_n = self._calc_cub_n(l0_n_factor_lis, l0_m, global_ub, ub_buffer_size)

        return aub_m, bub_n, cub_n

    def _calc_cub_n(self, l0_n_factor_lis: list, l0_m: int, global_ub: int, ub_buffer_size: int) -> int:
        """
        calculate axis n in cub_buffer

        Parameters
        ----------
        l0_n_factor_lis : axis n's factor list in l0_buffer, list
        l0_m : axis m in l0_buffer, number
        global_ub : global ub_buffer occupation, number
        ub_buffer_size : total ub_buffer size

        Returns
        -------
        cub_n : axis n in ub_buffer after CUBE unit, number
        """
        cub_n = 1
        for cub_n in l0_n_factor_lis:
            matmul_tensor_count = 2
            matmul_out = matmul_tensor_count * FP32_TYPE * cub_n * l0_m * self.cube_unit * self.cube_unit
            local_tensor_count = 5
            local_size = local_tensor_count * FP32_TYPE * l0_m * self.cube_unit
            if not self.vcmin_fp32_supported and self.high_perf:
                extra_size = VNCHWCONV_MIN_SIZE * cub_n * self.cube_unit * FP16_TYPE
                # VNCHWCONV_MIN_SIZE * 11 is the total size of extra tensors in high_performance mode
                extra_size += VNCHWCONV_MIN_SIZE * 11 * FP16_TYPE
                extra_size += VNCHWCONV_MIN_SIZE * FP32_TYPE
                local_size += extra_size
            sum_square_y_size = cub_n * self.cube_unit * FP32_TYPE
            tensor_size = matmul_out + local_size + sum_square_y_size + global_ub
            if tensor_size <= ub_buffer_size:
                break
        return cub_n

    def _calc_n_one_loop_cost(self, l0_m, l0_n):
        """
        calculate cost of axis n's one loop

        Parameters
        ----------
        l0_m : axis m in l0_buffer, number
        l0_n : axis n in l0_buffer, number

        Returns
        -------
        sum_cost : cost of axis n's one loop, number
        """
        fp32_parallelism = 64
        mn_cost = self._ceil(l0_m * l0_n, fp32_parallelism)
        if self.vcmin_fp32_supported or self.high_perf:
            mn_count = 4
            m_cost = self._ceil(l0_m, fp32_parallelism)
            m_count = 7
            sum_cost = mn_cost * mn_count + m_cost * m_count
            return sum_cost

        one_loop_unit = 8
        mn_count = 3
        get_min_loop = (l0_m * self.cube_unit) // one_loop_unit
        cmpare_cost = 433
        sum_cost = mn_cost * mn_count + get_min_loop * cmpare_cost
        return sum_cost

    def _tiling_process(self):
        """
        process tiling in multi-core level,
            one-core level and buffer level
        """
        self._tiling_multi_core()
        self._tiling_one_core_matmul()
        self._tiling_one_core_argmin()

    def _tiling_multi_core(self):
        """ tiling in multi-core level
        """
        shape_x1 = self.input_x1.get("shape")
        shape_x2 = self.input_x2.get("shape")
        self._check_tiling_key(self.tiling,
            ["AUB_shape", "block_dim", "AL0_matrix", "CL0_matrix",
             "CUB_matrix", "BUB_shape", "BL0_matrix"])
        if shape_x1[0] < self.aic_cnt * self.tiling["AUB_shape"][1]:
            self.aic_cnt = self._ceil(shape_x1[0], self.tiling["AUB_shape"][1])
        # m axis bounds multi-core
        self.tiling["block_dim"][2] = self.aic_cnt

        if shape_x1[0] < self.tiling["AUB_shape"][1]:
            self.tiling["AUB_shape"][1] = shape_x1[0]
            self.tiling["AL0_matrix"][0] = self._ceil(shape_x1[0], 16)
            self.tiling["CL0_matrix"][1] = self._ceil(shape_x1[0], 16)
            if self._ceil(shape_x1[0], 16) < self.tiling["CUB_matrix"][1]:
                self.tiling["CUB_matrix"][1] = self._ceil(shape_x1[0], 16)
            else:
                self.tiling["CUB_matrix"][1] = min(self.tiling["CUB_matrix"][1],
                                                   self._ceil(shape_x1[0], 16) // 2)
        if shape_x2[0] < self.tiling["BUB_shape"][1]:
            self.tiling["BUB_shape"][1] = shape_x2[0]
            self.tiling["BL0_matrix"][1] = self._ceil(shape_x2[0], 16)
            self.tiling["CL0_matrix"][0] = self._ceil(shape_x2[0], 16)
            if self._ceil(shape_x2[0], 16) < self.tiling["CUB_matrix"][0]:
                self.tiling["CUB_matrix"][0] = self._ceil(shape_x2[0], 16)
            else:
                self.tiling["CUB_matrix"][0] = min(self.tiling["CUB_matrix"][0],
                                                   self._ceil(shape_x2[0], 16) // 2)

        m_dim = self.tiling["block_dim"][2]
        n_dim = self.tiling["block_dim"][1]
        self.m_each_core = max(self.tiling["AUB_shape"][1], self._ceil(shape_x1[0], m_dim))
        self.m_last_core = shape_x1[0] % self.m_each_core
        self.n_each_core = self._ceil(shape_x2[0], n_dim)
        self.n_last_core = shape_x2[0] % self.n_each_core

    def _tiling_one_core_matmul(self):
        """ matmul tiling in each core
        """
        self._check_tiling_key(self.tiling,
            ["AUB_shape", "BUB_shape", "AL1_shape", "AL0_matrix",
             "BL1_shape", "BL0_matrix", "CL0_matrix", "CUB_matrix"])
        m_aub = self.tiling["AUB_shape"][1]
        n_bub = self.tiling["BUB_shape"][1]
        m_al1 = self.tiling["AL1_shape"][1] * self.tiling["AL0_matrix"][0] * self.tiling["AL0_matrix"][2]
        n_bl1 = self.tiling["BL1_shape"][1] * self.tiling["BL0_matrix"][1] * self.tiling["BL0_matrix"][2]
        ma, ka = self.tiling["AL0_matrix"][:2]
        kb, nb = self.tiling["BL0_matrix"][:2]
        nc, mc = self.tiling["CL0_matrix"][:2]
        nc_factor, mc_factor = self.tiling["CUB_matrix"][:2]

        self.m_tiling_loop = self.m_each_core // m_aub
        self.m_tiling_left = self.m_each_core % m_aub
        if self.m_last_core > 0:
            self.m_last_tiling_loop = self.m_last_core // m_aub
            self.m_last_tiling_left = self.m_last_core % m_aub
        self.n_tiling_loop = self.n_each_core // n_bub
        self.n_tiling_left = self.n_each_core % n_bub

        self.m_tiling_ub_loop = mc // mc_factor
        self.n_tiling_ub_loop = nc // nc_factor
        if self.n_tiling_left > 0:
            self.n_tiling_cub_loop = self.n_tiling_left // (nc_factor * 16)
            self.n_tiling_cub_left = self.n_tiling_left % (nc_factor * 16)

        self.shape_x_ub = (m_aub, self.tiling["AUB_shape"][0])
        self.shape_x_ub_trans = (self._ceil(m_aub, 16), self.tiling["AUB_shape"][0], 16)
        self.shape_y_ub = (n_bub, self.tiling["BUB_shape"][0])
        self.shape_y_ub_trans = (self._ceil(n_bub, 16), self.tiling["BUB_shape"][0], 16)

        self.shape_x_l1 = (self._ceil(m_al1, 16), self.tiling["AL1_shape"][0], 16)
        self.shape_y_l1 = (self._ceil(n_bl1, 16), self.tiling["BL1_shape"][0], 16)
        self.shape_x_l0a = (ma, ka, 16, 16)
        self.shape_y_l0b = (kb, nb, 16, 16)
        self.shape_z_l0c = (nc, mc * 16, 16)

        self.shape_z_ub = (nc_factor, mc_factor * 16, 16)
        self.shape_z_ub_extend = (nc_factor, mc_factor * 16 + 1, 16)
        self.shape_z_ub_nd = (mc_factor * 16, nc_factor * 16)

    def _tiling_one_core_argmin(self):
        """ argmin and UnsortedSegmentSum tiling in each core
        """
        self._check_tiling_key(self.tiling, ["CUB_matrix"])
        nc_factor, mc_factor, m0, n0 = self.tiling["CUB_matrix"][:4]
        self.m_tiling = mc_factor * m0
        self.n_tiling = nc_factor * n0
        self.shape_input_3_ub = (self.m_tiling, 1)
        self.shape_input_4_ub = (1, self.n_tiling)
        self.shape_broadcast_ub = (self.m_tiling, self.n_tiling)
        self.shape_broadcast_ub_extend = (self.m_tiling + 1, self.n_tiling)
        self.shape_global_min_distance_ub = (self.m_tiling * self.m_tiling_ub_loop,)

    def _init_tensor(self):
        """ init fixed-shape tensor
        """
        shape_x1 = self.input_x1.get("shape")
        shape_x2 = self.input_x2.get("shape")
        shape_x4 = self.input_x4.get("shape")
        self.input_dtype = self.input_x1.get("dtype")
        output_dtype = self.output_y1.get("dtype")
        n_gm, d_gm = shape_x2
        self.ub_min_num = UB_BLOCK_SIZE // FP32_TYPE

        self.data_input_gm_1 = self.tik_instance.Tensor(self.input_dtype, shape_x1,
                                                        name="data_input_gm_1", scope=tik.scope_gm)
        self.data_input_gm_2 = self.tik_instance.Tensor(self.input_dtype, shape_x2,
                                                        name="data_input_gm_2", scope=tik.scope_gm)
        if self.use_actual_distance:
            shape_x3 = self.input_x3.get("shape")
            self.data_input_gm_3 = self.tik_instance.Tensor(self.input_dtype, shape_x3,
                                                            name="data_input_gm_3", scope=tik.scope_gm)
        self.data_input_gm_4 = self.tik_instance.Tensor(self.input_dtype, shape_x4,
                                                        name="data_input_gm_4", scope=tik.scope_gm)
        self.data_output_gm_1 = self.tik_instance.Tensor(output_dtype, (n_gm, d_gm),
                                                         name="data_output_gm_1", scope=tik.scope_gm,
                                                         is_atomic_add=True)
        self.data_output_gm_2 = self.tik_instance.Tensor(output_dtype, (n_gm, 1),
                                                         name="data_output_gm_2", scope=tik.scope_gm,
                                                         is_atomic_add=True)
        self.data_output_gm_3 = self.tik_instance.Tensor(output_dtype, self.shape_total_distance,
                                                         name="data_output_gm_3", scope=tik.scope_gm,
                                                         is_atomic_add=True)

        self.data_input_l1_1 = self.tik_instance.Tensor("float16", self.shape_x_l1,
                                                        name="data_input_l1_1", scope=tik.scope_cbuf)
        self.data_input_l1_2 = self.tik_instance.Tensor("float16", self.shape_y_l1,
                                                        name="data_input_l1_2", scope=tik.scope_cbuf)
        self.data_input_l0a = self.tik_instance.Tensor("float16", self.shape_x_l0a,
                                                       name="data_input_l0a", scope=tik.scope_ca)
        self.data_input_l0b = self.tik_instance.Tensor("float16", self.shape_y_l0b,
                                                       name="data_input_l0b", scope=tik.scope_cb)
        self.data_output_l0c = self.tik_instance.Tensor(output_dtype, self.shape_z_l0c,
                                                        name="data_output_l0c", scope=tik.scope_cc)

    def _init_matmul_tensor_a_ub(self):
        """ init tensor_a of matmul in ub buffer
        """
        self.data_input_ub_1 = self.tik_instance.Tensor("float32", self.shape_x_ub,
                                                        name="data_input_ub_1", scope=tik.scope_ubuf)
        self.data_input_ub_1_fp16 = self.tik_instance.Tensor("float16", self.shape_x_ub,
                                                             name="data_input_ub_1_fp16", scope=tik.scope_ubuf)
        self.data_input_ub_1_trans = self.tik_instance.Tensor("float16", self.shape_x_ub_trans,
                                                              name="data_input_ub_1_trans", scope=tik.scope_ubuf)

    def _init_matmul_tensor_a_ub_db(self, shape_x_ub, shape_x_ub_trans, double_buffer):
        """
        double buffer strategy

        Parameters:
        -------------
        shape_x_ub: (m, k)
        shape_x_ub_trans: (m1, k, m0)
        double_buffer: support 1,2,4,8

        Returns:
        -------------
        None
        """
        self.data_input_ub_1 = self.tik_instance.Tensor("float32", self.shape_x_ub,
                                                        name="data_input_ub_1", scope=tik.scope_ubuf)
        self.data_input_ub_1_fp16_list = []
        self.data_input_ub_1_trans_list = []
        for db_idx in range(double_buffer):
            tensor_name1 = "data_input_ub_1_fp16_%d" % (db_idx + 1)
            tensor_name2 = "data_input_ub_1_trans_%d" % (db_idx + 1)
            tensor_ins1 = self.tik_instance.Tensor("float16", shape_x_ub,
                                                   name=tensor_name1, scope=tik.scope_ubuf)
            tensor_ins2 = self.tik_instance.Tensor("float16", shape_x_ub_trans,
                                                   name=tensor_name2, scope=tik.scope_ubuf)
            self.data_input_ub_1_fp16_list.append(tensor_ins1)
            self.data_input_ub_1_trans_list.append(tensor_ins2)

    def _init_matmul_tensor_b_ub(self):
        """ init tensor_b of matmul in ub buffer
        """
        self.data_input_ub_2 = self.tik_instance.Tensor("float32", self.shape_y_ub,
                                                        name="data_input_ub_2", scope=tik.scope_ubuf)
        self.data_input_ub_2_fp16 = self.tik_instance.Tensor("float16", self.shape_y_ub,
                                                             name="data_input_ub_2_fp16", scope=tik.scope_ubuf)
        self.data_input_ub_2_trans = self.tik_instance.Tensor("float16", self.shape_y_ub_trans,
                                                              name="data_input_ub_2_trans", scope=tik.scope_ubuf)

    def _init_matmul_tensor_b_ub_db(self, shape_y_ub, shape_y_ub_trans, double_buffer):
        """
        double buffer strategy

        Parameters:
        -------------
        shape_y_ub: (n, k)
        shape_y_ub_trans: (n1, k, n0)
        double_buffer: support 1,2,4,8

        Returns:
        -------------
        None
        """
        self.data_input_ub_2 = self.tik_instance.Tensor("float32", self.shape_y_ub,
                                                        name="data_input_ub_2", scope=tik.scope_ubuf)
        self.data_input_ub_2_fp16_list = []
        self.data_input_ub_2_trans_list = []
        for db_idx in range(double_buffer):
            tensor_name1 = "data_input_ub_2_fp16_%d" % (db_idx + 1)
            tensor_name2 = "data_input_ub_2_trans_%d" % (db_idx + 1)
            tensor_ins1 = self.tik_instance.Tensor("float16", shape_y_ub,
                                                   name=tensor_name1, scope=tik.scope_ubuf)
            tensor_ins2 = self.tik_instance.Tensor("float16", shape_y_ub_trans,
                                                   name=tensor_name2, scope=tik.scope_ubuf)
            self.data_input_ub_2_fp16_list.append(tensor_ins1)
            self.data_input_ub_2_trans_list.append(tensor_ins2)

    def _init_tensor_ub(self):
        """ init tensor_c of matmul, normal tensor of argmin and scalar
        """
        self.matmul_output_ub = self.tik_instance.Tensor(self.input_dtype, self.shape_z_ub_extend,
                                                         name="matmul_output_ub", scope=tik.scope_ubuf)
        self.matmul_output_ub_nd = self.tik_instance.Tensor(self.input_dtype, self.shape_z_ub_nd,
                                                            name="matmul_output_ub_nd", scope=tik.scope_ubuf)

        if self.vcmin_fp32_supported or self.high_perf:
            self.min_distance_ub = self.tik_instance.Tensor(self.input_dtype, (self.m_tiling, 2),
                                                            name="min_distance_ub_fp32", scope=tik.scope_ubuf)
            self.local_min_distance_ub = self.tik_instance.Tensor(self.input_dtype, (self.m_tiling, 1),
                                                                  name="local_min_distance_ub", scope=tik.scope_ubuf)
            self.local_min_index_ub = self.tik_instance.Tensor(self.input_dtype, (self.m_tiling,),
                                                               name="local_min_index_ub", scope=tik.scope_ubuf)
        else:
            self.ub_min_8 = self.tik_instance.Tensor(self.input_dtype, (8, 8),
                                                     name="ub_min_8", scope=tik.scope_ubuf)
            self.cmp_mask_ub = self.tik_instance.Tensor(MASK_DTYPE, (self.n_tiling,),
                                                        name="cmp_mask_ub", scope=tik.scope_ubuf)
            self.ub_index_int32 = self.tik_instance.Tensor(INDEX_DTYPE, (8, 8),
                                                           name="ub_index_int32", scope=tik.scope_ubuf)

        self.scalar_two = self.tik_instance.Scalar(dtype=self.input_dtype, init_value=2)
        self.scalar_vd = self.tik_instance.Scalar(dtype=self.input_dtype)
        self.scalar_index_offset = self.tik_instance.Scalar(dtype=INDEX_DTYPE)
        self.global_scalar_min = self.tik_instance.Scalar(dtype=self.input_dtype)

    def _init_tensor_ub_global(self):
        """ init tensor of global domain in ub buffer
        """
        if self.vcmin_fp32_supported or self.high_perf:
            global_index_dtype = self.input_dtype
        else:
            global_index_dtype = INDEX_DTYPE
        self.global_min_index_ub = self.tik_instance.Tensor(global_index_dtype, self.shape_global_min_distance_ub,
                                                            name="global_min_index_ub", scope=tik.scope_ubuf)
        self.output_count_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_min_num,),
                                                        name="output_count_ub", scope=tik.scope_ubuf)
        self.global_min_distance_ub = self.tik_instance.Tensor(self.input_dtype, self.shape_global_min_distance_ub,
                                                               name="global_min_distance_ub", scope=tik.scope_ubuf)
        self.output_total_distance_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_min_num,),
                                                                 name="output_total_distance_ub", scope=tik.scope_ubuf)

        self.scalar_max_fp32 = self.tik_instance.Scalar(dtype=self.input_dtype, init_value=SCALAR_MAX_FP32)
        self.scalar_zero = self.tik_instance.Scalar(dtype=self.input_dtype, init_value=0)
        self.scalar_one = self.tik_instance.Scalar(dtype=self.input_dtype, init_value=1)

    def _init_tensor_high_perf(self, n_tiling):
        """
        init tensor for high_performance mode

        Parameters
        ----------
        n_tiling : valid data length of axis n, number

        Returns
        -------
        None
        """
        high_perf_dtype = "float16"
        m_tiling_real = VNCHWCONV_MIN_SIZE
        self.vcmin_input_fp16 = \
            self.tik_instance.Tensor(high_perf_dtype, (m_tiling_real, n_tiling),
                                     name="vcmin_input_fp16", scope=tik.scope_ubuf)
        self.vcmin_output_fp16 = \
            self.tik_instance.Tensor(high_perf_dtype, (m_tiling_real, 2),
                                     name="vcmin_output_fp16", scope=tik.scope_ubuf)
        self.local_min_distance_ub_fp16 = \
            self.tik_instance.Tensor(high_perf_dtype, (m_tiling_real, 1),
                                     name="local_min_distance_ub_fp16", scope=tik.scope_ubuf)
        self.vcmin_output_trans1 = \
            self.tik_instance.Tensor(high_perf_dtype, (2, m_tiling_real),
                                     name="vcmin_output_trans1", scope=tik.scope_ubuf)
        self.vcmin_output_trans2 = \
            self.tik_instance.Tensor(high_perf_dtype, (2, m_tiling_real),
                                     name="vcmin_output_trans2", scope=tik.scope_ubuf)
        self.index_double = \
            self.tik_instance.Tensor(high_perf_dtype, (2 * m_tiling_real,),
                                     name="index_double", scope=tik.scope_ubuf)
        self.index_trans_to_int32 = \
            self.tik_instance.Tensor(high_perf_dtype, (2 * m_tiling_real,),
                                     name="index_trans_to_int32", scope=tik.scope_ubuf)
        self.tensor_index_offset_int32 = \
            self.tik_instance.Tensor(INDEX_DTYPE, (m_tiling_real,),
                                     name="tensor_index_offset_int32", scope=tik.scope_ubuf)
        self.scalar_zero_fp16 = self.tik_instance.Scalar(dtype=high_perf_dtype, init_value=0)

    def _compute_one_core(self, blk_idx):
        """
        compute in m tiling loop

        Parameters:
        -------------
        blk_idx: index of ai-core, expr

        Returns:
        -------------
        None
        """
        self._check_tiling_key(self.tiling, ["AUB_shape"])
        m_aub = self.tiling["AUB_shape"][1]
        with self.tik_instance.if_scope(tik.all(blk_idx == self.aic_cnt - 1, self.m_last_core > 0)):
            if self.m_last_tiling_loop > 0:
                with self.tik_instance.for_range(0, self.m_last_tiling_loop) as mlt_idx:
                    start_gm = self.m_each_core * blk_idx + mlt_idx * m_aub
                    self._matmul_gm_to_l0a(self.data_input_gm_1[start_gm:(start_gm + m_aub), :])
                    self._compute_one_core_n(start_gm)
            if self.m_last_tiling_left > 0:
                start_gm = self.m_each_core * blk_idx + self.m_last_tiling_loop * m_aub
                self._matmul_gm_to_l0a_tail(self.data_input_gm_1[start_gm:, :], is_last_core=True)
                self._compute_one_core_n(start_gm, is_last_core=True, is_m_tail=True)
        with self.tik_instance.else_scope():
            if self.m_tiling_loop > 0:
                with self.tik_instance.for_range(0, self.m_tiling_loop) as mt_idx:
                    start_gm = self.m_each_core * blk_idx + mt_idx * m_aub
                    self._matmul_gm_to_l0a_db(self.data_input_gm_1[start_gm:(start_gm + m_aub), :])
                    self._compute_one_core_n(start_gm)
            if self.m_tiling_left > 0:
                start_gm = self.m_each_core * blk_idx + self.m_tiling_loop * m_aub
                self._matmul_gm_to_l0a_tail(self.data_input_gm_1[start_gm:, :])
                self._compute_one_core_n(start_gm, is_m_tail=True)

    def _compute_one_core_n(self, m_gm_idx, is_last_core=False, is_m_tail=False):
        """
        compute matmul, argmin and unsorted_segment_sum in n tiling loop,
        then move result to gm

        Parameters:
        -------------
        m_gm_idx: global index of m axis in gm, expr
        is_last_core: whether is last core, bool
        is_m_tail: whether axis m has tail in each core, whose tail length
            maybe is different between the last core and the other cores, bool

        Returns:
        -------------
        None
        """
        self._init_tensor_ub_global()
        vdup_rpt = self.m_tiling * self.m_tiling_ub_loop // FP32_MASK
        vdup_mask_left = self.m_tiling * self.m_tiling_ub_loop % FP32_MASK
        if vdup_rpt > 0:
            self.tik_instance.vector_dup(FP32_MASK, self.global_min_distance_ub, self.scalar_max_fp32,
                                         vdup_rpt, 1, 8)
        if vdup_mask_left > 0:
            self.tik_instance.vector_dup(vdup_mask_left, self.global_min_distance_ub[vdup_rpt * FP32_MASK],
                                         self.scalar_max_fp32, 1, 1, 8)

        self._check_tiling_key(self.tiling, ["BUB_shape"])
        n_bub = self.tiling["BUB_shape"][1]
        param_dict = {"is_last_core": is_last_core, "is_m_tail": is_m_tail, "is_n_tail": False}
        if self.n_tiling_loop > 0:
            with self.tik_instance.for_range(0, self.n_tiling_loop) as nt_idx:
                start_gm = nt_idx * n_bub
                self._matmul_gm_to_l0b_db(self.data_input_gm_2[start_gm:(start_gm + n_bub), :])
                self._mmad(start_gm, param_dict)
        if self.n_tiling_left > 0:
            start_gm = self.n_tiling_loop * n_bub
            self._matmul_gm_to_l0b_tail(self.data_input_gm_2[start_gm:, :])
            param_dict["is_n_tail"] = True
            self._mmad(start_gm, param_dict)

        self._unsorted_segment_sum(m_gm_idx, is_last_core=is_last_core, is_m_tail=is_m_tail)

    def _matmul_gm_to_l0a(self, tensor_a_gm):
        """
        move tensor_a: gm -> ub -> l1 -> l0a

        Parameters:
        -------------
        tensor_a_gm: tensor_a in gm

        Returns:
        -------------
        None
        """
        # release ub buffer of tensor_a when tensor_a moves to l1
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            self._init_matmul_tensor_a_ub()
            mv_burst_len = self._elecnt_of_shape(self.shape_x_ub) * FP32_TYPE // UB_BLOCK_SIZE
            self.tik_instance.data_move(self.data_input_ub_1, tensor_a_gm,
                                        0, 1, mv_burst_len, 0, 0)

            self.vconv(self.data_input_ub_1_fp16, self.data_input_ub_1, self.shape_x_ub)

            self.nd_to_zn_3d(self.data_input_ub_1_trans, self.data_input_ub_1_fp16, self.shape_x_ub_trans)
            mv_burst_len = self._elecnt_of_shape(self.shape_x_ub_trans) * FP16_TYPE // UB_BLOCK_SIZE
            self.tik_instance.data_move(self.data_input_l1_1, self.data_input_ub_1_trans,
                                        0, 1, mv_burst_len, 0, 0)

        self.zn_to_zz(self.data_input_l0a, self.data_input_l1_1, self.shape_x_l0a)

    def _matmul_gm_to_l0a_tail(self, tensor_a_gm, is_last_core=False):
        """
        move tensor_a_tail: gm -> ub -> l1 -> l0a

        Parameters:
        -------------
        tensor_a_gm: tensor_a_tail in gm
        is_last_core: whether is last core, bool

        Returns:
        -------------
        None
        """
        m_aub = self.m_tiling_left
        if is_last_core:
            m_aub = self.m_last_tiling_left
        k_aub = tensor_a_gm.shape[1]
        shape_x_ub = (m_aub, k_aub)
        m1_aub = self._ceil(m_aub, 16)
        shape_x_ub_trans = (m1_aub, k_aub, 16)
        k1_aub = self._ceil(k_aub, 16)
        shape_x_l0a = (m1_aub, k1_aub, 16, 16)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            self._init_matmul_tensor_a_ub()
            self.tik_instance.data_move(self.data_input_ub_1[:m_aub, :], tensor_a_gm,
                                        0, 1, self._elecnt_of_shape(shape_x_ub) * FP32_TYPE // UB_BLOCK_SIZE, 0, 0)

            self.vconv(self.data_input_ub_1_fp16[:m_aub, :], self.data_input_ub_1[:m_aub, :], shape_x_ub)

            self.nd_to_zn_3d(self.data_input_ub_1_trans[:m1_aub, :, :], self.data_input_ub_1_fp16[:m_aub, :],
                             shape_x_ub_trans)

            self.tik_instance.data_move(self.data_input_l1_1[:m1_aub, :, :], self.data_input_ub_1_trans[:m1_aub, :, :],
                                        0, 1, self._elecnt_of_shape(shape_x_ub_trans) * FP16_TYPE // UB_BLOCK_SIZE,
                                        0, 0)

        self.zn_to_zz(self.data_input_l0a[:m1_aub, :, :, :], self.data_input_l1_1[:m1_aub, :, :], shape_x_l0a)

    def _matmul_gm_to_l0a_db(self, tensor_a_gm):
        """
        move tensor_a: (gm -> ub -> l1) * double_buffer -> l0a

        Parameters:
        -------------
        tensor_a_gm: tensor_a in gm

        Returns:
        -------------
        None
        """
        self._check_tiling_key(self.tiling, ["manual_pingpong_buffer"])
        self._check_tiling_key(self.tiling["manual_pingpong_buffer"], ["AUB_pbuffer"])
        double_buffer = self.tiling["manual_pingpong_buffer"]["AUB_pbuffer"]
        m_aub, k_aub = tensor_a_gm.shape
        m_aub //= double_buffer
        shape_x_ub = (m_aub, k_aub)
        m1_aub = self._ceil(m_aub, 16)
        shape_x_ub_trans = (m1_aub, k_aub, 16)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            self._init_matmul_tensor_a_ub_db(shape_x_ub, shape_x_ub_trans, double_buffer)
            for idx in range(double_buffer):
                self.tik_instance.data_move(
                    self.data_input_ub_1[idx * m_aub:(idx + 1) * m_aub, :],
                    tensor_a_gm[idx * m_aub:(idx + 1) * m_aub, :],
                    0, 1, self._elecnt_of_shape(shape_x_ub) * FP32_TYPE // UB_BLOCK_SIZE, 0, 0
                )

                self.vconv(self.data_input_ub_1_fp16_list[idx], self.data_input_ub_1[idx * m_aub:(idx + 1) * m_aub, :],
                           shape_x_ub)

                self.nd_to_zn_3d(self.data_input_ub_1_trans_list[idx], self.data_input_ub_1_fp16_list[idx],
                                 shape_x_ub_trans)

                self.tik_instance.data_move(
                    self.data_input_l1_1[idx * m1_aub:(idx + 1) * m1_aub, :, :], self.data_input_ub_1_trans_list[idx],
                    0, 1, self._elecnt_of_shape(shape_x_ub_trans) * FP16_TYPE // UB_BLOCK_SIZE, 0, 0
                )

        self.zn_to_zz(self.data_input_l0a, self.data_input_l1_1, self.shape_x_l0a)

    def _matmul_gm_to_l0b_tail(self, tensor_b_gm):
        """
        move tensor_b_tail: gm -> ub -> l1 -> l0b

        Parameters:
        -------------
        tensor_b_gm: tensor_b_tail in gm

        Returns:
        -------------
        None
        """
        n_bub, k_bub = tensor_b_gm.shape
        shape_y_ub = (n_bub, k_bub)
        n1_bub = self._ceil(n_bub, 16)
        shape_y_ub_trans = (n1_bub, k_bub, 16)
        k1_bub = self._ceil(k_bub, 16)
        shape_y_l0b = (k1_bub, n1_bub, 16, 16)
        # release ub buffer of tensor_b when tensor_b moves to l1
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            self._init_matmul_tensor_b_ub()
            self.tik_instance.data_move(self.data_input_ub_2[:n_bub, :], tensor_b_gm,
                                        0, 1, self._elecnt_of_shape(shape_y_ub) * FP32_TYPE // UB_BLOCK_SIZE, 0, 0)

            self.vconv(self.data_input_ub_2_fp16[:n_bub, :], self.data_input_ub_2[:n_bub, :], shape_y_ub)

            self.nd_to_zn_3d(self.data_input_ub_2_trans[:n1_bub, :, :], self.data_input_ub_2_fp16[:n_bub, :],
                             shape_y_ub_trans)

            self.tik_instance.data_move(self.data_input_l1_2[:n1_bub, :, :], self.data_input_ub_2_trans[:n1_bub, :, :],
                                        0, 1, self._elecnt_of_shape(shape_y_ub_trans) * FP16_TYPE // UB_BLOCK_SIZE,
                                        0, 0)

        self.nz_to_zn(self.data_input_l0b[:, :n1_bub, :, :], self.data_input_l1_2[:n1_bub, :, :], shape_y_l0b)

    def _matmul_gm_to_l0b_db(self, tensor_b_gm):
        """
        move tensor_b: (gm -> ub -> l1) * double_buffer -> l0b

        Parameters:
        -------------
        tensor_b_gm: tensor_b in gm

        Returns:
        -------------
        None
        """
        self._check_tiling_key(self.tiling, ["manual_pingpong_buffer"])
        self._check_tiling_key(self.tiling["manual_pingpong_buffer"], ["BUB_pbuffer"])
        double_buffer = self.tiling["manual_pingpong_buffer"]["BUB_pbuffer"]
        n_bub, k_bub = tensor_b_gm.shape
        n_bub //= double_buffer
        shape_y_ub = (n_bub, k_bub)
        n1_bub = self._ceil(n_bub, 16)
        shape_y_ub_trans = (n1_bub, k_bub, 16)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            self._init_matmul_tensor_b_ub_db(shape_y_ub, shape_y_ub_trans, double_buffer)
            for idx in range(double_buffer):
                self.tik_instance.data_move(
                    self.data_input_ub_2[idx * n_bub:(idx + 1) * n_bub, :],
                    tensor_b_gm[idx * n_bub:(idx + 1) * n_bub, :],
                    0, 1, self._elecnt_of_shape(shape_y_ub) * FP32_TYPE // UB_BLOCK_SIZE, 0, 0
                )

                self.vconv(self.data_input_ub_2_fp16_list[idx], self.data_input_ub_2[idx * n_bub:(idx + 1) * n_bub, :],
                           shape_y_ub)

                self.nd_to_zn_3d(self.data_input_ub_2_trans_list[idx], self.data_input_ub_2_fp16_list[idx],
                                 shape_y_ub_trans)

                self.tik_instance.data_move(
                    self.data_input_l1_2[idx * n1_bub:(idx + 1) * n1_bub, :, :], self.data_input_ub_2_trans_list[idx],
                    0, 1, self._elecnt_of_shape(shape_y_ub_trans) * FP16_TYPE // UB_BLOCK_SIZE, 0, 0
                )

        self.nz_to_zn(self.data_input_l0b, self.data_input_l1_2, self.shape_y_l0b)

    def _mmad(self, n_gm_idx: int, param_dict: dict) -> None:
        """
        mmad: A x B = C

        Parameters:
        -------------
        n_gm_idx: global index of n axis in gm, expr
        param_dict: include below
            is_last_core: whether is last core, bool
            is_m_tail: whether axis m has tail in each core
            is_n_tail: whether axis n has tail in each core, bool

        Returns:
        -------------
        None
        """
        self._init_tensor_ub()
        mmad_m = self.shape_x_l0a[0] * self.shape_x_l0a[2]
        mmad_k = self.shape_x_l0a[1] * self.shape_x_l0a[3]
        mmad_n = self.shape_y_l0b[1] * self.shape_y_l0b[2]
        self.tik_instance.mmad(self.data_output_l0c, self.data_input_l0a, self.data_input_l0b,
                               mmad_m, mmad_k, mmad_n, 0)

        self._check_tiling_key(self.tiling, ["CUB_matrix"])
        param_dict_argmin = {"is_last_core": param_dict.get("is_last_core", False),
                             "is_m_tail": param_dict.get("is_m_tail", False),
                             "is_n_tail": False}
        if param_dict["is_n_tail"]:
            if self.n_tiling_cub_loop > 0:
                with self.tik_instance.for_range(0, self.n_tiling_cub_loop) as ntc_idx:
                    start_gm = n_gm_idx + ntc_idx * self.tiling["CUB_matrix"][0] * 16
                    start_l0c = ntc_idx * self.tiling["CUB_matrix"][0]
                    self._matmul_l0c_to_ub(0, start_l0c, self.shape_z_ub)
                    self._argmin(start_gm, 0, param_dict_argmin)
            if self.n_tiling_cub_left > 0:
                start_gm = n_gm_idx + self.n_tiling_cub_loop * self.tiling["CUB_matrix"][0] * 16
                start_l0c = self.n_tiling_cub_loop * self.tiling["CUB_matrix"][0]
                shape_z_ub = (self._ceil(self.n_tiling_cub_left, 16), self.tiling["CUB_matrix"][1] * 16, 16)
                self._matmul_l0c_to_ub(0, start_l0c, shape_z_ub, is_n_tail=True)
                param_dict_argmin["is_n_tail"] = True
                self._argmin(start_gm, 0, param_dict_argmin)
        else:
            with self.tik_instance.for_range(0, self.m_tiling_ub_loop) as mtu_idx:
                m_l0c_idx = mtu_idx * self.tiling["CUB_matrix"][1] * 16
                with self.tik_instance.for_range(0, self.n_tiling_ub_loop) as ntu_idx:
                    start_gm = n_gm_idx + ntu_idx * self.tiling["CUB_matrix"][0] * 16
                    start_l0c = ntu_idx * self.tiling["CUB_matrix"][0]
                    self._matmul_l0c_to_ub(m_l0c_idx, start_l0c, self.shape_z_ub)
                    self._argmin(start_gm, m_l0c_idx, param_dict_argmin)

    def _matmul_l0c_to_ub(self, m_l0c_idx, n_l0c_idx, shape_z_ub, is_n_tail=False):
        """
        move tensor_c: l0c -> ub

        Parameters:
        -------------
        m_l0c_idx: local index of axis m in l0c, expr
        n_l0c_idx: local index of axis n1 in l0c, expr
        shape_z_ub: input shape of tensor_c, tuple
        is_n_tail: whether axis n has tail in cub, bool

        Returns:
        -------------
        None
        """
        nc_factor, mc_ub, n0 = shape_z_ub
        # bank conflict issue
        # read-write conflict in the same bank: set large tensor shape of matmul_output_ub and matmul_output_ub_nd,
        # for the sake of buffer neighbors of these two tensors.
        # read-read conflict in the same bank group: insert fake data between each [m, n0] of shape [n1, m, n0],
        # which is the meaning of parameter EXTEND_LENGTH
        burst_len = mc_ub * n0 * FP32_TYPE // L0C_BLOCK_SIZE
        self.tik_instance.data_move(self.matmul_output_ub, self.data_output_l0c[n_l0c_idx, m_l0c_idx, 0],
                                    0, nc_factor, burst_len, (self.m_tiling_ub_loop - 1) * burst_len,
                                    EXTEND_LENGTH)
        self._nz_to_nd(self.matmul_output_ub_nd[:, :nc_factor * n0],
                       self.matmul_output_ub[:nc_factor, :, :], shape_z_ub, is_n_tail=is_n_tail)

    def _argmin(self, n_gm_idx: int, m_l0c_idx: int, param_dict: dict) -> None:
        """
        compute argmin in ub buffer

        Parameters:
        -------------
        n_gm_idx: global index of axis n in gm, expr
        m_l0c_idx: local index of axis m in l0c, expr
        param_dict: include below
            is_last_core: whether is last core,bool
            is_m_tail: whether axis m has tail in each core, bool
            is_n_tail: whether axis n has tail in cub, bool

        Returns:
        -------------
        None
        """
        is_last_core = param_dict.get("is_last_core", False)
        is_m_tail = param_dict.get("is_m_tail", False)
        is_n_tail = param_dict.get("is_n_tail", False)
        m_tiling = self.m_tiling
        n_tiling = self.n_tiling
        if is_m_tail:
            if is_last_core:
                m_tiling = self.m_last_tiling_left
            else:
                m_tiling = self.m_tiling_left
        if is_n_tail:
            n_tiling = self.n_tiling_cub_left

        self.data_input_ub_4_broadcast = self.matmul_output_ub.reshape(self.shape_broadcast_ub_extend)
        self.data_input_ub_4_broadcast = self.data_input_ub_4_broadcast[:self.m_tiling, :]
        self.data_input_ub_4 = self.tik_instance.Tensor(self.input_dtype, (1, self.n_tiling),
                                                        name="data_input_ub_4", scope=tik.scope_ubuf)
        self.scalar_index_offset.set_as(n_gm_idx)
        # move sum_square_centroid from gm to ub
        mv_burst = n_tiling // self.ub_min_num
        with self.tik_instance.if_scope(mv_burst > 0):
            self.tik_instance.data_move(self.data_input_ub_4, self.data_input_gm_4[0, n_gm_idx],
                                        0, 1, mv_burst, 0, 0)

        self._broadcast(m_tiling, n_tiling)

        self._monocular_operator(self.matmul_output_ub_nd, self.matmul_output_ub_nd, self.scalar_two, "vmuls")

        self._binary_operator(self.data_input_ub_4_broadcast, self.data_input_ub_4_broadcast,
                              self.matmul_output_ub_nd, "vsub")

        if self.vcmin_fp32_supported or self.high_perf:
            self._calc_min_distance_perf_func(m_tiling, n_tiling, m_l0c_idx)
        else:
            self._calc_min_distance(m_tiling, n_tiling)

    def _calc_min_distance_perf_func(self, m_tiling, n_tiling, m_l0c_idx):
        """
        Calculate minimum distance using vcmin

        Parameters:
        -------------
        m_tiling: valid data length of axis m, number
        n_tiling: valid data length of axis n, number
        m_l0c_idx: local index of axis m in l0c, expr

        Returns:
        -------------
        None
        """
        # n_tiling belongs to [1, 64] for vcmin
        n_vcmin_loop = n_tiling // FP32_MASK
        n_vcmin_left = n_tiling % FP32_MASK
        if not self.vcmin_fp32_supported:
            self._init_tensor_high_perf(n_tiling)
        if n_vcmin_loop > 0:
            for nvl_idx in range(n_vcmin_loop):
                n_cub_idx = nvl_idx * FP32_MASK
                if self.vcmin_fp32_supported:
                    self._calc_min_distance_perf_fp32(m_tiling, FP32_MASK, m_l0c_idx, n_cub_idx)
                else:
                    self._calc_min_distance_perf(m_tiling, FP32_MASK, m_l0c_idx, n_cub_idx)
        if n_vcmin_left > 0:
            n_cub_idx = n_vcmin_loop * FP32_MASK
            if self.vcmin_fp32_supported:
                self._calc_min_distance_perf_fp32(m_tiling, n_vcmin_left, m_l0c_idx, n_cub_idx)
            else:
                self._calc_min_distance_perf(m_tiling, n_vcmin_left, m_l0c_idx, n_cub_idx)

    def _broadcast(self, m_tiling, n_tiling):
        """
        broadcast sum_square_sample from (1, n) to (m, n)

        Paramters
        --------------
        m_tiling: valid data length of axis m, number
        n_tiling: valid data length of axis n, number

        Returns:
        --------------
        None
        """
        vadds_repeat_loop = m_tiling // VECTOR_REPEAT_MAX
        vadds_repeat_tail = m_tiling % VECTOR_REPEAT_MAX
        mask_loop = n_tiling // FP32_MASK
        mask_tail = n_tiling % FP32_MASK

        if vadds_repeat_loop > 0:
            with self.tik_instance.for_range(0, vadds_repeat_loop) as vrl_idx:
                self._broadcast_sub_func(mask_loop, mask_tail, vrl_idx, VECTOR_REPEAT_MAX)
        if vadds_repeat_tail > 0:
            self._broadcast_sub_func(mask_loop, mask_tail, vadds_repeat_loop, vadds_repeat_tail)

    def _broadcast_sub_func(self, mask_loop, mask_tail, repeat_offset, repeat):
        """
        broadcast sum_square_sample from (1, n) to (m, n)

        Paramters
        --------------
        mask_loop: loop in the n_tiling
        mask_tail: tail in the n_tiling
        repeat_offset: offset in dst tensor
        repeat: repeat of vadds

        Returns:
        --------------
        None
        """
        vadds_dst_rpt_stride = self.n_tiling // self.ub_min_num
        if mask_loop > 0:
            with self.tik_instance.for_range(0, mask_loop) as ml_idx:
                dst = self.data_input_ub_4_broadcast[repeat_offset * VECTOR_REPEAT_MAX, ml_idx * FP32_MASK]
                src = self.data_input_ub_4[0, ml_idx * FP32_MASK]
                self.tik_instance.vadds(FP32_MASK, dst, src, self.scalar_zero,
                                        repeat, 1, 1, vadds_dst_rpt_stride, 0)
        if mask_tail > 0:
            dst_tail = self.data_input_ub_4_broadcast[repeat_offset * VECTOR_REPEAT_MAX, mask_loop * FP32_MASK]
            src_tail = self.data_input_ub_4[0, mask_loop * FP32_MASK]
            self.tik_instance.vadds(mask_tail, dst_tail, src_tail, self.scalar_zero,
                                    repeat, 1, 1, vadds_dst_rpt_stride, 0)

    def _calc_min_distance_perf_fp32(self, m_tiling, n_tiling, m_l0c_idx, n_cub_idx):
        """
        Calculate minimum distance from samples to centroids using optimized vcmin fp32

        Parameters:
        -------------
        m_tiling: valid data length of axis m, number
        n_tiling: valid data length of axis n, number
        m_l0c_idx: local index of axis m in l0c, expr
        n_cub_idx: local index of axis n in src_tensor, expr

        Returns:
        -------------
        None
        """
        # Use vcmin to get the minimum value of each row in m_tiling rows
        param_dict = {"m_tiling": m_tiling, "n_tiling": n_tiling, "n_cub_idx": n_cub_idx}
        self._vcmin(self.min_distance_ub, self.data_input_ub_4_broadcast, param_dict)
        # Obtain the minimum and minimum indexes from the results of vcmin, respectively
        vr_loop = (m_tiling * 2) // FP32_MASK
        vr_tail = (m_tiling * 2) % FP32_MASK
        if vr_loop > 0:
            self.tik_instance.vreduce(FP32_MASK, self.local_min_distance_ub, self.min_distance_ub,
                                      1, vr_loop, 1, 8, 8, 0, None, "normal")
            self.tik_instance.vreduce(FP32_MASK, self.local_min_index_ub, self.min_distance_ub,
                                      2, vr_loop, 1, 8, 8, 0, None, "normal")
        if vr_tail > 0:
            self.tik_instance.vreduce(vr_tail, self.local_min_distance_ub[vr_loop * FP32_MASK // 2:, :],
                                      self.min_distance_ub[vr_loop * FP32_MASK // 2:, :],
                                      1, 1, 1, 0, 0, 0, None, "counter")
            self.tik_instance.vreduce(vr_tail, self.local_min_index_ub[vr_loop * FP32_MASK // 2],
                                      self.min_distance_ub[vr_loop * FP32_MASK // 2:, :],
                                      2, 1, 1, 0, 0, 0, None, "counter")
        # Compare local and global minimums and update
        self._cmp_value_perf(m_tiling, m_l0c_idx)
        local_min_index_ub_int32 = self.local_min_index_ub.reinterpret_cast_to(INDEX_DTYPE)
        # Local minimum index adds tiling offset
        vadds_loop = m_tiling // FP32_MASK
        vadds_tail = m_tiling % FP32_MASK
        scalar_index_offset =  self.scalar_index_offset
        if n_cub_idx > 0:
            s_64 = self.tik_instance.Scalar(dtype="int32", init_value=FP32_MASK)
            scalar_index_offset.set_as(self.scalar_index_offset + s_64)
        if vadds_loop > 0:
            self.tik_instance.vadds(FP32_MASK, local_min_index_ub_int32, local_min_index_ub_int32,
                                    scalar_index_offset, vadds_loop, 1, 1, 8, 8)
        if vadds_tail > 0:
            self.tik_instance.vadds(vadds_tail, local_min_index_ub_int32[vadds_loop * FP32_MASK],
                                    local_min_index_ub_int32[vadds_loop * FP32_MASK],
                                    scalar_index_offset, 1, 1, 1, 8, 8)
        self.local_min_index_ub = local_min_index_ub_int32.reinterpret_cast_to("float32")
        # Update the global minimum index
        self._update_global_index(m_tiling, m_l0c_idx)

    def _calc_min_distance(self, m_tiling, n_tiling):
        """
        Calculate minimum distance from samples to centroids using vmin

        Parameters:
        -------------
        m_tiling: valid data length of axis m, number
        n_tiling: valid data length of axis n, number

        Returns:
        -------------
        None
        """
        row_batch = 8
        get_min_loop = m_tiling // row_batch
        get_min_left = m_tiling % row_batch
        with self.tik_instance.for_range(0, get_min_loop) as gm_idx:
            self._calc_min_distance_sub_func(n_tiling, gm_idx, row_batch)
        if get_min_left > 0:
            self._calc_min_distance_sub_func(n_tiling, get_min_loop, get_min_left)

    def _calc_min_distance_sub_func(self, n_tiling, gm_idx, row_batch_real):
        """
        sub-function of _calc_min_distance()

        Parameters:
        -------------
        n_tiling: valid data length of axis n, number
        gm_idx: index of get_min_loop, expr
        row_batch_real: real row_batch which is smaller than or equal to 8, number

        Returns:
        -------------
        None
        """
        # The number of rows processed at one time is 8
        row_batch = 8
        # The number of cols processed at one time is 8
        col_batch = 8
        mask = row_batch_real * col_batch
        self.tik_instance.vector_dup(FP32_MASK, self.ub_min_8, SCALAR_MAX_FP32, 1, 1, 8)
        vmin_rpt = n_tiling // col_batch
        vmin_blk_stride = self.n_tiling // col_batch
        # Get the minimum 8 values in each of 8 rows at a time
        self.tik_instance.vmin(mask, self.ub_min_8, self.data_input_ub_4_broadcast[gm_idx * row_batch, 0],
                               self.ub_min_8, vmin_rpt, 1, vmin_blk_stride, 1, 0, 1, 0)
        self._set_init_index(gm_idx, row_batch, vmin_rpt, vmin_blk_stride)
        # Get the minimum value and the minimum index of the eight values in the 8 rows
        self.min_value = self.tik_instance.Scalar(self.input_dtype)
        self.min_index = self.tik_instance.Scalar(INDEX_DTYPE)
        with self.tik_instance.for_range(0, row_batch_real) as r_idx:
            self.min_value.set_as(self.ub_min_8[r_idx, 0])
            self.min_index.set_as(self.ub_index_int32[r_idx, 0])
            # Get the minimum value and the minimum index of the eight values in 1 row
            with self.tik_instance.for_range(1, col_batch) as c_idx:
                self._cmp_value(r_idx, c_idx)
            # Compare local and global minimums and update
            global_value = self.tik_instance.Scalar(self.input_dtype)
            global_index = self.tik_instance.Scalar(INDEX_DTYPE)
            global_value.set_as(self.global_min_distance_ub[gm_idx * row_batch + r_idx])
            global_index.set_as(self.global_min_index_ub[gm_idx * row_batch + r_idx])
            with self.tik_instance.if_scope(self.min_value < global_value):
                self.global_min_distance_ub[gm_idx * row_batch + r_idx].set_as(self.min_value)
                self.global_min_index_ub[gm_idx * row_batch + r_idx].set_as(self.min_index + self.scalar_index_offset)

    def _cmp_value(self, r_idx, c_idx):
        """
        Calculate minimum distance and index from one row

        Parameters:
        -------------
        r_idx: row index in 8 * 8 minimum matrix
        c_idx: col index in 8 * 8 minimum matrix

        Returns:
        -------------
        None
        """
        min_cmp_value = self.tik_instance.Scalar(self.input_dtype)
        min_cmp_index = self.tik_instance.Scalar(INDEX_DTYPE)
        min_cmp_value.set_as(self.ub_min_8[r_idx, c_idx])
        min_cmp_index.set_as(self.ub_index_int32[r_idx, c_idx])
        with self.tik_instance.if_scope(min_cmp_value < self.min_value):
            self.min_value.set_as(self.ub_min_8[r_idx, c_idx])
            self.min_index.set_as(min_cmp_index + c_idx)
        with self.tik_instance.if_scope(tik.all(min_cmp_value == self.min_value,
                                                min_cmp_index + c_idx < self.min_index)):
            self.min_value.set_as(self.ub_min_8[r_idx, c_idx])
            self.min_index.set_as(min_cmp_index + c_idx)

    def _calc_min_distance_perf(self, m_tiling, n_tiling, m_l0c_idx, n_cub_idx):
        """
        Calculate minimum distance from samples to centroids using optimized vcmin

        Parameters:
        -------------
        m_tiling: valid data length of axis m, number
        n_tiling: valid data length of axis n, number
        m_l0c_idx: local index of axis m in l0c, expr
        n_cub_idx: local index of axis n in src_tensor, expr

        Returns:
        -------------
        None
        """
        self.vconv(self.vcmin_input_fp16, self.data_input_ub_4_broadcast, self.shape_broadcast_ub,
                   src_dtype="float32")
        self._vcmin(self.vcmin_output_fp16, self.vcmin_input_fp16,
                    {"m_tiling": m_tiling, "n_tiling": n_tiling, "n_cub_idx": n_cub_idx})
        # Obtain the minimum and minimum index from results of vcmin, respectively
        trans_1_dst_list = [self.vcmin_output_trans1[16 * i] for i in range(16)]
        trans_1_src_list = [self.vcmin_output_fp16[32 * i] for i in range(16)]
        self.tik_instance.vnchwconv(False, False, trans_1_dst_list, trans_1_src_list, 2, 16, 1)

        trans_2_dst_list = [self.vcmin_output_trans2[16 * i] for i in range(16)]
        trans_2_src_list = [self.vcmin_output_trans1[32 * i] for i in range(16)]
        self.tik_instance.vnchwconv(False, False, trans_2_dst_list, trans_2_src_list, 2, 16, 1)

        self.tik_instance.data_move(self.local_min_distance_ub_fp16, self.vcmin_output_trans2,
                                    0, 1, self.m_tiling // 16, 0, 0)
        self.vconv(self.local_min_distance_ub, self.local_min_distance_ub_fp16, (self.m_tiling, 1),
                   src_dtype="float16")
        self.tik_instance.vector_dup(FP16_MASK, self.index_double, 0, 4, 1, 8)
        self.tik_instance.vadds(FP16_MASK, self.index_double, self.vcmin_output_trans1[0, 16],
                                self.scalar_zero_fp16, 2, 2, 2, 16, 16)

        index_trans_2_dst_list = [self.index_trans_to_int32[32 * i] for i in range(16)]
        index_trans_2_src_list = [self.index_double[16 * i] for i in range(16)]
        self.tik_instance.vnchwconv(False, False, index_trans_2_dst_list, index_trans_2_src_list, 2, 1, 16)

        # compare local and global minimums and update
        self._cmp_value_perf(m_tiling, m_l0c_idx)
        # Local minimum index adds tiling offset
        local_min_index_ub_int32 = self.index_trans_to_int32.reinterpret_cast_to("int32")
        vadd_loop = m_tiling // FP32_MASK
        vadd_tail = m_tiling % FP32_MASK
        scalar_index_offset = self.scalar_index_offset
        if n_cub_idx > 0:
            s_64 = self.tik_instance.Scalar(dtype="int32", init_value=FP32_MASK)
            scalar_index_offset.set_as(self.scalar_index_offset + s_64)
        if vadd_loop > 0:
            self.tik_instance.vector_dup(FP32_MASK, self.tensor_index_offset_int32, scalar_index_offset,
                                         vadd_loop, 1, 8)
            self.tik_instance.vadd(FP32_MASK, local_min_index_ub_int32, local_min_index_ub_int32,
                                   self.tensor_index_offset_int32, vadd_loop, 1, 1, 1, 8, 8, 8)
        if vadd_tail > 0:
            self.tik_instance.vector_dup(vadd_tail,
                                         self.tensor_index_offset_int32[vadd_loop * FP32_MASK],
                                         scalar_index_offset,
                                         1, 1, 8)
            self.tik_instance.vadd(vadd_tail,
                                   local_min_index_ub_int32[vadd_loop * FP32_MASK],
                                   local_min_index_ub_int32[vadd_loop * FP32_MASK],
                                   self.tensor_index_offset_int32[vadd_loop * FP32_MASK],
                                   1, 1, 1, 1, 8, 8, 8)
        self.local_min_index_ub = local_min_index_ub_int32.reinterpret_cast_to("float32")
        # Update the global minimum index
        self._update_global_index(m_tiling, 0)

    def _cmp_value_perf(self, m_tiling, m_l0c_idx):
        """
        Calculate minimum distance in perf mode

        Parameters:
        -------------
        m_tiling: row index in 8 * 8 minimum matrix
        m_l0c_idx: col index in 8 * 8 minimum matrix

        Returns:
        -------------
        None
        """
        vmin_loop = m_tiling // FP32_MASK
        vmin_tail = m_tiling % FP32_MASK
        self.global_min_dist_ub_tmp = self.tik_instance.Tensor(self.input_dtype, self.shape_global_min_distance_ub,
                                                               name="global_min_dist_ub_tmp", scope=tik.scope_ubuf)
        if vmin_loop > 0:
            self.tik_instance.vmin(FP32_MASK, self.global_min_dist_ub_tmp[m_l0c_idx], self.local_min_distance_ub,
                                   self.global_min_distance_ub[m_l0c_idx], vmin_loop, 1, 1, 1, 8, 8, 8)
        if vmin_tail > 0:
            self.tik_instance.vmin(vmin_tail, self.global_min_dist_ub_tmp[m_l0c_idx + vmin_loop * FP32_MASK],
                                   self.local_min_distance_ub[vmin_loop * FP32_MASK],
                                   self.global_min_distance_ub[m_l0c_idx + vmin_loop * FP32_MASK],
                                   1, 1, 1, 1, 8, 8, 8)

    def _vcmin(self, dst_tensor, src_tensor, param_dict: dict) -> None:
        """
        Use vcmin to get the minimum value of each row in m_tiling rows

        Parameters:
        -------------
        dst_tensor: min_distance_ub
        src_tensor: data_input_ub_4_broadcast
        param_dict: include below
            m_tiling: valid data length of axis m, number
            n_tiling: valid data length of axis n, number
            n_cub_idx: local index of axis n in src_tensor, expr

        Returns:
        -------------
        None
        """
        m_tiling = param_dict.get("m_tiling")
        n_tiling = param_dict.get("n_tiling")
        n_cub_idx = param_dict.get("n_cub_idx")
        ub_min_num = 8 if src_tensor.dtype == "float32" else 16
        vcmin_loop = m_tiling // VECTOR_REPEAT_MAX
        vcmin_tail = m_tiling % VECTOR_REPEAT_MAX
        vcmin_src_rpt_stride = self.n_tiling // ub_min_num
        if vcmin_loop > 0:
            with self.tik_instance.for_range(0, vcmin_loop) as vl_idx:
                self.tik_instance.vcmin(n_tiling, dst_tensor[vl_idx * VECTOR_REPEAT_MAX, 0],
                                        src_tensor[vl_idx * VECTOR_REPEAT_MAX, n_cub_idx],
                                        VECTOR_REPEAT_MAX, 1, 1, vcmin_src_rpt_stride)
        if vcmin_tail > 0:
            self.tik_instance.vcmin(n_tiling, dst_tensor[vcmin_loop * VECTOR_REPEAT_MAX, 0],
                                    src_tensor[vcmin_loop * VECTOR_REPEAT_MAX, n_cub_idx],
                                    vcmin_tail, 1, 1, vcmin_src_rpt_stride)

    def _update_global_index(self, m_tiling, m_l0c_idx):
        """
        update global index of minimum distance

        Parameters:
        -------------
        m_tiling: valid data length of axis m, number
        m_l0c_idx: local index of axis m in l0c, expr

        Returns:
        -------------
        None
        """
        update_loop = m_tiling // FP32_MASK
        update_tail = m_tiling % FP32_MASK

        with self.tik_instance.for_range(0, update_loop) as u_idx:
            dst_offset = m_l0c_idx + u_idx * FP32_MASK
            cmp_mask = self.tik_instance.vcmp_eq(FP32_MASK, self.global_min_dist_ub_tmp[dst_offset],
                                                 self.global_min_distance_ub[dst_offset], 1, 1)
            self.tik_instance.vsel(FP32_MASK, 0, self.global_min_index_ub[dst_offset], cmp_mask,
                                   self.global_min_index_ub[dst_offset],
                                   self.local_min_index_ub[u_idx * FP32_MASK],
                                   1, 1, 1, 1, 8, 8, 8)
        if update_tail > 0:
            dst_offset = m_l0c_idx + update_loop * FP32_MASK
            cmp_mask_left = self.tik_instance.vcmp_eq(update_tail, self.global_min_dist_ub_tmp[dst_offset],
                                                 self.global_min_distance_ub[dst_offset], 1, 1)
            self.tik_instance.vsel(update_tail, 0, self.global_min_index_ub[dst_offset], cmp_mask_left,
                                   self.global_min_index_ub[dst_offset],
                                   self.local_min_index_ub[update_loop * FP32_MASK],
                                   1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.data_move(self.global_min_distance_ub, self.global_min_dist_ub_tmp,
                                    0, 1, self.m_tiling // self.ub_min_num, 0, 0)

    def _set_init_index(self, m_idx, row_batch, rpt, blk_stride):
        """
        Assign an initial value to the index with the smallest 8 values in each row

        Parameters:
        -------------
        m_idx: row index
        row_batch: Number of rows processed at a time
        rpt: repeat of vcmpv_eq
        blk_stride: block stride of vcmpv_eq

        Returns:
        -------------
        None
        """
        self.tik_instance.vcmpv_eq(self.cmp_mask_ub, self.ub_min_8,
                                   self.data_input_ub_4_broadcast[m_idx * row_batch, 0],
                                   rpt, 1, blk_stride, 0, 1)
        self.tik_instance.vector_dup(FP32_MASK, self.ub_index_int32, 0, 1, 1, 8)
        with self.tik_instance.for_range(0, rpt) as update_idx:
            index = rpt - 1 - update_idx
            mask_l = self.tik_instance.Scalar(MASK_DTYPE)
            mask_h = self.tik_instance.Scalar(MASK_DTYPE)
            mask_l.set_as(self.cmp_mask_ub[index])
            mask_h.set_as(0)
            with self.tik_instance.if_scope(mask_l != 0):
                self.tik_instance.vector_dup([mask_h, mask_l], self.ub_index_int32,
                                              index * 8, 1, 1, 8)

    def _unsorted_segment_sum(self, m_gm_idx, is_last_core=False, is_m_tail=False):
        """
        unsorted segment sum,
            sum and count distance result of each cluster

        Parameters:
        -------------
        m_gm_idx: global index of m axis in gm, expr
        is_last_core: whether is last core, bool
        is_m_tail: whether axis m has tail in each core, bool

        Returns:
        -------------
        None
        """
        shape_x1 = self.input_x1.get("shape")
        d_gm = shape_x1[1]
        cur_m = self.m_tiling * self.m_tiling_ub_loop
        if is_m_tail:
            if is_last_core:
                cur_m = self.m_last_tiling_left
            else:
                cur_m = self.m_tiling_left

        self.tik_instance.set_atomic_add(1)
        self.tik_instance.vector_dup(8, self.output_count_ub, self.scalar_zero, 1, 1, 8)
        self.tik_instance.vector_dup(8, self.output_total_distance_ub, self.scalar_zero, 1, 1, 8)
        self.output_count_ub[0].set_as(self.scalar_one)

        if self.use_actual_distance:
            self._calc_actual_distance(cur_m, m_gm_idx)

        self._output_loss(cur_m)

        min_index_to_gm = self.tik_instance.Scalar(dtype=INDEX_DTYPE)
        if self.vcmin_fp32_supported:
            global_min_index_ub_int32 = self.global_min_index_ub.reinterpret_cast_to("int32")
            once_m = cur_m
            once_sample_dma_burst = (d_gm * once_m) // self.ub_min_num
            once_sample = self.tik_instance.Tensor(self.input_dtype, (once_m, d_gm),
                                                   name="once_sample", scope=tik.scope_ubuf)
            self.tik_instance.data_move(once_sample, self.data_input_gm_1[m_gm_idx, 0],
                                        0, 1, once_sample_dma_burst, 0, 0)
        else:
            global_min_index_ub_int32 = self.global_min_index_ub
            once_sample = self.tik_instance.Tensor(self.input_dtype, (d_gm,),
                                                   name="once_sample", scope=tik.scope_ubuf)
        once_sample_out_dma_burst = d_gm // self.ub_min_num
        with self.tik_instance.for_range(0, cur_m) as m_idx:
            min_index_to_gm.set_as(global_min_index_ub_int32[m_idx])
            if self.vcmin_fp32_supported:
                self.tik_instance.data_move(self.data_output_gm_1[min_index_to_gm, 0], once_sample[m_idx, 0],
                                            0, 1, once_sample_out_dma_burst, 0, 0)
            else:
                self.tik_instance.data_move(once_sample, self.data_input_gm_1[m_gm_idx + m_idx, 0],
                                            0, 1, once_sample_out_dma_burst, 0, 0)
                self.tik_instance.data_move(self.data_output_gm_1[min_index_to_gm, 0], once_sample,
                                            0, 1, once_sample_out_dma_burst, 0, 0)
            self.tik_instance.data_move(self.data_output_gm_2[min_index_to_gm, 0], self.output_count_ub,
                                        0, 1, 1, 0, 0)
        self.tik_instance.set_atomic_add(0)

    def _calc_actual_distance(self, cur_m, m_gm_idx):
        """
        calculate the true minimum distance

        Parameters:
        -------------
        cur_m: actual size of M currently processed
        m_gm_idx: global index of m axis in gm

        Returns:
        -------------
        None
        """
        cur_m_align = self._ceil(cur_m, self.ub_min_num) * self.ub_min_num
        input_3_dma_burst = cur_m_align // self.ub_min_num
        shape_input_3_ub = (cur_m_align, 1)
        data_input_ub_3 = self.tik_instance.Tensor(self.input_dtype, shape_input_3_ub,
                                                   name="data_input_ub_3", scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_input_ub_3, self.data_input_gm_3[m_gm_idx, 0],
                                    0, 1, input_3_dma_burst, 0, 0)
        vadd_rpt = cur_m // FP32_MASK
        vadd_left = cur_m % FP32_MASK
        if vadd_rpt > 0:
            self.tik_instance.vadd(FP32_MASK, self.global_min_distance_ub, self.global_min_distance_ub,
                                   data_input_ub_3, vadd_rpt, 1, 1, 1, 8, 8, 8)
        if vadd_left > 0:
            self.tik_instance.vadd(vadd_left, self.global_min_distance_ub[vadd_rpt * FP32_MASK],
                                   self.global_min_distance_ub[vadd_rpt * FP32_MASK],
                                   data_input_ub_3[vadd_rpt * FP32_MASK], 1, 1, 1, 1, 8, 8, 8)

    def _output_loss(self, cur_m):
        """
        output the sum of the minimum distance

        Parameters:
        -------------
        cur_m: actual size of M currently processed

        Returns:
        -------------
        None
        """
        vcadd_loop = cur_m // FP32_MASK
        vcadd_left = cur_m % FP32_MASK
        if vcadd_loop > 0:
            with self.tik_instance.for_range(0, vcadd_loop) as vca_idx:
                self.tik_instance.vcadd(FP32_MASK, self.output_total_distance_ub,
                                        self.global_min_distance_ub[vca_idx * FP32_MASK],
                                        1, 1, 1, 8)
                self.tik_instance.data_move(self.data_output_gm_3,
                                            self.output_total_distance_ub,
                                            0, 1, 1, 0, 0)
        if vcadd_left > 0:
            self.tik_instance.vcadd(vcadd_left, self.output_total_distance_ub,
                                    self.global_min_distance_ub[vcadd_loop * FP32_MASK],
                                    1, 1, 1, 8)
            self.tik_instance.data_move(self.data_output_gm_3,
                                        self.output_total_distance_ub,
                                        0, 1, 1, 0, 0)

    def _nz_to_nd(self, dst, src, shape_z_ub, is_n_tail=False):
        """
        reshape tensor_c from format Nz to ND in ub buffer using vadds

        Parameters:
        -------------
        dst: tensor (m, n), fp32
        src: tensor (n1, m, n0), fp32
        shape_z_ub: input shape, tuple
        is_n_tail: whether axis n has tail in cub, bool

        Returns:
        -------------
        None
        """
        n1 = shape_z_ub[0]
        # mask belongs to [1,64] when data type is fp32
        repeat_loop = n1 * 16 // FP32_MASK
        repeat_left = n1 * 16 % FP32_MASK
        nz_to_nd_para = {
            "shape_z_ub": shape_z_ub,
            "is_n_tail": is_n_tail
        }
        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                nz_to_nd_para["mask"] = 32
                nz_to_nd_para["rpt_idx"] = rpt_idx
                self._nz_to_nd_sub_func(dst, src, nz_to_nd_para)
        if repeat_left > 0:
            nz_to_nd_para["mask"] = repeat_left // 2
            nz_to_nd_para["rpt_idx"] = repeat_loop
            self._nz_to_nd_sub_func(dst, src, nz_to_nd_para)

    def _nz_to_nd_sub_func(self, dst, src, nz_to_nd_para):
        """
        reshape tensor_c from format Nz to ND in ub buffer using vadds

        Parameters:
        -------------
        dst: tensor (m, n), fp32
        src: tensor (n1, m, n0), fp32
        nz_to_nd_para: dict of parameter

        Returns:
        -------------
        None
        """
        shape_z_ub = nz_to_nd_para.get("shape_z_ub")
        # repeat_times belongs to [1,255]
        m1_255 = shape_z_ub[1] // VECTOR_REPEAT_MAX
        m1_255_left = shape_z_ub[1] % VECTOR_REPEAT_MAX
        if m1_255 > 0:
            with self.tik_instance.for_range(0, m1_255) as m1_255_idx:
                nz_to_nd_para['m1_255_idx'] = m1_255_idx
                nz_to_nd_para['repeat_times'] = VECTOR_REPEAT_MAX
                self._vadds(dst, src, nz_to_nd_para)
        if m1_255_left > 0:
            nz_to_nd_para['m1_255_idx'] = m1_255
            nz_to_nd_para['repeat_times'] = m1_255_left
            self._vadds(dst, src, nz_to_nd_para)

    def _vadds(self, dst, src, nz_to_nd_para):
        """
        vadds entity

        Parameters:
        -------------
        dst: tensor (m, n), fp32
        src: tensor (n1, m, n0), fp32
        nz_to_nd_para: dict of parameter

        Returns:
        -------------
        None
        """
        for key in ("m1_255_idx", "rpt_idx", "mask", "repeat_times", "shape_z_ub", "is_n_tail"):
            if key not in nz_to_nd_para:
                error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                          "Lost key %s of nz_to_nd_para." % key)
        # default params
        scalar = 0
        dst_blk_stride = 2
        src_blk_stride = nz_to_nd_para["shape_z_ub"][1] * 2 + EXTEND_LENGTH
        dst_rep_stride = nz_to_nd_para["shape_z_ub"][0] * 2
        if nz_to_nd_para["is_n_tail"]:
            dst_rep_stride = self.shape_z_ub[0] * 2
        src_rep_stride = 2

        dst_part1 = dst[nz_to_nd_para["m1_255_idx"] * VECTOR_REPEAT_MAX, nz_to_nd_para["rpt_idx"] * 64]
        src_part1 = src[nz_to_nd_para["rpt_idx"] * 4, nz_to_nd_para["m1_255_idx"] * VECTOR_REPEAT_MAX, 0]
        dst_part2 = dst[nz_to_nd_para["m1_255_idx"] * VECTOR_REPEAT_MAX, nz_to_nd_para["rpt_idx"] * 64 + 8]
        src_part2 = src[nz_to_nd_para["rpt_idx"] * 4, nz_to_nd_para["m1_255_idx"] * VECTOR_REPEAT_MAX, 8]
        self.tik_instance.vadds(nz_to_nd_para["mask"], dst_part1, src_part1, scalar, nz_to_nd_para["repeat_times"],
                                dst_blk_stride, src_blk_stride, dst_rep_stride, src_rep_stride)
        self.tik_instance.vadds(nz_to_nd_para["mask"], dst_part2, src_part2, scalar, nz_to_nd_para["repeat_times"],
                                dst_blk_stride, src_blk_stride, dst_rep_stride, src_rep_stride)

    def _binary_operator(self, dst, src0, src1, operator):
        """
        binary operator vsub or vadd in ub buffer

        Parameters:
        -------------
        dst: tensor (m, n), fp32
        src0: tensor (m, n), fp32
        src1: tensor (m, n), fp32
        operator: binary operator type

        Returns:
        -------------
        None
        """
        n = self.shape_z_ub_nd[1]

        binary_operator_dict = {
            "vsub": self.tik_instance.vsub,
            "vadd": self.tik_instance.vadd
        }
        if operator not in binary_operator_dict:
            error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                      "Illegal binary operator.")

        unit = 64  # for fp32
        func = binary_operator_dict.get(operator)
        repeat = (self.shape_z_ub_nd[0] * n) // unit
        left = (self.shape_z_ub_nd[0] * n) % unit
        repeat_loop = repeat // VECTOR_REPEAT_MAX
        repeat_left = repeat % VECTOR_REPEAT_MAX

        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * VECTOR_REPEAT_MAX * unit
                func(unit, dst[offset // n, offset % n],
                     src0[offset // n, offset % n],
                     src1[offset // n, offset % n], VECTOR_REPEAT_MAX, 1, 1, 1, 8, 8, 8)

        if repeat_left > 0:
            offset = repeat_loop * VECTOR_REPEAT_MAX * unit
            func(unit, dst[offset // n, offset % n],
                 src0[offset // n, offset % n],
                 src1[offset // n, offset % n], repeat_left, 1, 1, 1, 8, 8, 8)

        if left > 0:
            offset = repeat * unit
            func(left, dst[offset // n, offset % n],
                 src0[offset // n, offset % n],
                 src1[offset // n, offset % n], 1, 1, 1, 1, 8, 8, 8)

    def _monocular_operator(self, dst, src, scalar, operator):
        """
        monocular operator vmuls in ub buffer

        Parameters:
        -------------
        dst: tensor (m, n), fp32
        src: tensor (m, n), fp32
        scalar: scalar, fp32
        operator: monocular operator type

        Returns:
        -------------
        None
        """
        n = self.shape_z_ub_nd[1]

        monocular_operator_dict = {
            "vmuls": self.tik_instance.vmuls
        }
        if operator not in monocular_operator_dict:
            error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                      "Illegal monocular operator.")

        unit = 64  # for fp32
        func = monocular_operator_dict.get(operator)
        repeat = (self.shape_z_ub_nd[0] * n) // unit
        left = (self.shape_z_ub_nd[0] * n) % unit
        repeat_loop = repeat // VECTOR_REPEAT_MAX
        repeat_left = repeat % VECTOR_REPEAT_MAX

        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * VECTOR_REPEAT_MAX * unit
                func(unit, dst[offset // n, offset % n],
                     src[offset // n, offset % n], scalar, VECTOR_REPEAT_MAX, 1, 1, 8, 8)

        if repeat_left > 0:
            offset = repeat_loop * VECTOR_REPEAT_MAX * unit
            func(unit, dst[offset // n, offset % n],
                 src[offset // n, offset % n], scalar, repeat_left, 1, 1, 8, 8)

        if left > 0:
            offset = repeat * unit
            func(left, dst[offset // n, offset % n],
                 src[offset // n, offset % n], scalar, 1, 1, 1, 8, 8)


def _shape_range_check(shape_x1, shape_x2):
    """
    shape range check

    Parameters:
    -------------
    shape_x1: list or tuple
    shape_x2: list or tuple

    Returns:
    -------------
    None
    """
    if shape_x1[0] <= 0 or shape_x1[1] <= 0 or shape_x2[0] <= 0:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Shape value must be larger than zero.")
    if shape_x1[0] % 16 != 0 or shape_x1[1] % 16 != 0 or shape_x2[0] % 16 != 0:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Axis m or k or n must be aligned to 16.")
    if shape_x1[1] > MAX_K_SIZE:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Axis k must be smaller than or equal to 1792.")
    if shape_x1[0] > INT64_SIZE or shape_x1[1] > INT64_SIZE or shape_x2[0] > INT64_SIZE:
        error_manager_cube.raise_err_message_cube("k_means_centroids", "Shape is too large.")


def _shape_check(para_dict):
    """
    shape check

    Parameters:
    -------------
    para_dict: inputs and outputs parameters, dict

    Returns:
    -------------
    None
    """
    x = para_dict.get("x")
    y = para_dict.get("y")
    sum_square_x = para_dict.get("sum_square_x")
    sum_square_y = para_dict.get("sum_square_y")

    shape_x1 = x.get("shape", ())
    shape_x2 = y.get("shape", ())
    shape_x4 = sum_square_y.get("shape", ())
    len_shape_x1 = len(shape_x1)
    len_shape_x2 = len(shape_x2)
    len_shape_x4 = len(shape_x4)

    if sum_square_x:
        # sum_square_x maybe is None
        shape_x3 = sum_square_x.get("shape", ())
        len_shape_x3 = len(shape_x3)
        if len_shape_x3 != INPUT_LENGTH:
            reason = ("Shape length of sum_square_x must be equal to 2, " +
                      "but recently is %d." % len_shape_x3)
            error_manager_cube.raise_err_message_cube("k_means_centroids", reason)
        if shape_x1[0] != shape_x3[0]:
            error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                      "Axis m of samples and sum_square_samples must be equal.")
        if shape_x3[1] != 1:
            error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                      "1 axis of sum_square_samples must be 1.")

    if len_shape_x1 != INPUT_LENGTH:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Shape length of x must be equal to 2, " +
                                                  "but recently is %d." % len_shape_x1)
    if len_shape_x2 != INPUT_LENGTH:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Shape length of y must be equal to 2, " +
                                                  "but recently is %d." % len_shape_x2)
    if len_shape_x4 != INPUT_LENGTH:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Shape length of sum_square_y must be equal to 2, " +
                                                  "but recently is %d." % len_shape_x4)
    if shape_x1[1] != shape_x2[1]:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Axis k of samples and centroids must be equal.")
    if shape_x2[0] != shape_x4[1]:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Axis n of centroids and sum_square_centroids must be equal.")
    if shape_x4[0] != 1:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "0 axis of sum_square_centroids must be 1.")
    _shape_range_check(shape_x1, shape_x2)


def _data_type_check(para_dict):
    """
    data type and format check

    Parameters:
    -------------
    para_dict: inputs and outputs parameters, dict

    Returns:
    -------------
    None
    """
    for key in ("x", "y", "sum_square_x", "sum_square_y",
                "segment_sum", "segment_count", "kmean_total_sum"):
        if key not in para_dict:
            error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                      "Lost key %s of para_dict." % key)
        if isinstance(para_dict[key], dict) and ("dtype" not in para_dict[key]):
            error_manager_cube.raise_err_message_cube("k_means_centroids", "Key error, lost dtype.")

    sum_square_x = para_dict.get("sum_square_x")
    support_dtype = ("float32",)
    input1_dtype = para_dict["x"]["dtype"]
    input2_dtype = para_dict["y"]["dtype"]
    input4_dtype = para_dict["sum_square_y"]["dtype"]
    output1_dtype = para_dict["segment_sum"]["dtype"]
    output2_dtype = para_dict["segment_count"]["dtype"]
    output3_dtype = para_dict["kmean_total_sum"]["dtype"]

    if sum_square_x:
        input3_dtype = sum_square_x.get("dtype")
        if input3_dtype not in support_dtype:
            reason = (("Input3 dtype only supports %s, " % (support_dtype,)) +
                      ("but recently is %s." % input3_dtype))
            error_manager_cube.raise_err_message_cube("k_means_centroids", reason)

    if input1_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Input1 dtype only supports %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % input1_dtype))
    if input2_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Input2 dtype only supports %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % input2_dtype))
    if input4_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Input4 dtype only supports %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % input4_dtype))
    if output1_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Output1 dtype only supports %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % output1_dtype))
    if output2_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Output2 dtype only supports %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % output2_dtype))
    if output3_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Output3 dtype only supports %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % output3_dtype))


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_BOOL,
    para_check.KERNEL_NAME,
)
def k_means_centroids(
    x,
    y,
    sum_square_y,
    sum_square_x,
    segment_sum,
    segment_count,
    kmean_total_sum,
    use_actual_distance=False,
    kernel_name="k_means_centroids",
    impl_mode="high_performance"
):
    """
    algorithm k_means_centroids

    Parameters:
    -------------
    x: dict
        samples, shape (m, d), fp32

    y: dict
        centroids, shape (n, d), fp32

    sum_square_x: dict
        sum of squares of samples, shape (m, 1), fp32

    sum_square_y: dict
        sum of squares of centroids, shape (1, n), fp32

    segment_sum: dict
        sum of distance result in each cluster, shape (n, d), fp32

    segment_count: dict
        count of distance result in each cluster, shape (n,), fp32

    kmean_total_sum: dict
        sum of all samples' distance to centroids, shape (1,), fp32

    use_actual_distance: bool
        whether to use actual distance

    kernel_name: str

    impl_mode : str
        assign high_performance or high_precision

    Returns:
    -------------
    tik_instance: tik instance
    """
    para_dict = {
        "x": x,
        "y": y,
        "sum_square_x": sum_square_x,
        "sum_square_y": sum_square_y,
        "segment_sum": segment_sum,
        "segment_count": segment_count,
        "kmean_total_sum": kmean_total_sum,
        "use_actual_distance": use_actual_distance,
        "kernel_name": kernel_name,
        "impl_mode": impl_mode
    }

    _shape_check(para_dict)
    _data_type_check(para_dict)

    kmeans = KMeansCentroids(para_dict)

    tik_instance = kmeans.k_means_centroids_compute()

    return tik_instance
