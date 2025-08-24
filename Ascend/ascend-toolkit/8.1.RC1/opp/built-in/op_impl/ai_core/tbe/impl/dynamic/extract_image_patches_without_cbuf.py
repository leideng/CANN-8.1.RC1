# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
#
# licensed under the Apache License, Version 2.0 (the "License");
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
from tbe.common.utils.errormgr import error_manager_vector
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import var
from tbe.dsl.base.operation import get_context
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.dynamic.extract_image_patches import tf_get_windowed_output_size_verbose_dynamic


class ExtractImagePatchesWithoutCbuf:
    # 'pylint: disable=too-many-arguments'
    def __init__(self, shape_input_4d, dtype, ksizes, strides, rates, padding, kernel_name, pads=None):
        self.is_const_shape = get_context().get("is_const_shape")
        self.is_binary = get_context().get("is_binary")
        if self.is_binary:
            self.kh = tbe.var("kernel_h_milan", (1, None))
            self.kw = tbe.var("kernel_w_milan", (1, None))
            self.sh = tbe.var("stride_h_milan", (1, None))
            self.sw = tbe.var("stride_w_milan", (1, None))
            self.rh = tbe.var("dilate_h_milan", (1, None))
            self.rw = tbe.var("dilate_w_milan", (1, None))
        else:
            _, self.kh, self.kw, _ = ksizes
            _, self.sh, self.sw, _ = strides
            _, self.rh, self.rw, _ = rates

        if self.is_const_shape:
            self.n, self.hi, self.wi, self.c = shape_input_4d
        else:
            self.n = var("n", (1, None))
            self.hi = var("hi", (1, None))
            self.wi = var("wi", (1, None))
            self.c = var("c", (1, None))

        self.dtype = dtype
        dtype_2_c0_dict = {"int8": 32, "uint8": 32, "int16": 16, "float16": 16, "float":16, "float32":16}
        if self.dtype not in dtype_2_c0_dict:
            error_manager_vector.raise_err_specific_reson(
                "extract_image_patches",
                f"excepted dtype uint8/int8/float16/float32, actual {self.dtype}"
            )
        self.c0 = dtype_2_c0_dict.get(self.dtype)
        self.c1 = (self.c + self.c0 - 1) // self.c0
        self.padding = padding

        if padding in ("SAME", "VALID"):
            self.ho, self.pt, self.pb = \
                tf_get_windowed_output_size_verbose_dynamic(self.hi, self.kh, self.rh, self.sh, padding, kernel_name)
            self.wo, self.pl, self.pr = \
                tf_get_windowed_output_size_verbose_dynamic(self.wi, self.kw, self.rw, self.sw, padding, kernel_name)
            
        else:
            if len(pads) == 1:
                self.pt = pads[0]
                self.pb = pads[0]
                self.pl = pads[0]
                self.pr = pads[0]
            else:
                self.pt, self.pb, self.pl, self.pr = pads
            self.ho = (self.hi + self.pt + self.pb - (self.rh * (self.kh - 1) + 1)) // self.sh + 1
            self.wo = (self.wi + self.pl + self.pr - (self.rw * (self.kw - 1) + 1)) // self.sw + 1

        self.hi_p = self.hi + self.pt + self.pb
        self.wi_p = self.wi + self.pl + self.pr
        self.kernel_name = kernel_name

    def do_without_cbuf(self, env_without_cbuf):
        x = tvm.placeholder((self.n, self.c1, self.hi, self.wi, self.c0), name="x", dtype=self.dtype)
        with tbe.compute():
            x_ub = tvm.compute((self.n, self.c1, self.hi, self.wi, self.c0), lambda *i: x[i], name="x_ub")
            padding_computes, x_p = self._padding(x_ub)
            y_5hd = tvm.compute(
                (self.n, self.c1, self.ho, self.wo, self.kh, self.kw, self.c0),
                lambda i0, i1, i2, i3, i4, i5, i6:
                    x_p[i0, i1, i2 * self.sh + i4 * self.rh, i3 * self.sw + i5 * self.rw, i6],
                name="y_5hd")
            workspace = tvm.compute(
                (self.n, self.c1, self.ho, self.wo, self.kh, self.kw, self.c0),
                lambda *i: y_5hd[i],
                name="workspace")
            workspace_ub = tvm.compute(
                (self.n, self.c1, self.ho, self.wo, self.kh, self.kw, self.c0),
                lambda *i: workspace[i],
                name="workspace_ub")
            y_transform = tvm.compute(
                (self.n, self.ho, self.wo, self.kh, self.kw, self.c1, self.c0),
                lambda i0, i1, i2, i3, i4, i5, i6:
                    workspace_ub[i0, i5, i1, i2, i3, i4, i6],
                name="y_transform")
            y_ub = tvm.compute(
                (self.n, self.ho, self.wo, self.kh, self.kw, self.c),
                lambda i0, i1, i2, i3, i4, i5:
                    y_transform[i0, i1, i2, i3, i4, i5 // self.c0, i5 % self.c0],
                name="y_ub")
            y = tvm.compute(
                (self.n, self.ho, self.wo, self.kh, self.kw, self.c),
                lambda *i: y_ub[i],
                name="y",
                attrs={
                    "computes": (x, x_ub, padding_computes, x_p, y_5hd, workspace, workspace_ub, y_transform, y_ub),
                    "padding": self.padding,
                    "env_without_cbuf": env_without_cbuf,
                    "compute_without_cbuf": True
                })

        with tvm.target.cce():
            sch = tbe.auto_schedule(y)

        return [x, y], sch

    def add_compile_info(self):
        add_compile_info("envWithoutCbuf", True)
        add_compile_info("socVersion", tbe_platform.get_soc_spec("SHORT_SOC_VERSION"))
        add_compile_info("coreNum", tbe_platform.get_soc_spec("CORE_NUM"))
        add_compile_info("SIZE_L1", tbe_platform.get_soc_spec("L1_SIZE"))
        add_compile_info("SIZE_UB", tbe_platform.get_soc_spec("UB_SIZE"))
        add_compile_info("dtypeInput", self.dtype)
        add_compile_info("paddingType", self.padding)
        add_compile_info("isDB", True)
        add_compile_info("isVar", True)
        add_compile_info("isConst", self.is_const_shape)
        add_compile_info("isBinary", self.is_binary)

    def _padding(self, x_ub):
        x_data = tvm.compute(
            (self.n, self.c1, self.hi_p, self.wi_p, self.c0),
            lambda i0, i1, i2, i3, i4:
                tvm.select(i2 >= self.pt,
                           tvm.select(i2 < self.hi + self.pt,
                                      tvm.select(i3 >= self.pl,
                                                 tvm.select(i3 < self.wi + self.pl,
                                                            x_ub[i0, i1, i2 - self.pt, i3 - self.pl, i4])))),
            name="x_data")

        top = tvm.compute(
            (self.n, self.c1, self.hi_p, self.wi_p, self.c0),
            lambda i0, i1, i2, i3, i4:
                tvm.select(i2 < self.pt, tvm.const(0.0, self.dtype)),
            name="top")

        bottom = tvm.compute(
            (self.n, self.c1, self.hi_p, self.wi_p, self.c0),
            lambda i0, i1, i2, i3, i4:
                tvm.select(i2 >= self.pt + self.hi, tvm.const(0.0, self.dtype)),
            name="bottom")

        left = tvm.compute(
            (self.n, self.c1, self.hi_p, self.wi_p, self.c0),
            lambda i0, i1, i2, i3, i4:
                tvm.select(i3 < self.pl, tvm.const(0.0, self.dtype)),
            name="left")

        right = tvm.compute(
            (self.n, self.c1, self.hi_p, self.wi_p, self.c0),
            lambda i0, i1, i2, i3, i4:
                tvm.select(i3 >= self.pl + self.wi, tvm.const(0.0, self.dtype)),
            name="right")

        padding_computes = (x_data, top, bottom, left, right)

        x_p = tvm.compute(
            (self.n, self.c1, self.hi_p, self.wi_p, self.c0),
            lambda i0, i1, i2, i3, i4:
                tvm.select(i2 < self.pt,
                           top[i0, i1, i2, i3, i4],
                           tvm.select(i2 >= self.pt + self.hi,
                                      bottom[i0, i1, i2, i3, i4],
                                      tvm.select(i3 < self.pl,
                                                 left[i0, i1, i2, i3, i4],
                                                 tvm.select(i3 >= self.pl + self.wi,
                                                            right[i0, i1, i2, i3, i4],
                                                            x_data[i0, i1, i2, i3, i4])))),
            name="x_p")

        return padding_computes, x_p
