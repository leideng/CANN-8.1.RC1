"""
scatter
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tik
from impl.dynamic.scatter_kv_cache import ScatterKvCacheDynImpl
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform


class Constant:
    """
    The class for constant
    """
    TILING_ARG_NUM = 22
    MAX_INT64 = 2**63 - 1


# 'pylint: disable=too-many-locals, too-many-arguments
def check_supported(var, indices, updates, var_out, reduce, axis=0, kernel_name="scatter"):
    """
        check the op support situation.
    """
    var_dtype = var.get("dtype")
    updates_dtype = updates.get("dtype")
    out_dtype = var_out.get("dtype")

    if var_dtype != updates_dtype or var_dtype != out_dtype:
        reason = f"var_dtype is {var_dtype}, updates_dtype is {updates_dtype}, out_dtype is {out_dtype}"
        return False, reason
    return "Unknown"


class Scatter:
    """
    The class for scatter
    """
    def __init__(self, data, indices, updates, result, axis, reduction, kernel_name) -> None:
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.tiling_param_dtype = "int64"
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_param_dtype, (Constant.TILING_ARG_NUM,),
                                                  name='tiling_gm',
                                                  scope=tik.scope_gm)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.total_core_number = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.support_data_move_pad = tbe_platform.api_check_support("tik.data_move_pad")
        self.dtype_data = data.get("dtype").lower()
        self.dtype_indices = indices.get("dtype").lower()
        self.dtype_updates = updates.get("dtype").lower()
        self.dtype_out = result.get("dtype").lower()
        if self.dtype_data == "bfloat16":
            self.dtype_data = "float16"
            self.dtype_updates = "float16"
            self.dtype_out = "float16"
        elif self.dtype_data == "int32":
            self.dtype_data = "float32"
            self.dtype_updates = "float32"
            self.dtype_out = "float32"

        self.data_gm = self.tik_instance.Tensor(self.dtype_data, [Constant.MAX_INT64],
                                                name="data_gm",
                                                scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.dtype_indices, [Constant.MAX_INT64],
                                                   name="indices_gm",
                                                   scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.dtype_updates, [Constant.MAX_INT64],
                                                   name="updates_gm",
                                                   scope=tik.scope_gm)
        self.result_gm = self.tik_instance.Tensor(self.dtype_out, [Constant.MAX_INT64],
                                                  name="result_gm",
                                                  scope=tik.scope_gm)
        self.tiling_ub = self.tik_instance.Tensor(self.tiling_param_dtype, (Constant.TILING_ARG_NUM,),
                                                  name='tiling_ub',
                                                  scope=tik.scope_ubuf)
        self.tiling_mode = self.tik_instance.Scalar(self.tiling_param_dtype, name='tiling_mode')
        self.used_aicore_num = self.tik_instance.Scalar(self.tiling_param_dtype, name='used_aicore_num')

    def get_tilings(self):
        """
        get_tilings
        """
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, (Constant.TILING_ARG_NUM + 3) // 4, 0,
                                    0)  # 4 for int64
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.used_aicore_num.set_as(self.tiling_ub[1])

    def compute(self):
        self.get_tilings()
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            obj_kvcache = ScatterKvCacheDynImpl(self)
            obj_kvcache.compute(i)

        tbe_context.get_context().add_compile_info('vars',
                                                   {'core_num': self.total_core_number,
                                                    'ub_size': self.ub_size_bytes,
                                                    'support_data_move_pad': self.support_data_move_pad})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.data_gm, self.indices_gm, self.updates_gm],
                                   outputs=[self.result_gm],
                                   flowtable=[self.tiling_gm])


# 'pylint: disable=unused-argument,too-many-arguments
@register_operator("Scatter")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def scatter(var, indices, updates, var_out, reduce, axis=0, kernel_name="scatter", impl_mode="high_precision"):
    """
    scatter_mul interface

    Parameters
    ----------
    var_dict: input var shape, dtype and range
    indices_dict: input indices shape, dtype and range
    updates_dict: input updates shape, dtype and range
    var_out_dict: output shape, dtype and range
    reduce: type of scatter op, support "update", "add", "mul"
    kernel_name: kernel name of scatter op
    impl_mode: high_precision or high_performance
    Returns
    -------
    compile info
    """
    obj = Scatter(var, indices, updates, var_out, axis, reduce, kernel_name)
    return obj.compute()
