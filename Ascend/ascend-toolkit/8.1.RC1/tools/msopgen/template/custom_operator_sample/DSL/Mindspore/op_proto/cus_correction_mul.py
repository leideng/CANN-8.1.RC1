import numpy as np
from mindspore import Tensor
import mindspore.context as context
from mindspore.common import dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.ops import prim_attr_register
from mindspore.ops import PrimitiveWithInfer


class CusCorrectionMul(PrimitiveWithInfer):
    """
    Scales the weights with a correction factor to the long term statistics
    prior to quantization. This ensures that there is no jitter in the quantized weights
    due to batch to batch variation.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C)`.
        - **batch_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_std** (Tensor) - Tensor of shape :math:`(C,)`.

    Outputs:
        - **out** (Tensor) - Tensor has the same shape as x.

    Examples:
        >>> cus_correction_mul = CusCorrectionMul()
        >>> input_x = Tensor(np.random.randint(-8, 12, (3, 4)), mindspore.float32)
        >>> batch_std = Tensor(np.array([1.5, 3, 2]), mindspore.float32)
        >>> running_std = Tensor(np.array([2, 1.2, 0.5]), mindspore.float32)
        >>> out = cus_correction_mul(input_x, batch_std, running_std)
    """

    @prim_attr_register
    def __init__(self, channel_axis=0):
        """Initialize correction mul layer"""
        if context.get_context('device_target') == "Ascend":
            from cus_correction_mul_impl import cus_correction_mul
        self.channel_axis = channel_axis
        self.init_prim_io_names(inputs=['x', 'batch_std', 'running_std'],
                                outputs=['out'])

    def infer_shape(self, x_shape, batch_std_shape, running_std_shape):
        validator.check("batch_std shape", batch_std_shape, "running_std shape", running_std_shape, Rel.EQ, self.name)
        validator.check("batch_std_shape[0]", batch_std_shape[0], "x_shape channel size", x_shape[self.channel_axis],
                        Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_type, batch_std_type, running_std_type):
        args = {"x": x_type, "batch_std": batch_std_type, "running_std": running_std_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        return x_type
