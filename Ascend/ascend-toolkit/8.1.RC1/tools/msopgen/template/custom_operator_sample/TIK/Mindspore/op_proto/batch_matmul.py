import numpy as np
from mindspore import Tensor
from mindspore.ops import prim_attr_register
from mindspore.ops import PrimitiveWithInfer


class BatchMatmul(PrimitiveWithInfer):
    """
    Multiplies matrix `a` by matrix `b` in batch.

    The rank of input tensors must be `3`.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, D, D)`.
        - **input_y** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(N, D, D)`. If
          `transpose_b` is True.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, D, D)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[2, 128, 128]), mindspore.float32)
        >>> input_y = Tensor(np.ones(shape=[2, 128, 128]), mindspore.float32)
        >>> cus_batch_matmul = BatchMatmul()
        >>> output = cus_batch_matmul(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False):
        """Initialize BatchMatMul"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])
        from batch_matmul_impl import batch_matmul

    def infer_shape(self, data1_shape, data2_shape):
        return data1_shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return data1_dtype