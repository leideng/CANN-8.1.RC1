from mindspore.ops import prim_attr_register
from mindspore.ops import PrimitiveWithInfer


class CholeskyTrsm(PrimitiveWithInfer):
    """
    L * LT = A.
    LT * (LT)^-1 = I.
    return (LT)^-1.
    Only compute the res of the diag part of input matrix with dim 128.
    The rank of input tensors must be `2`.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, N)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N // Split_dim, Split_dim, Split_dim)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[256, 256]), mindspore.float32)
        >>> cus_choleskytrsm = CholeskyTrsm()
        >>> output = cus_choleskytrsm(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CusCholeskyTrsm"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        from cholesky_trsm_impl import cholesky_trsm

    def infer_shape(self, data1_shape):
        ll = []
        m, _ = data1_shape
        if m >= 128:
            ll = [m // 128, 128, 128]
        else:
            ll = [1, 64, 64]
        return ll

    def infer_dtype(self, data1_dtype):
        return data1_dtype
