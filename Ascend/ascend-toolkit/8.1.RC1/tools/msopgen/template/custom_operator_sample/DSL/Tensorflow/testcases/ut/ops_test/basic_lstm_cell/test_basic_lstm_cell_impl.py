from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("BasicLSTMCell", "impl.basic_lstm_cell", "basic_lstm_cell")

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def matrix_to_zZ(matrix, shape, dtype):  # m , k
    h = shape[-2]
    w = shape[-1]
    tmp = np.zeros(np.prod(shape), dtype=dtype)
    idx = 0
    if (h == 1):
        if len(shape) > 2:
            for batch in range(np.prod(shape[:-2])):
                for j in range(0, w):
                    tmp[idx] = matrix[batch][0][idx]
                    idx = idx + 1
        else:
            for j in range(0, w):
                tmp[idx] = matrix[0][idx]
                idx = idx + 1
    else:
        if len(shape) > 2:
            for batch in range(np.prod(shape[:-2])):
                for i in range(0, h // 16):
                    for j in range(0, w // 16):
                        for ii in range(0, 16):
                            for jj in range(0, 16):
                                tmp[idx] = matrix[batch][i * 16 + ii][j * 16 + jj]
                                idx = idx + 1
        else:
            for i in range(0, h // 16):
                for j in range(0, w // 16):
                    for ii in range(0, 16):
                        for jj in range(0, 16):
                            tmp[idx] = matrix[i * 16 + ii][j * 16 + jj]
                            idx = idx + 1
    return tmp

def matrix_to_nZ(matrix, shape, dtype):  # k,n
    h = shape[-2]
    w = shape[-1]
    tmp = np.zeros(np.prod(shape), dtype=dtype)
    idx = 0
    if (w == 1):
        if len(shape)>2:
            for batch in range(np.prod(shape[:-2])):
                for i in range(0, h):
                    tmp[idx] = matrix[batch][idx][0]
                    idx = idx + 1
        else:
            for i in range(0, h // 16):
                tmp[idx] = matrix[idx][0]
                idx = idx + 1
    else:
        if len(shape)>2:
            for batch in range(0, np.prod(shape[:-2])):
                for i in range(0, h // 16):
                    for j in range(0, w // 16):
                        for jj in range(0, 16):
                            for ii in range(0, 16):
                                tmp[idx] = matrix[batch][i * 16 + ii][j * 16 + jj]
                                idx = idx + 1
        else:
            for i in range(0, h // 16):
                for j in range(0, w // 16):
                    for jj in range(0, 16):
                        for ii in range(0, 16):
                            tmp[idx] = matrix[i * 16 + ii][j * 16 + jj]
                            idx = idx + 1
    return tmp

def matrix_to_zN(matrix, shape, dtype):  # m, n
    h = shape[-2]
    w = shape[-1]
    tmp = np.zeros(np.prod(shape), dtype=dtype)
    idx = 0
    if len(shape) > 2:
        if (h == 1):
            for batch in range(np.prod(shape[:-2])):
                for j in range(0, w):
                    tmp[idx] = matrix[batch][0][idx]
                    idx = idx + 1
        elif (w == 1):
            for batch in range(np.prod(shape[:-2])):
                for i in range(0, h):
                    tmp[idx] = matrix[batch][idx][0]
                    idx = idx + 1
        else:
            for batch in range(np.prod(shape[:-2])):
                for j in range(0, w // 16):
                    for i in range(0, h // 16):
                        for ii in range(0, 16):
                            for jj in range(0, 16):
                                tmp[idx] = matrix[batch][i * 16 + ii][j * 16 + jj]
                                idx = idx + 1
    else:
        if (h == 1):
            for j in range(0, w):
                tmp[idx] = matrix[0][idx]
                idx = idx + 1
        elif (w == 1):
            for i in range(0, h):
                tmp[idx] = matrix[idx][0]
                idx = idx + 1
        else:
            for j in range(0, w // 16):
                for i in range(0, h // 16):
                    for ii in range(0, 16):
                        for jj in range(0, 16):
                            tmp[idx] = matrix[i * 16 + ii][j * 16 + jj]
                            idx = idx + 1
    return tmp

def maxtrix_zN_reverse(matrix, shape, dtype):
    idx = 0
    j_outer,i_outer,i_inner,j_inner = shape[-4],shape[-3],shape[-2],shape[-1]
    h = i_outer*i_inner
    w = j_outer*j_inner

    if len(shape) is 5:
        batch_shape = shape[0]
        tmp = np.zeros((batch_shape,h,w), dtype=dtype)
        # print((batch_shape,h,w),matrix.shape)
        for batch in range(batch_shape):
            for j in range(0, j_outer):
                for i in range(0, i_outer):
                    for ii in range(0, i_inner):
                        for jj in range(0, j_inner):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix[idx]
                            idx = idx + 1
    elif len(shape) is 4:
        tmp = np.zeros((h,w), dtype=dtype)
        for j in range(0, j_outer):
            for i in range(0, i_outer):
                for ii in range(0, i_inner):
                    for jj in range(0, j_inner):
                        tmp[i * 16 + ii][j * 16 + jj]= matrix[idx]
                        idx = idx + 1
        # print((h,w))

    return tmp


    idx = 0
    if len(shape)==2:
        h = shape[0]*16
        tmp = np.zeros((h,1), dtype=dtype)
        for i in range(0, h // 16):
            tmp[idx][0]= matrix[idx]
            idx = idx + 1
    if len(shape)==3:
        batch = shape[0]
        h = shape[1]*16
        tmp = np.zeros((batch,h,1), dtype=dtype)
        for batch in range(np.prod(shape[:-2])):
            for i in range(0, h):
                tmp[batch][i][0] = matrix[idx]
                idx = idx + 1
    elif len(shape)==4:
        h,w = shape[0]*16,shape[1]*16
        tmp = np.zeros((h,w), dtype=dtype)
        for i in range(0, h // 16):
            for j in range(0, w // 16):
                for jj in range(0, 16):
                    for ii in range(0, 16):
                        tmp[i * 16 + ii][j * 16 + jj]= matrix[idx]
                        idx = idx + 1
    elif len(shape)==5:
        batch = shape[0]
        h,w = shape[1]*16,shape[2]*16
        tmp = np.zeros((batch,h,w), dtype=dtype)
        for batch in range(0, np.prod(shape[:-4])):
            for i in range(0, h // 16):
                for j in range(0, w // 16):
                    for jj in range(0, 16):
                        for ii in range(0, 16):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix[idx]
                            idx = idx + 1
    return tmp

def maxtrix_nZ_reverse(matrix, shape, dtype):

    idx = 0
    i_outer,j_outer,j_inner,i_inner = shape[-4],shape[-3],shape[-2],shape[-1]
    h = i_outer*i_inner
    w = j_outer*j_inner

    if len(shape) is 5:
        batch_shape = shape[0]
        tmp = np.zeros((batch_shape,h,w), dtype=dtype)
        # print((batch_shape,h,w),matrix.shape)
        for batch in range(batch_shape):
            for i in range(0, i_outer):
                for j in range(0, j_outer):
                    for jj in range(0, j_inner):
                        for ii in range(0, i_inner):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix[idx]
                            idx = idx + 1
    elif len(shape) is 4:
        tmp = np.zeros((h,w), dtype=dtype)
        for i in range(0, i_outer):
            for j in range(0, j_outer):
                for jj in range(0, j_inner):
                    for ii in range(0, i_inner):
                        tmp[i * 16 + ii][j * 16 + jj]= matrix[idx]
                        idx = idx + 1
        # print((h,w))

    return tmp


def calc_expect_func(x, h, c, w, b, mask, ct, ht, it, jt, ft, ot, tanhct):

    x_data = maxtrix_zN_reverse(x["value"].flatten(), x["shape"], np.float16)
    h_data = maxtrix_zN_reverse(h["value"].flatten(), h["shape"], np.float16)
    c_data = maxtrix_zN_reverse(c["value"].flatten(), c["shape"], c["dtype"])
    w_data = maxtrix_nZ_reverse(w["value"].flatten(), w["shape"], np.float16)

    mat_res = np.matmul(np.concatenate([x_data, h_data],axis=1), w_data) + b["value"]

    i, j, ff,o = np.split(mat_res, 4, axis=1)
    f = ff + 1.0
    j = np.tanh(j)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    o = sigmoid(o)
    i = sigmoid(i)
    f = sigmoid(f)

    c_t = f * c_data + i * j
    tanh_ct = np.tanh(c_t)
    h_t = o * tanh_ct
    h_t = h_t.astype('float16')

    c_t = matrix_to_zN(c_t, c_t.shape, c_t.dtype).reshape(ct["shape"])
    h_t = matrix_to_zN(h_t, h_t.shape, h_t.dtype).reshape(ht["shape"])
    i = matrix_to_zN(i, i.shape, i.dtype).reshape(it["shape"])
    j = matrix_to_zN(j, j.shape, j.dtype).reshape(jt["shape"])
    f = matrix_to_zN(f, f.shape, f.dtype).reshape(ft["shape"])
    o = matrix_to_zN(o, o.shape, o.dtype).reshape(ot["shape"])
    tanh_ct = matrix_to_zN(tanh_ct, tanh_ct.shape, tanh_ct.dtype).reshape(tanhct["shape"])

    return c_t,h_t,i,j,f,o,tanh_ct

case1 = {"params": [{"shape": (3, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 48),"ori_format": "ND", "param_type":"input"}, #x
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND", "param_type":"input"}, #h
                    {"shape": (2, 4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND", "param_type":"input"}, #c
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND", "param_type":"input"}, #w
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND", "param_type":"input"},  #b
                    None, #mask
                    {"shape": (2, 4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND", "param_type":"output"}, #ct
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND", "param_type":"output"}, #ht
                    {"shape": (2, 4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND", "param_type":"output"}, #it
                    {"shape": (2, 4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND", "param_type":"output"}, #jt
                    {"shape": (2, 4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND", "param_type":"output"}, #ft
                    {"shape": (2, 4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND", "param_type":"output"}, #ot
                    {"shape": (2, 4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND", "param_type":"output"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_1",
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.1, 0.1)}
case2 = {"params": [{"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND", "param_type":"input"}, #x
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND", "param_type":"input"}, #h
                    {"shape": (8, 128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND", "param_type":"input"}, #c
                    {"shape": (16,32,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (256,512),"ori_format": "ND", "param_type":"input"}, #w
                    {"shape": (512,), "dtype": "float32", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "param_type":"input"},  #b
                    None, #mask
                    {"shape": (8, 128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND", "param_type":"output"}, #ct
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND", "param_type":"output"}, #ht
                    {"shape": (8, 128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND", "param_type":"output"}, #it
                    {"shape": (8, 128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND", "param_type":"output"}, #jt
                    {"shape": (8, 128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND", "param_type":"output"}, #ft
                    {"shape": (8, 128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND", "param_type":"output"}, #ot
                    {"shape": (8, 128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND", "param_type":"output"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_2",
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.1, 0.1)}
case3 = {"params": [{"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #x
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #h
                    {"shape": (128, 8,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #c
                    {"shape": (256,512,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (4096,8192),"ori_format": "ND"}, #w
                    {"shape": (8192,), "dtype": "float32", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"},  #b
                    {"shape": (8192,), "dtype": "uint8", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"}, #mask
                    {"shape": (128, 8,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ct
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ht
                    {"shape": (128, 8,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #it
                    {"shape": (128, 8,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #jt
                    {"shape": (128, 8,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ft
                    {"shape": (128, 8,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ot
                    {"shape": (128, 8,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (3, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (1000, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16000),"ori_format": "ND"}, #x
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (1002, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (16032, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_5",
         "expect": RuntimeError,
         "support_expect": True}

case6 = {"params": [{"shape": (1, 1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 16),"ori_format": "ND"}, #x
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #c
                    {"shape": (3, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (48, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_6",
         "expect": RuntimeError,
         "support_expect": True}

case7 = {"params": [{"shape": (1, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16),"ori_format": "ND"}, #x
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #h
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #c
                    {"shape": (101, 400,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (1616, 6400),"ori_format": "ND"}, #w
                    {"shape": (6400,), "dtype": "float32", "format": "ND", "ori_shape": (6400,),"ori_format": "ND"},  #b
                    {"shape": (6400,), "dtype": "uint8", "format": "ND", "ori_shape": (6400,),"ori_format": "ND"}, #mask
                    {"shape": (100, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ct
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ht
                    {"shape": (100, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #it
                    {"shape": (100, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #jt
                    {"shape": (100, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ft
                    {"shape": (100, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ot
                    {"shape": (100, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_7",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": [{"shape": (3, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 48),"ori_format": "ND"}, #x
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 16),"ori_format": "ND"}, #c
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_8",
         "expect": RuntimeError,
         "support_expect": True}

case9 = {"params": [{"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #x
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 64),"ori_format": "ND"}, #h
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #c
                    {"shape": (16,32,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (256,512),"ori_format": "ND"}, #w
                    {"shape": (512,), "dtype": "float16", "format": "ND", "ori_shape": (512,),"ori_format": "ND"},  #b
                    {"shape": (512,), "dtype": "uint8", "format": "ND", "ori_shape": (512,),"ori_format": "ND"}, #mask
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ct
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ht
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #it
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #jt
                    {"shape":(8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ft
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ot
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_9",
         "expect": RuntimeError,
         "support_expect": True}

case10 = {"params": [{"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #x
                     {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #h
                     {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #c
                     {"shape": (256,512,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (4096,8192),"ori_format": "ND"}, #w
                     {"shape": (8192,), "dtype": "float16", "format": "ND", "ori_shape": (8190,),"ori_format": "ND"},  #b
                     {"shape": (8192,), "dtype": "uint8", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"}, #mask
                     {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ct
                     {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ht
                     {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #it
                     {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #jt
                     {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ft
                     {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ot
                     {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #tanhct
                     ],
          "case_name": "BasicLSTMCell_10",
          "expect": RuntimeError,
          "support_expect": True}

case11 = {"params": [{"shape": (3, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                     {"shape": (5, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                     {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                     {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #it
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #jt
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ft
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ot
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #tanhct
                     ],
          "case_name": "BasicLSTMCell_11",
          "expect": "success",
          "support_expect": True}

case12 = {"params": [{"shape": (1000, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16000),"ori_format": "ND"}, #x
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                     {"shape": (1002, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (16032, 128),"ori_format": "ND"}, #w
                     {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                     {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #it
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #jt
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ft
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ot
                     {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #tanhct
                     ],
          "case_name": "BasicLSTMCell_12",
          "expect": "success",
          "support_expect": True}

case13 = {"params": [{"shape": (1, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 16),"ori_format": "ND"}, #x
                     {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #h
                     {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #c
                     {"shape": (3, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (48, 128),"ori_format": "ND"}, #w
                     {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                     {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                     {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ct
                     {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ht
                     {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #it
                     {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #jt
                     {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ft
                     {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ot
                     {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #tanhct
                     ],
          "case_name": "BasicLSTMCell_13",
          "expect": "success",
          "support_expect": True}

case14 = {"params": [{"shape": (1, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16),"ori_format": "ND"}, #x
                     {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #h
                     {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #c
                     {"shape": (101, 400,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (1616, 6400),"ori_format": "ND"}, #w
                     {"shape": (6400,), "dtype": "float16", "format": "ND", "ori_shape": (6400,),"ori_format": "ND"},  #b
                     {"shape": (6400,), "dtype": "uint8", "format": "ND", "ori_shape": (6400,),"ori_format": "ND"}, #mask
                     {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ct
                     {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ht
                     {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #it
                     {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #jt
                     {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ft
                     {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ot
                     {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #tanhct
                     ],
          "case_name": "BasicLSTMCell_14",
          "expect": "success",
          "support_expect": True}

ut_case.add_precision_case("Ascend910A", case1)
ut_case.add_precision_case("Ascend910A", case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)
ut_case.add_case(["Ascend910A"], case9)
ut_case.add_case(["Ascend910A"], case10)
# ut_case.add_case(["Ascend910"], case11)
# ut_case.add_case(["Ascend910"], case12)
# ut_case.add_case(["Ascend910"], case13)
# ut_case.add_case(["Ascend910"], case14)

