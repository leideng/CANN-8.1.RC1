/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file swin_batchmatmul.h
 * \brief
 */
#ifndef __SWIN_BATCH_MATMUL_H__
#define __SWIN_BATCH_MATMUL_H__

#include "swin_util.h"

namespace matmul {
using namespace AscendC;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE> class BatchMatmulImpl {
    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;

public:
    __aicore__ inline BatchMatmulImpl() {};
    __aicore__ inline void Init(uint32_t m, uint32_t n, uint32_t k, uint32_t b, TPipe* tpipe, int8_t event_id, int8_t l1_buffer_cnt = 1);
    __aicore__ inline void CopyTensorA(const GlobalTensor<SrcT>& gm, bool isTransposeA = false, int buf_idx = 0);
    __aicore__ inline void CopyTensorB(const GlobalTensor<SrcT>& gm, bool isTransposeB = false, int buf_idx = 0);
    __aicore__ inline void SetBias(const GlobalTensor<BiasT>& biasGlobal);

    template <bool sync = true> __aicore__ inline bool Iterate(bool enPartialSum = false, int buf_idx = 0);
    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& gm, bool enAtomicAdd = false,
        bool enSequentialWrite = false);
    __aicore__ inline void End() {}

private:
    __aicore__ inline bool OnCopyInA1(const LocalTensor<SrcT>& aMatrix);
    __aicore__ inline bool OnCopyInA1Trans(const LocalTensor<SrcT>& aMatrix);
    __aicore__ inline bool OnCopyInB1(const LocalTensor<SrcT>& bMatrix);
    __aicore__ inline bool OnCopyInB1Trans(const LocalTensor<SrcT>& bMatrix);
    __aicore__ inline void OnLoadInA2(const LocalTensor<SrcT>& dst, const LocalTensor<SrcT>& aMatrix);
    __aicore__ inline void OnLoadInB2(const LocalTensor<SrcT>& dst, const LocalTensor<SrcT>& bMatrix);
    __aicore__ inline void CopyND2NZ(const LocalTensor<SrcT>& dst, GlobalTensor<SrcT> src, const int height, const int width);

private:
    static const int num_L1_buffers = 2;
    TQue<TPosition::A1, num_L1_buffers, GetNdNzMask(CubeFormat::NZ, A_TYPE::format)> L1A_Que;
    TQue<TPosition::B1, num_L1_buffers, GetNdNzMask(CubeFormat::NZ, B_TYPE::format)> L1B_Que;
    TQue<TPosition::A2, QUEUE_DEPTH> L0A_Que;
    TQue<TPosition::B2, QUEUE_DEPTH> L0B_Que;
    TQue<TPosition::CO1, QUEUE_DEPTH> L0C_Que;

    LocalTensor<SrcT> L0_aMatrix;
    LocalTensor<SrcT> L0_bMatrix;
    LocalTensor<L0cT> L0_cMatrix;
    LocalTensor<SrcT> L1_aMatrix[4];
    LocalTensor<SrcT> L1_bMatrix[4];
    GlobalTensor<SrcT> aGlobal;
    GlobalTensor<SrcT> bGlobal;

    int baseM;
    int baseN;
    int baseK;
    int batch;
    int baseMK;
    int baseNK;
    int baseMN;
    int eid;

    bool isTransposeA_;
    bool isTransposeB_;

    constexpr static int L1Size_ = 512 * 1024;
    constexpr static int L0CSize_ = 128 * 1024;
    constexpr static int L0ASize_ = 64 * 1024;
    constexpr static int L0BSize_ = 64 * 1024;
    constexpr static int32_t factor_ = AuxGetFactor<SrcT>();
    constexpr static int32_t c0Size_ = AuxGetC0Size<SrcT>();
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::Init(uint32_t m,
    uint32_t n, uint32_t k, uint32_t b, TPipe* tpipe, int8_t event_id, int8_t L1_buf_cnt) {
    baseM = m;
    baseN = n;
    baseK = k;
    baseMK = baseM * baseK;
    baseNK = baseN * baseK;
    baseMN = baseM * baseN;
    batch = b;
    tpipe->InitBuffer(L1A_Que, L1_buf_cnt, batch * baseMK * sizeof(SrcT));
    tpipe->InitBuffer(L1B_Que, L1_buf_cnt, batch * baseNK * sizeof(SrcT));

    for (int i = 0;i < L1_buf_cnt; i++) {
        L1_aMatrix[i] = L1A_Que.template AllocTensor<SrcT>();
        L1_bMatrix[i] = L1B_Que.template AllocTensor<SrcT>();
    }

    uint32_t shareLens[3] = {static_cast<uint32_t>(0), static_cast<uint32_t>(L0CSize_), static_cast<uint32_t>(0)};
    InitShareBufStart(tpipe, 0, shareLens, 3, 0);
    tpipe->InitBuffer(L0A_Que, 1, batch * baseMK * sizeof(SrcT));
    tpipe->InitBuffer(L0B_Que, 1, batch * baseNK * sizeof(SrcT));
    tpipe->InitBuffer(L0C_Que, 1, batch * baseMN * sizeof(L0cT));

    L0_aMatrix = L0A_Que.template AllocTensor<SrcT>();
    L0_bMatrix = L0B_Que.template AllocTensor<SrcT>();
    L0_cMatrix = L0C_Que.template AllocTensor<L0cT>();
    L0_cMatrix.SetSize(batch * baseMN * CUBE_MAX_SIZE);
    InitShareBufEnd(tpipe);
    eid = event_id;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
template <bool sync>
__aicore__ inline bool BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::Iterate(bool enPartialSum, int buf_idx) {
    OnLoadInA2(L0_aMatrix, L1_aMatrix[buf_idx]);
    OnLoadInB2(L0_bMatrix, L1_bMatrix[buf_idx]);

    set_flag(PIPE_MTE1, PIPE_M, eid);
    wait_flag(PIPE_MTE1, PIPE_M, eid);

    MmadParams mmadParams;
    mmadParams.m = baseM;
    mmadParams.n = baseN;
    mmadParams.isBias = 0; // enPartialSum; // 缺省为false, 表示清0
    mmadParams.k = baseK;

    for (int i = 0;i < batch; i++) {
        Mmad(L0_cMatrix[i * baseMN], L0_aMatrix[i * baseMK], L0_bMatrix[i * baseNK], mmadParams);
    }
    set_flag(PIPE_M, PIPE_FIX, eid);
    wait_flag(PIPE_M, PIPE_FIX, eid);
    return true;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::CopyTensorA(const GlobalTensor<SrcT>& gm,
    bool isTransposeA, int buf_idx) {
    aGlobal = gm;
    isTransposeA_ = isTransposeA;
    OnCopyInA1(L1_aMatrix[buf_idx]);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::CopyTensorB(const GlobalTensor<SrcT>& gm,
    bool isTransposeB, int buf_idx) {
    bGlobal = gm;
    isTransposeB_ = isTransposeB;
    OnCopyInB1(L1_bMatrix[buf_idx]);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::CopyND2NZ(const LocalTensor<SrcT>& dst,
    GlobalTensor<SrcT> src, const int height, const int width)
{
    ASSERT(height > 0);
    ASSERT(width > 0);

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = batch;
    nd2nzParams.nValue = height;
    nd2nzParams.dValue = width;
    nd2nzParams.srcNdMatrixStride = width * height;
    nd2nzParams.srcDValue = width;
    nd2nzParams.dstNzC0Stride = Ceil(height, BLOCK_CUBE) * BLOCK_CUBE;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = width * height;
    DataCopy(dst, src, nd2nzParams);
    return;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline bool BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::OnCopyInA1(const LocalTensor<SrcT>& aMatrix)
{
    if (isTransposeA_) {
        return OnCopyInA1Trans(aMatrix);
    } else {
        if constexpr (A_TYPE::format == CubeFormat::ND) {
            CopyND2NZ(aMatrix, aGlobal, baseM, baseK);
        } else {
            return false;
        }
    }
    return true;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline bool BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::OnCopyInA1Trans(const LocalTensor<SrcT>& aMatrix)
{
    if constexpr (A_TYPE::format == CubeFormat::ND) {
        CopyND2NZ(aMatrix, aGlobal, baseK, baseM);
    } else {
        return false;
    }
    return true;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline bool BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::OnCopyInB1(const LocalTensor<SrcT>& bMatrix)
{
    if (isTransposeB_) {
        return OnCopyInB1Trans(bMatrix);
    } else {
        if constexpr (B_TYPE::format == CubeFormat::ND) {
            CopyND2NZ(bMatrix, bGlobal, baseK, baseN);
        } else {
            return false;
        }
    }
    return true;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline bool BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::OnCopyInB1Trans(const LocalTensor<SrcT>& bMatrix)
{
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        CopyND2NZ(bMatrix, bGlobal, baseN, baseK);
    } else {
        return false;
    }
    return true;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::OnLoadInA2(const LocalTensor<SrcT>& dst,
    const LocalTensor<SrcT>& aMatrix)
{   // check: a2 is not correct
    int blockUseM_ = Ceil(baseM, BLOCK_CUBE);
    int blockUseK_ = Ceil(baseK, c0Size_);
    int singleCoreK_ = baseK;
    int singleCoreM_ = baseM;
    if (isTransposeA_) {
        if constexpr (sizeof(SrcT) == sizeof(float)) {
            // float is not supported
            return;
        } else {
            LoadData2dParams loadDataParams;
            int dstOffset = blockUseK_ * CUBE_MAX_SIZE / factor_;
            int srcOffset = singleCoreK_ * c0Size_;
            if constexpr (!(A_TYPE::pos == TPosition::A1 || A_TYPE::pos == TPosition::B1 || A_TYPE::pos == TPosition::C1 || A_TYPE::pos == TPosition::SHM || A_TYPE::pos == TPositio n::TSCM)) {
                srcOffset = blockUseK_ * c0Size_ * BLOCK_CUBE;
            }
            loadDataParams.repeatTimes = blockUseK_;
            loadDataParams.srcStride = 1;
            loadDataParams.ifTranspose = true;

            if (blockUseK_ == 1) {
                loadDataParams.repeatTimes = blockUseM_ * batch;
                loadDataParams.srcStride = 1;
                load_cbuf_to_ca((__ca__ SrcT*)(dst.GetPhyAddr()), (__cbuf__ SrcT*)(aMatrix.GetPhyAddr()),
                    loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride,
                    loadDataParams.dstGap, loadDataParams.sid, 1, inc);
            } else {
                for (int i = 0; i < blockUseM_ * batch; i++) {
                    load_cbuf_to_ca((__ca__ SrcT*)(dst[i * dstOffset].GetPhyAddr()),
                        (__cbuf__ SrcT*)(aMatrix[i * srcOffset].GetPhyAddr()), loadDataParams.startIndex,
                        loadDataParams.repeatTimes, loadDataParams.srcStride, loadDataParams.dstGap, loadDataParams.sid,
                        1, inc);
                }
            }
        }
    } else {
        LoadData2dParams loadDataParams;
        int dstOffset = blockUseK_ * CUBE_MAX_SIZE / factor_;
        int srcOffset = CUBE_MAX_SIZE / factor_;
        loadDataParams.repeatTimes = blockUseK_;
        if constexpr (A_TYPE::pos == TPosition::A1 || A_TYPE::pos == TPosition::B1 || A_TYPE::pos == TPosition::C1 || A_TYPE::pos == TPosition::SHM || A_TYPE::pos == TPosition::TSCM) {
            // alL A matrix is in L1 buffer
            loadDataParams.srcStride = Ceil(singleCoreM_, BLOCK_CUBE);
        } else {
            loadDataParams.srcStride = blockUseM_;
        }
        loadDataParams.ifTranspose = false;

        if (blockUseK_ == 1) {
            loadDataParams.repeatTimes = blockUseM_ * batch;
            loadDataParams.srcStride = 1;
            load_cbuf_to_ca((__ca__ SrcT*)(dst.GetPhyAddr()), (__cbuf__ SrcT*)(aMatrix.GetPhyAddr()),
                loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride, loadDataParams.dstGap,
                loadDataParams.sid, 0, inc);
        } else {
            // check: design loop on M K shape
            loadDataParams.dstGap = loadDataParams.repeatTimes - 1;
            loadDataParams.repeatTimes = blockUseM_;
            loadDataParams.srcStride = 1;
            srcOffset = CUBE_MAX_SIZE / factor_;
            dstOffset = blockUseM_ * CUBE_MAX_SIZE / factor_;
            for (int b = 0;b < batch; b++) {
                for (int i = 0; i < blockUseK_; i++) {
                    load_cbuf_to_ca((__ca__ SrcT*)(dst[srcOffset * i + b * baseMK].GetPhyAddr()),
                        (__cbuf__ SrcT*)(aMatrix[dstOffset * i + b * baseMK].GetPhyAddr()), loadDataParams.startIndex,
                        loadDataParams.repeatTimes, loadDataParams.srcStride, loadDataParams.dstGap, loadDataParams.sid, 0,
                        inc);
                }
            }
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::OnLoadInB2(const LocalTensor<SrcT>& dst,
    const LocalTensor<SrcT>& bMatrix)
{
    int blockUseK_ = Ceil(baseK, c0Size_);
    int blockUseN = Ceil(baseN, BLOCK_CUBE);
    int singleCoreK_ = baseK;
    int singleCoreN_ = baseN;
    if (isTransposeB_) {
        LoadData2dParams loadDataParams;
        int dstOffset = blockUseN * CUBE_MAX_SIZE / factor_;
        int srcOffset = singleCoreN_ * c0Size_;
        if constexpr (!(B_TYPE::pos == TPosition::A1 || B_TYPE::pos == TPosition::B1 || B_TYPE::pos == TPosition::C1 || B_TYPE::pos == TPosition::SHM || B_TYPE::pos == TPosition::TSCM)) {
            srcOffset = blockUseN * BLOCK_CUBE * c0Size_;
        }
        loadDataParams.repeatTimes = blockUseN;
        loadDataParams.srcStride = 1;
        loadDataParams.ifTranspose = false;

        if (blockUseN == 1) {
            loadDataParams.repeatTimes = blockUseK_ * batch;
            loadDataParams.srcStride = 1;
            load_cbuf_to_cb((__cb__ SrcT*)(dst.GetPhyAddr()), (__cbuf__ SrcT*)(bMatrix.GetPhyAddr()),
                loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride, loadDataParams.dstGap,
                loadDataParams.sid, 0, inc);
        } else {
            // check: if there are no tails
            loadDataParams.repeatTimes = blockUseN * blockUseK_ * batch;
            load_cbuf_to_cb((__cb__ SrcT*)(dst.GetPhyAddr()),
                (__cbuf__ SrcT*)(bMatrix.GetPhyAddr()), loadDataParams.startIndex,
                loadDataParams.repeatTimes, loadDataParams.srcStride, loadDataParams.dstGap, loadDataParams.sid, 0, inc);
        }
    } else {
        if constexpr (sizeof(SrcT) == sizeof(float)) {
            return;
        } else {
            LoadData2dParams loadDataParams;
            int dstOffset = blockUseN * CUBE_MAX_SIZE;
            constexpr int srcOffset = CUBE_MAX_SIZE;
            loadDataParams.repeatTimes = blockUseN;
            if constexpr (B_TYPE::pos == TPosition::A1 || B_TYPE::pos == TPosition::B1 || B_TYPE::pos == TPosition::C1 || B_TYPE::pos == TPosition::SHM || B_TYPE::pos == TPosition::TSCM) {
                // alL B matrix is in L1 buffer
                loadDataParams.srcStride = Ceil(singleCoreK_, BLOCK_CUBE);
            } else {
                loadDataParams.srcStride = blockUseK_;
            }
            loadDataParams.ifTranspose = true;
            if (blockUseN == 1) {
                loadDataParams.repeatTimes = blockUseK_ * batch;
                loadDataParams.srcStride = 1;
                load_cbuf_to_cb((__cb__ SrcT*)(dst.GetPhyAddr()), (__cbuf__ SrcT*)(bMatrix.GetPhyAddr()),
                    loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride,
                    loadDataParams.dstGap, loadDataParams.sid, 1, inc);
            } else {
                loadDataParams.dstGap = loadDataParams.repeatTimes - 1;
                loadDataParams.repeatTimes = blockUseK_;
                loadDataParams.srcStride = 1;
                dstOffset = blockUseK_ * CUBE_MAX_SIZE;

                for (int b = 0; b < batch; b++) {
                    for (int i = 0;i < blockUseN; i++) {
                        load_cbuf_to_cb((__cb__ SrcT*)(dst[i * srcOffset + b * baseNK].GetPhyAddr()),
                            (__cbuf__ SrcT*)(bMatrix[i * dstOffset + b * baseNK].GetPhyAddr()), loadDataParams.startIndex,
                            loadDataParams.repeatTimes, loadDataParams.srcStride, loadDataParams.dstGap, loadDataParams.sid,
                            1, inc);
                    }
                }
            }
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
template <bool sync>
__aicore__ inline void BatchMatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::GetTensorC(const GlobalTensor<DstT>& gm,
    bool enAtomicAdd, bool enSequentialWrite)
{
    const uint16_t baseUseM_ = baseM;
    const uint16_t baseUseN_ = baseN;
    auto blockUseM_ = Ceil(baseUseM_, BLOCK_CUBE);
    auto blockUseN_ = Ceil(baseUseN_, BLOCK_CUBE);
    auto enUnitFlag_ = false;
    auto batch_ = batch;
    auto co1Local = L0_cMatrix;
    auto baseMN_ = baseMN;
    auto curM_ = 0;
    auto curN_ = 0;

    if (enAtomicAdd) {
        SetAtomicAdd<DstT>();
    }
    int32_t dimN = baseN;
    int blockCount = sizeof(L0cT) == B32_BYTE_SIZE ? BLOCK_CUBE : ONE_BLK_SIZE / sizeof(L0cT);
    if constexpr (C_TYPE::format == CubeFormat::ND_ALIGN) {
        dimN = Ceil(baseN, blockCount) * blockCount;
    }

    if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
        uint16_t batch_cast = (uint16_t) batch;
        uint32_t src_nd_size = baseMN * sizeof(L0cT);
        uint32_t fractal_size = 1024;
        uint16_t src_nd_stride = (uint16_t)(src_nd_size / fractal_size);
        uint16_t dst_nd_stride = baseMN;
        if (enSequentialWrite) {
            // count, len, src, dst
            FixpipeParams<L0cT> fixpipeParams(blockUseN_,
                static_cast<uint16_t>(baseUseM_ * BLOCK_CUBE * sizeof(L0cT) / ONE_BLK_SIZE), 0, baseUseN_);
            fixpipeParams.nz2ndParams = { true, batch_cast, src_nd_stride, dst_nd_stride, baseUseN_ };
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantParams = { halfQuantParams };
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantParams = { bf16QuantParams };
            }
            if (enUnitFlag_) {
                fixpipeParams.unitFlag = 3;
            }
            Fixpipe(gm, co1Local, fixpipeParams);
        } else {
            int dstOffset = curM_ * baseM * dimN + curN_ * baseN;
            FixpipeParams<L0cT> fixpipeParams(blockUseN_,
                static_cast<uint16_t>(baseUseM_ * BLOCK_CUBE * sizeof(L0cT) / ONE_BLK_SIZE), 0, dimN);
            fixpipeParams.nz2ndParams = { true, batch_cast, src_nd_stride, dst_nd_stride, baseUseN_ };
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantParams = { halfQuantParams };
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantParams = { bf16QuantParams };
            }
            if (enUnitFlag_) {
                fixpipeParams.unitFlag = 3;
            }
            Fixpipe(gm, co1Local, fixpipeParams);
        }
    } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
        if (enSequentialWrite) {
            FixpipeParams<L0cT> fixpipeParams(blockUseN_,
                static_cast<uint16_t>(baseUseM_ * BLOCK_CUBE * sizeof(L0cT) / ONE_BLK_SIZE), 0,
                static_cast<uint16_t>((blockUseM_ * BLOCK_CUBE - baseUseM_) * BLOCK_CUBE * sizeof(DstT) /
                ONE_BLK_SIZE));
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantParams = { halfQuantParams };
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantParams = { bf16QuantParams };
            }
            if (enUnitFlag_) {
                fixpipeParams.unitFlag = 3;
            }
            Fixpipe(gm, co1Local, fixpipeParams);
        } else {
            int dstOffset = curN_ * baseN * baseM + curM_ * baseM * BLOCK_CUBE;
            FixpipeParams<L0cT> fixpipeParams(blockUseN_,
                static_cast<uint16_t>(baseUseM_ * BLOCK_CUBE * sizeof(L0cT) / ONE_BLK_SIZE), 0,
                static_cast<uint16_t>((baseM - baseUseM_) * BLOCK_CUBE * sizeof(DstT) / ONE_BLK_SIZE));
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantParams = { halfQuantParams };
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantParams = { bf16QuantParams };
            }
            if (enUnitFlag_) {
                fixpipeParams.unitFlag = 3;
            }
            Fixpipe(gm[dstOffset], co1Local, fixpipeParams);
        }
    } else {
        ASSERT(false && "Data format of C matrix should be ND, ND_ALIGN or NZ.");
    }

    if (enAtomicAdd) {
        SetAtomicNone();
    }
}

} // namespace matmul

#endif
