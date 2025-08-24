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
 * \file apply_came_part2.h
 * \brief
 */
#ifndef _APPLY_CAME_PART2_H_
#define _APPLY_CAME_PART2_H_

#include "kernel_operator.h"
#include "apply_came_part2_common.h"

using namespace AscendC;

template <typename T>
class ApplyCamePart2 {
public:
    __aicore__ inline ApplyCamePart2() {};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR sumGradR, GM_ADDR sumGradC,
                                GM_ADDR sumGradRc, GM_ADDR rIn, GM_ADDR cIn, GM_ADDR beta2,
                                GM_ADDR sumR, GM_ADDR globalShape,
                                GM_ADDR rOut, GM_ADDR cOut, GM_ADDR u, GM_ADDR sumSquareU,
                                GM_ADDR workspace, const ApplyCamePart2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCoreR();
    __aicore__ inline void ProcessTailCoreR();
    __aicore__ inline void ProcessPerCoreC();
    __aicore__ inline void ProcessTailCoreC();
    __aicore__ inline void ProcessPerCoreU();
    __aicore__ inline void ProcessTailCoreU();
    __aicore__ inline void CopyTilingData(const ApplyCamePart2TilingData* tilingData);
    __aicore__ inline void CopyInScalar();
    __aicore__ inline void ProcessR();
    __aicore__ inline void ProcessC();
    __aicore__ inline void ProcessU();
    __aicore__ inline void CopyInR(int loopIdx, int64_t dataCount, bool tail);
    __aicore__ inline void ComputeR(int64_t dataCount);
    __aicore__ inline void CopyOutR(int loopIdx, int64_t dataCount);
    __aicore__ inline void CopyInC(int loopIdx, int64_t dataCount, bool tail);
    __aicore__ inline void ComputeC(int64_t dataCount);
    __aicore__ inline void CopyOutC(int loopIdx, int64_t dataCount);
    __aicore__ inline void CopyInGrad(int rLoopIdx, int cLoopIdx,
                                      int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void CopyInNotAlignedGrad(int rLoopIdx, int cLoopIdx,
                                                int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void CopyInUr(int loopIdx, int64_t dataCount);
    __aicore__ inline void CopyOutU(int rLoopIdx, int cLoopIdx,
                                    int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void CopyInUc(int loopIdx, int64_t dataCount);
    __aicore__ inline void ComputeU(int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void CopyInNotAlignedUc(int loopIdx, int64_t dataCount);
    __aicore__ inline void CopyInNotAlignedUr(int loopIdx, int64_t dataCount);
    __aicore__ inline void ReduceSumU(LocalTensor<T> &src, int64_t dataCount,
                                      int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void CalcU(int rLoopIdx, int cLoopIdx,
                                 int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC,
                                                       const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);
    __aicore__ inline void BroadcastR(LocalTensor<T> &dst, LocalTensor<T> &src, int64_t numR, int64_t numC);
    __aicore__ inline void TransposeR(LocalTensor<T> &dst, LocalTensor<T> &src, int64_t numR, int64_t numC);
    __aicore__ inline void MulRC(LocalTensor<T> &dst, LocalTensor<T> &r, LocalTensor<T> &c,
                                  int64_t numR, int64_t numC);
    __aicore__ inline void CalcRcCycleMode(LocalTensor<T> &dst, LocalTensor<T> &src,
                                           LocalTensor<T> &srcScalar,
                                           int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline int64_t GetMaxCoreNumToUse();
    __aicore__ inline void CopyInNotAlignedR(int loopIdx, int64_t dataCount);
    template <typename T1, typename T2> __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
        if (b == 0) {
            return 0;
        }

        return (a + b - 1) / b;
    };

    template <typename T1, typename T2> __aicore__ inline T1 Max(T1 a, T2 b) {
        return a > b ? a : b;
    };

    __aicore__ inline void CalcWorkLocal();
    __aicore__ inline void InitTque();

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueR_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueSumGradR_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueC_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueSumGradC_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueGrad_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueU_; // shape: (n, m)
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueSumSquareU_; // tensor
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueR_; // shape: (n)
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueC_; // shape: (m)
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueSumR_; // scalar

    // each tbuf may be used for multiple purposes
    TBuf<QuePosition::VECCALC> ub1Buf_; // 1. rMatrix after broadcast from r
    TBuf<QuePosition::VECCALC> ub2Buf_; // 1. rMatrix after transpose; 2. r * c
    TBuf<QuePosition::VECCALC> ub3Buf_; // grad

    TBuf<QuePosition::VECCALC> scalarBuf_;

    GlobalTensor<T> rInGm_;
    GlobalTensor<T> sumGradRInGm_;
    GlobalTensor<T> cInGm_;
    GlobalTensor<T> sumGradCInGm_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> rOutGm_;
    GlobalTensor<T> cOutGm_;
    GlobalTensor<T> sumSquareUGm_;
    GlobalTensor<T> uGm_;

    GlobalTensor<T> sumGradRcGm_;
    GlobalTensor<T> beta2Gm_;
    GlobalTensor<T> sumRGm_;
    GlobalTensor<int64_t> globalShapeGm_;

    // for SyncAll
    GlobalTensor<int32_t> syncGlobal_;
    TQue<QuePosition::VECIN, 1> workQueForSyncAll_;

    // for ReduceSum
    TQue<QuePosition::VECOUT, BUFFER_NUM> reduceSumWorkQueue_;
    GlobalTensor<T> sumRWorkspace_;
    GlobalTensor<float> sumSquareUWorkspace_;
    int64_t workLocalNeedSize_;

    bool isInputSumR_; // is sum_r in input null?
    bool isInputGlobalShape;

    // scalar
    float sumGradRc_;
    float beta2_;
    float sumR_;
    float M_;
    float N_;

    float rcCoefficient_;
    int64_t blockIdx_;

    // tiling data
    ApplyCamePart2TilingData tilingData_;

    // block offset
    int64_t rBlockOffset_;
    int64_t cBlockOffset_;
    int64_t rRcBlockOffset_;
    int64_t sumUOffset_;

    const int64_t NUM_PER_BLOCK = BLOCK_SIZE / sizeof(T);

    GM_ADDR workspace_;

    LocalTensor<T> rRcLocalTensor_;
};
#endif // _APPLY_CAME_PART2_H_
