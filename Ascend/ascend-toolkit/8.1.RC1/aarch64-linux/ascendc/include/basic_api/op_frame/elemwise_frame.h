/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file elemwise_frame.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_FRAME_H
#define ASCENDC_KERNEL_FRAME_H
#include "kernel_operator.h"

namespace AscendC {
#define QUE_MAX_DEPTH 2
#ifndef STAGE_NUM
#define STAGE_NUM 3
#endif

class ElemwiseOpBase {
public:
    __aicore__ ElemwiseOpBase() {}

    __aicore__ inline void Init(int32_t loop, int32_t vecInLen, int32_t lcmLen, int32_t vecOutLen)
    {
        SetStagePieces(loop, 0);
        const uint8_t pingpongNum = 2;
        pipe.InitBuffer(inQueue, pingpongNum, vecInLen);
        pipe.InitBuffer(outQueue, pingpongNum, vecOutLen);
        pipe.InitBuffer(tbuf, lcmLen);
    }

    __aicore__ inline void SetStagePieces(const int32_t loopValue, const int32_t progressValue)
    {
        for (int32_t i = 0; i < STAGE_NUM; ++i) {
            stagePieceNum[i] = loopValue;
            stageProgress[i] = progressValue;
        }
    }

protected:
    int32_t stagePieceNum[STAGE_NUM];
    int32_t stageProgress[STAGE_NUM];
    void* tiling;
    TPipe pipe;
    TBuf<TPosition::LCM> tbuf;
    TQue<TPosition::VECIN, QUE_MAX_DEPTH> inQueue;
    TQue<TPosition::VECOUT, QUE_MAX_DEPTH> outQueue;

    enum DataLenType : uint8_t {
        DEFAULT = 0,
    };
};

template <class Op> class ElemwiseFrame : public Op {
public:
    __aicore__ ElemwiseFrame() {}
    __aicore__ inline bool CopyIn(int32_t progress);
    __aicore__ inline bool Compute(int32_t progress);
    __aicore__ inline bool CopyOut(int32_t progress);
    __aicore__ inline bool RunStagePieces(int32_t curStage, int32_t progress);
    __aicore__ inline void Process();
};
template <class Op> __aicore__ inline void ElemwiseFrame<Op>::Process()
{
    int32_t done = 0;
    while (done < STAGE_NUM) {
        done = 0;
        for (int32_t i = 0; i < STAGE_NUM; i++) {
            if (Op::stageProgress[i] >= Op::stagePieceNum[i]) {
                done++;
                continue;
            }
            if (RunStagePieces(i, Op::stageProgress[i])) {
                Op::stageProgress[i]++;
            }
        }
    }
};

template <class Op> __aicore__ inline bool ElemwiseFrame<Op>::RunStagePieces(int32_t curStage, int32_t progress)
{
    switch (curStage) {
        case 0:
            return CopyIn(progress);
        case 1:
            return Compute(progress);
        case 2:
            return CopyOut(progress);
        default:
            ASSERT(0);
            break;
    }
    return true;
}

template <class Op> __aicore__ inline bool ElemwiseFrame<Op>::CopyIn(int32_t progress)
{
    if (!Op::inQueue.VacantInQue()) {
        return false;
    }
    auto xBuf = Op::inQueue.template AllocTensor<typename Op::DType>();
    Op::MyCopyIn(progress, xBuf);
    Op::inQueue.EnQue(xBuf);
    return true;
}

template <class Op> __aicore__ inline bool ElemwiseFrame<Op>::Compute(int32_t progress)
{
    if (!Op::outQueue.VacantInQue()) {
        return false;
    }
    if (Op::inQueue.GetTensorCountInQue() == 0) {
        return false;
    }

    auto xBuf = Op::inQueue.template DeQue<typename Op::DType>();
    auto yBuf = Op::outQueue.template AllocTensor<typename Op::DType>();
    Op::MyCompute(progress, xBuf, yBuf);

    Op::outQueue.template EnQue<typename Op::DType>(yBuf);
    Op::inQueue.template FreeTensor<typename Op::DType>(xBuf);
    return true;
}

template <class Op> __aicore__ inline bool ElemwiseFrame<Op>::CopyOut(int32_t progress)
{
    if (Op::outQueue.GetTensorCountInQue() == 0) {
        return false;
    }

    auto yBuf = Op::outQueue.template DeQue<typename Op::DType>();
    Op::MyCopyOut(progress, yBuf);

    Op::outQueue.FreeTensor(yBuf);
    return true;
}
} // namespace AscendC

#endif // ASCENDC_KERNEL_FRAME_H