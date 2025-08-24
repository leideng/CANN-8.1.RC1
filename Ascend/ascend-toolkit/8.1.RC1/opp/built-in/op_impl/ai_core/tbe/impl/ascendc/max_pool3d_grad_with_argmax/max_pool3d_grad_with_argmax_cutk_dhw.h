/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

/* !
 * \file max_pool3d_grad_with_argmax_cutk_dhw.h
 * \brief
 */
#ifndef MAX_POOL_GRAD3D_WITH_ARGMAX_CUTK_KDHW
#define MAX_POOL_GRAD3D_WITH_ARGMAX_CUTK_KDHW

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "max_pool3d_grad_with_argmax_cutk_base.h"

namespace MaxPool3DGradWithArgmax {
using namespace AscendC;
using namespace MaxPool3DGradWithArgmaxComm;

template <typename TX, typename TGrad, typename TArgmax, typename TY, bool isOverlap>
class MaxPool3DGradWithArgmaxCutKDHW : public MaxPool3DGradWithArgmaxCutKBase<TX, TGrad, TArgmax, TY, isOverlap> {
public:
__aicore__ inline MaxPool3DGradWithArgmaxCutKDHW(TPipe *tmpPipe) {
    this->pipe = tmpPipe;
}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y,
                              GM_ADDR usrWorkspace, const MaxPool3DGradWithArgmaxTilingData* tiling)
{
    // load tiling data
    this->InitParams(tiling);
    InitUbParamsKDHW();
    // set global buffer
    this->InitInputsOutputs(x, grad, argmax, y, usrWorkspace);
    // ub buffer init
    this->InitUbBuffer();
}

__aicore__ inline void InitUbParamsKDHW()
{
    this->baseDp = this->params_.baseDo;
    this->baseHp = this->params_.baseHo;
    this->baseWp = this->params_.baseWo;
    if constexpr (!isOverlap && (std::is_same<TY, half>::value || std::is_same<TY, bfloat16_t>::value)) {
        this->baseWp8Align = this->baseWp * BLOCK_NUM_16;
    } else {
        this->baseWp8Align = this->baseWp * BLOCK_NUM_32;
    }
    this->baseNDHWpAlign = this->params_.baseNc * this->baseDp * this->baseHp * this->baseWp8Align;
}

__aicore__ inline void Process()
{
    uint64_t totalCnt = this->params_.ncCnt * this->params_.doCnt * this->params_.hoCnt * this->params_.woCnt;
    for (uint64_t index = 0; index < totalCnt; index++) {
        if (GetBlockIdx() == index % GetBlockNum()) {
            this->block_.ncCntIndex = index / (this->params_.doCnt * this->params_.hoCnt * this->params_.woCnt);
            this->CalBlockParams(index);
            if constexpr (isOverlap && (std::is_same<TGrad, half>::value || std::is_same<TGrad, bfloat16_t>::value)) {
                CalcBlockKDHW<float>(this->workspaceGm);
            } else {
                CalcBlockKDHW<TY>(this->yGm);
            }
        }
    }
    this->CastAndCarryOut();
}

__aicore__ inline void ProcessCutNc()
{
    uint64_t totalCnt = this->params_.doCnt * this->params_.hoCnt * this->params_.woCnt;
    uint64_t ncLoop = this->params_.ncCnt / GetBlockNum();
    uint64_t curLoop = GetBlockIdx() < (this->params_.ncCnt % GetBlockNum()) ? (ncLoop + 1) : ncLoop;
    for (uint64_t ncLoopIdx = 0; ncLoopIdx < curLoop; ncLoopIdx++) {
        this->block_.ncCntIndex = GetBlockIdx() < this->params_.ncCnt % GetBlockNum() ?
                                  ((ncLoop + 1) * GetBlockIdx() + ncLoopIdx) :
                                  (ncLoop * GetBlockIdx() + ncLoopIdx + this->params_.ncCnt % GetBlockNum());
        for (uint64_t index = 0; index < totalCnt; index++) {
            this->CalBlockParams(index);
            if constexpr (isOverlap && (std::is_same<TGrad, half>::value || std::is_same<TGrad, bfloat16_t>::value)) {
                CalcBlockKDHW<float>(this->workspaceGm);
            } else {
                CalcBlockKDHW<TY>(this->yGm);
            }
        }
    }
    this->CastAndCarryOut();
}

template<typename T>
__aicore__ inline void DepadCopyKDHW(const LocalTensor<T>& yTranUb, const LocalTensor<T>& yTranDepadUb)
{
    for (uint64_t doIdx = this->depad_.padDStartOffset; doIdx <= this->depad_.padDEndOffset; doIdx++) {
        for (uint64_t hoIdx = this->depad_.padHStartOffset; hoIdx <= this->depad_.padHEndOffset; hoIdx++) {
            for (uint64_t woIdx = this->depad_.padWStartOffset; woIdx <= this->depad_.padWEndOffset; woIdx++) {
                uint64_t srcOffset =
                    doIdx * this->block_.hoShape * this->block_.wiShape * this->params_.baseNc +
                    hoIdx * this->block_.wiShape * this->params_.baseNc +
                    woIdx * this->params_.baseNc;
                uint64_t dstOffset = 
                    (doIdx - this->depad_.padDStartOffset) * this->depad_.hiValid *
                    this->depad_.wiValidAlign * this->params_.baseNc +
                    (hoIdx - this->depad_.padHStartOffset) * this->depad_.wiValidAlign * this->params_.baseNc +
                    (woIdx - this->depad_.padWStartOffset) * BLOCK_SIZE / sizeof(T) * this->params_.baseNc;
                Adds(yTranDepadUb[dstOffset], yTranUb[srcOffset], static_cast<T>(0), VL_FP32, this->depad_.wiValid, 
                     {1, 1, VL_FP32 * sizeof(T) / BLOCK_SIZE, VL_FP32 * sizeof(T) / BLOCK_SIZE});
            }
        }
    }
    pipe_barrier(PIPE_V);
}

template<typename T>
__aicore__ inline void CopyOutKDHW(const LocalTensor<T>& yUb, const GlobalTensor<T>& dstGm)
{
    for (uint64_t ncLoop = 0; ncLoop < this->block_.ncShape; ncLoop++) {
        for (uint64_t dIdx = 0; dIdx < this->depad_.diValid; dIdx++) {
            for  (uint64_t hIdx = 0; hIdx < this->depad_.hiValid; hIdx++) {
                DataCopyExtParams copyParams {static_cast<uint16_t>(this->depad_.wiValid),
                                              static_cast<uint32_t>(1 * sizeof(T)),
                                              0,
                                              static_cast<uint32_t>((this->params_.sw - 1) * sizeof(T)),
                                              0};
                uint64_t srcOffset = 
                    ncLoop * this->depad_.diValid * this->depad_.hiValid * this->depad_.wiValidAlign +
                    dIdx * this->depad_.hiValid * this->depad_.wiValidAlign +
                    hIdx * this->depad_.wiValidAlign;
                uint64_t dstOffset = 
                    (ncLoop + this->block_.ncCntIndex * this->params_.baseNc) *
                    this->params_.diDim * this->params_.hiDim * this->params_.wiDim +
                    (this->depad_.diStartOffset + dIdx * this->params_.sd) *
                    this->params_.hiDim * this->params_.wiDim +
                    (this->depad_.hiStartOffset + hIdx * this->params_.sh) * this->params_.wiDim +
                    this->depad_.wiStartOffset;
                DataCopyPad(dstGm[dstOffset], yUb[srcOffset], copyParams);
            }
        }
    }
}

template<typename T>
__aicore__ inline void CalcBlockKDHW(const GlobalTensor<T>& dstGm)
{
    LocalTensor<TArgmax> argmaxTranUb = this->argmaxTransposeBuf.template Get<TArgmax>();
    LocalTensor<TGrad> gradTranUb = this->gradTransposeBuf.template Get<TGrad>();
    this->CopyInData(argmaxTranUb, gradTranUb);
    pipe_barrier(PIPE_V);
    for (uint64_t kdIdx = 0; kdIdx < this->params_.kd; kdIdx++) {
        for (uint64_t khIdx = 0; khIdx < this->params_.kh; khIdx++) {
            for (uint64_t kwIdx = 0; kwIdx < this->params_.kw; kwIdx++) {
                CalcGradKDHW<T>(dstGm, argmaxTranUb, gradTranUb, kdIdx, khIdx, kwIdx);
            }
        }
    }
}

__aicore__ inline void Img2ColPartKDHW(const LocalTensor<TArgmax>& indexColUb,
                                       const LocalTensor<TArgmax>& indexImgUb)
{   
    uint64_t dstOffset = 0;
    uint64_t srcOffset = 0;
    if (this->params_.sw * VL_FP32_BLOCK <= MAX_REP_NUM) {
        for (uint64_t doIdx = 0; doIdx < this->block_.doShape; doIdx++) {
            for (uint64_t hoIdx = 0; hoIdx < this->block_.hoShape; hoIdx++) {
                dstOffset = doIdx * this->block_.hoShape * this->block_.woShape * this->params_.baseNc +
                            hoIdx * this->block_.woShape * this->params_.baseNc;
                srcOffset = doIdx * this->block_.hoShape * this->block_.woShape * this->params_.baseNc + 
                            hoIdx * this->block_.woShape * this->params_.baseNc;
                Adds(indexColUb[dstOffset], indexImgUb[srcOffset], 0, VL_FP32, this->block_.woShape,
                    {1, 1, VL_FP32_BLOCK, VL_FP32_BLOCK});
            }
        }
    } else {
        for (uint64_t doIdx = 0; doIdx < this->block_.doShape; doIdx++) {
            for (uint64_t hoIdx = 0; hoIdx < this->block_.hoShape; hoIdx++) {
                for (uint64_t woIdx = 0; woIdx < this->block_.woShape; woIdx++) {
                    dstOffset = doIdx * this->block_.hoShape * this->block_.woShape * this->params_.baseNc +
                                hoIdx * this->block_.woShape * this->params_.baseNc+
                                woIdx * this->params_.baseNc;
                    srcOffset = doIdx * this->block_.hoShape * this->block_.woShape * this->params_.baseNc + 
                                hoIdx * this->block_.woShape * this->params_.baseNc+
                                woIdx * this->params_.baseNc;
                    Adds(indexColUb[dstOffset], indexImgUb[srcOffset], 0, this->params_.baseNc);
                }
            }
        }
    }
    pipe_barrier(PIPE_V);
}

__aicore__ inline void Col2ImgPartKDHW(const LocalTensor<float>& yTranUb,
                                       const LocalTensor<float>& tmpGradUb)
{
    uint64_t dstOffset = 0;
    uint64_t srcOffset = 0;
    BinaryRepeatParams addRepeatParams = {1, 1, 1, static_cast<uint8_t>(this->params_.sw * VL_FP32_BLOCK),
        static_cast<uint8_t>(VL_FP32_BLOCK), static_cast<uint8_t>(this->params_.sw * VL_FP32_BLOCK)};
    if (this->params_.sw * VL_FP32_BLOCK <= MAX_REP_NUM) {
        for (uint64_t doIdx = 0; doIdx < this->block_.doShape; doIdx++) {
            for (uint64_t hoIdx = 0; hoIdx < this->block_.hoShape; hoIdx++) {
                srcOffset = doIdx * this->block_.hoShape * this->block_.woShape * this->params_.baseNc +
                            hoIdx * this->block_.woShape * this->params_.baseNc;
                dstOffset = doIdx * this->block_.hoShape * this->block_.woShape * this->params_.baseNc + 
                            hoIdx * this->block_.woShape * this->params_.baseNc;
                Add(yTranUb[dstOffset], tmpGradUb[srcOffset], yTranUb[dstOffset], VL_FP32,
                    static_cast<uint8_t>(this->block_.woShape), addRepeatParams);
            }
        }
    } else {
        for (uint64_t doIdx = 0; doIdx < this->block_.doShape; doIdx++) {
            for (uint64_t hoIdx = 0; hoIdx < this->block_.hoShape; hoIdx++) {
                for (uint64_t woIdx = 0; woIdx < this->block_.woShape; woIdx++) {
                    srcOffset = doIdx * this->block_.hoShape * this->block_.woShape * this->params_.baseNc +
                                hoIdx * this->block_.woShape * this->params_.baseNc+
                                woIdx * this->params_.baseNc;
                    dstOffset = doIdx * this->block_.hoShape * this->block_.woShape * this->params_.baseNc + 
                                hoIdx * this->block_.woShape * this->params_.baseNc+
                                woIdx * this->params_.baseNc;
                    Add(yTranUb[dstOffset], tmpGradUb[srcOffset], yTranUb[dstOffset], this->params_.baseNc); 
                }
            }
        }
    }
    pipe_barrier(PIPE_V);
}

__aicore__ inline void CalcGradSubProcessKDHW(const LocalTensor<TArgmax>& indexImgUb,
                                              const LocalTensor<TArgmax>& argmaxTranUb,
                                              const LocalTensor<TGrad>& gradTranUb,
                                              const LocalTensor<float>& yTranUb)
{
    LocalTensor<TArgmax> indexColUb = this->indexColBuf.template Get<TArgmax>();
    LocalTensor<float> tmpGradUb = this->tmpGradBuf.template Get<float>();
    Img2ColPartKDHW(indexColUb, indexImgUb);
    this->SelectGradOut(gradTranUb, argmaxTranUb, indexColUb, tmpGradUb);
    Col2ImgPartKDHW(yTranUb, tmpGradUb);
}

template<typename T>
__aicore__ inline void  CalcGradKDHW(const GlobalTensor<T>& dstGm,
                                     const LocalTensor<TArgmax>& argmaxTranUb,
                                     const LocalTensor<TGrad>& gradTranUb,
                                     uint64_t kdIdx,
                                     uint64_t khIdx,
                                     uint64_t kwIdx)
{
    LocalTensor<TArgmax> indexImgUb = this->indexImgBuf.template Get<TArgmax>();
    LocalTensor<float> yUb = this->yQue.template AllocTensor<float>();
    LocalTensor<float> yTranUb = this->yTransposeBuf.template Get<float>();
    LocalTensor<float> yTranDepadUb = this->yTranDepadBuf.template Get<float>();
    this->CalcDepadParamsD(kdIdx);
    this->CalcDepadParamsH(khIdx);
    this->CalcDepadParamsW(kwIdx);
    bool dataValid = (this->depad_.wiValid > 0) && (this->depad_.hiValid > 0) && (this->depad_.diValid > 0);
    if (dataValid) {
        Duplicate(yTranUb, 0.0f, this->baseNDHWpAlign);
        GenIndicesParams genIndicesParams;
        genIndicesParams.dCount = this->block_.doShape;
        genIndicesParams.colCount = this->block_.woShape;
        genIndicesParams.rowCount = this->block_.hoShape;
        genIndicesParams.firstValue = 
            (this->block_.doCntIndex * this->params_.baseDo * this->params_.sd + kdIdx - this->params_.padDTop) *
            this->params_.hiDim * this->params_.wiDim +
            (this->block_.hoCntIndex * this->params_.baseHo * this->params_.sh + khIdx - this->params_.padHTop) *
            this->params_.wiDim +
            this->block_.woCntIndex * this->params_.baseWo * this->params_.sw + kwIdx - this->params_.padWTop;
        genIndicesParams.increaseWValue = this->params_.sw;
        genIndicesParams.increaseDValue = this->params_.hiDim * this->params_.wiDim * this->params_.sd;
        genIndicesParams.increaseHValue = this->params_.wiDim * this->params_.sh;
        genIndicesParams.vlValue = VL_FP32;
        
        // Generate all index data, compare and select the correct grad out
        this->GenKernelIndicesWithTranspose(indexImgUb, genIndicesParams);
        CalcGradSubProcessKDHW(indexImgUb, argmaxTranUb, gradTranUb, yTranUb);
        
        // Depad: remove invalid data
        if constexpr (!isOverlap && (std::is_same<TY, bfloat16_t>::value)) {
            Cast(yTranUb.ReinterpretCast<TY>(), yTranUb, RoundMode::CAST_RINT, this->baseNDHWpAlign);
            pipe_barrier(PIPE_V);
            DepadCopyKDHW<half>(yTranUb.ReinterpretCast<half>(), yTranDepadUb.ReinterpretCast<half>());
        } else if constexpr (!isOverlap && std::is_same<TY, half>::value) {
            Cast(yTranUb.ReinterpretCast<TY>(), yTranUb, RoundMode::CAST_NONE, this->baseNDHWpAlign);
            pipe_barrier(PIPE_V);
            DepadCopyKDHW<TY>(yTranUb.ReinterpretCast<TY>(), yTranDepadUb.ReinterpretCast<TY>());
        } else {
            DepadCopyKDHW<float>(yTranUb, yTranDepadUb);
        }
        // Transpose the input back and then move it to GM
        TranBackAndMoveOut<T>(dstGm, yUb, yTranDepadUb);
    }
    this->yQue.FreeTensor(yUb);
}

template<typename T>
__aicore__ inline void TranBackAndMoveOut (const GlobalTensor<T>& dstGm,
                                           const LocalTensor<float> yUb,
                                           const LocalTensor<float> yTranDepadUb) {
    int32_t eventIdVToMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    if constexpr (!isOverlap) {
        if constexpr (std::is_same<TY, float>::value) {
            TransposeBase8M16<float>(yUb, yTranDepadUb,
                this->depad_.diValid * this->depad_.hiValid * this->depad_.wiValidAlign, this->params_.baseNc);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            CopyOutKDHW<TY>(yUb, dstGm);
        } else {
            TransposeBase16M16<TY>(yUb.ReinterpretCast<TY>(), yTranDepadUb.ReinterpretCast<TY>(),
                this->depad_.diValid * this->depad_.hiValid * this->depad_.wiValidAlign, this->params_.baseNc);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            CopyOutKDHW<T>(yUb.ReinterpretCast<T>(), dstGm);
        }
    } else {
        TransposeBase8M16<float>(yUb, yTranDepadUb,
            this->depad_.diValid * this->depad_.hiValid * this->depad_.wiValidAlign, this->params_.baseNc);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        SetAtomicAdd<float>();
        CopyOutKDHW<T>(yUb.ReinterpretCast<T>(), dstGm);
        SetAtomicNone();
    }
    int32_t eventIdMte3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
}
};
}

#endif // MAX_POOL_GRAD3D_WITH_ARGMAX_CUTK_KDHW