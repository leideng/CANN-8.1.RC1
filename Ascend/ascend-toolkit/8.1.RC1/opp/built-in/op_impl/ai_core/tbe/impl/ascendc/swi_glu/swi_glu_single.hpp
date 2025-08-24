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
 * \file swi_glu_single.hpp
 * \brief
 */
#ifndef OPP_SWI_GLU_SINGLE_HPP
#define OPP_SWI_GLU_SINGLE_HPP
#include "glu_tiling_kernel.hpp"

using namespace AscendC;
template<typename ParentClass, typename inType, typename outType>
class SwigluSingle : public ParentClass {
public:
    __aicore__ inline SwigluSingle() = default;
    __aicore__ inline ~SwigluSingle() = default;

    __aicore__ inline void Init(GM_ADDR input_gm, GM_ADDR beta_gm, GM_ADDR output_gm, GM_ADDR tiling_gm)
    {
        singleTiling.GetTilingAndOffset(tiling_gm, sizeof(inType));
        InitGmBuffer(input_gm, beta_gm, output_gm);
        this->InitUbBuffer(singleTiling.tileLength);
    }

    __aicore__ inline void Process()
    {
        if (singleTiling.is32BAligned == 1) {
            SWIGLU_SINGLE_PROCESS(singleTiling);
        } else {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            SWIGLU_SINGLE_PROCESS_NON32BALIGNED(singleTiling);
#endif
        }
    }

protected:
    __aicore__ inline void InitGmBuffer(GM_ADDR input_gm, GM_ADDR beta_gm, GM_ADDR output_gm)
    {
        this->beta = -1.0f /* * (((__gm__ float*)beta_gm)[0]) */;
        // get start index for current core, core parallel
        this->aGm.SetGlobalBuffer((__gm__ inType*)input_gm, singleTiling.totalBlockLen);
        this->cGm.SetGlobalBuffer((__gm__ outType*)output_gm, singleTiling.totalBlockLen / 2);
    }

    __aicore__ inline void CopyIn(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyinParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                            SwiGluCopyParam.splitVecCopyParam.blockLen,
                                            SwiGluCopyParam.splitVecCopyParam.stride,
                                            0};
        // Copy A
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopy(aLocal, this->aGm[offsetParam.splitVecGmOffset1], splitCopyinParams);
        this->inQueueA.EnQue(aLocal);
        // Copy B
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopy(bLocal, this->aGm[offsetParam.splitVecGmOffset2], splitCopyinParams);
        this->inQueueB.EnQue(bLocal);
    }

    __aicore__ inline void CopyOut(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams indepCopyinParams = {SwiGluCopyParam.indepVecCopyParam.blockCount,
                                            SwiGluCopyParam.indepVecCopyParam.blockLen,
                                            0,
                                            SwiGluCopyParam.indepVecCopyParam.stride};
        // deque output tensor from VECOUT queue
        LocalTensor<outType> cLocal = this->outQueueC.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(this->cGm[offsetParam.indepVecGmoffset], cLocal, indepCopyinParams);
        // free output tensor for reuse
        this->outQueueC.FreeTensor(cLocal);
    }

    __aicore__ inline void CopyIn_Non32BAligned(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyinParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                            SwiGluCopyParam.splitVecCopyParam.blockLen,
                                            SwiGluCopyParam.splitVecCopyParam.stride,
                                            0};
        DataCopyPadParams copyPadParams = {false, 0, 0, 0};
        // Copy A
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopyPad(aLocal, this->aGm[offsetParam.splitVecGmOffset1], splitCopyinParams, copyPadParams);
        this->inQueueA.EnQue(aLocal);
        // Copy B
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopyPad(bLocal, this->aGm[offsetParam.splitVecGmOffset2], splitCopyinParams, copyPadParams);
        this->inQueueB.EnQue(bLocal);
    }

    __aicore__ inline void CopyOut_Non32BAligned(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams indepCopyinParams = {SwiGluCopyParam.indepVecCopyParam.blockCount,
                                            SwiGluCopyParam.indepVecCopyParam.blockLen,
                                            0,
                                            SwiGluCopyParam.indepVecCopyParam.stride};
        // deque output tensor from VECOUT queue
        LocalTensor<outType> cLocal = this->outQueueC.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopyPad(this->cGm[offsetParam.indepVecGmoffset], cLocal, indepCopyinParams);
        // free output tensor for reuse
        this->outQueueC.FreeTensor(cLocal);
    }

private:
    SwigluSingleTilingKernel singleTiling;
};
#endif // OPP_SWI_GLU_SINGLE_HPP
