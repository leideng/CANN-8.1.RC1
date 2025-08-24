/**
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

/*!
 * \file reflection_pad3d_grad_small.h
 * \brief
 */
#ifndef REFLECTION_PAD3D_GRAD_SMALL_H
#define REFLECTION_PAD3D_GRAD_SMALL_H
#include "reflection_pad3d_grad_init.h"

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::SmallProcess() {
    int64_t gmXOffset = 0; 
    int64_t gmYOffset = 0; 
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    for ( size_t loop = 0; loop < loopNC; loop++ ) {
        for ( size_t i = 0; i < curDepth; i++ ) {
            size_t cur_D = GetCurD(i); 
            bool isAtomicAdd = true;
            //top
            gmXOffset = (loop * curDepth * height * width
                        + i * height * width);
            gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                        + cur_D * outHeight * outWidth);
            CopyInSmall(gmXOffset);
            ComputeSmall();
            CopyOutSmall(gmYOffset, isAtomicAdd);
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
    } 
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::CopyInSmall(const int64_t offset) {
    LocalTensor<T> dstLocal = inQueueX.AllocTensor<T>();
    int64_t alignCalW = CeilAlign(width, perBlockCount);
    DataCopyParams copyParams = {1, 0, 0, 0};
    DataCopyPadParams padParams = {true, 0, 0, 0};
    copyParams.blockCount = height;
    copyParams.blockLen = width * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = ((alignWidth - alignCalW)) *sizeof(T) / 32; //((alignWidth - alignCalW)) *sizeof(T) / 32;
    padParams.isPad = true;
    padParams.rightPadding =  alignCalW - width;
    padParams.paddingValue = GetScalarBitcodeValue((T)0);
    DataCopyPad(dstLocal, xGm[offset], copyParams, padParams);
    inQueueX.EnQue(dstLocal);
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::ComputeSmall() {
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    if constexpr (std::is_same<T, bfloat16_t>::value){
        LocalTensor<float> tLocal = transposeBuf.Get<float>();
        LocalTensor<float> float32Tensor = float32Buf.Get<float>();
        int32_t totalData = alignHeight * alignWidth; 
        Cast(float32Tensor, xLocal, RoundMode::CAST_NONE, totalData);
        ComputeSmallBasic<float>(tLocal, float32Tensor);
        TransoseSmall<float>(float32Tensor, tLocal, alignWidth, alignHeight);
        Cast(yLocal, float32Tensor, RoundMode::CAST_RINT, totalData);
    } else {
        LocalTensor<T> tLocal = transposeBuf.Get<T>();
        ComputeSmallBasic<T>(tLocal, xLocal);
        TransoseSmall<T>(yLocal, tLocal, alignWidth, alignHeight);
    }
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::CopyOutSmall(const int64_t offset, const bool isAtomicAdd) {
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyParams copyParams = {1, 0, 0, 0};
    copyParams.blockCount = outHeight ;
    copyParams.blockLen = outWidth * sizeof(T);
    copyParams.srcStride = (alignWidth - outWidth) * sizeof(T) / 32;   //srcStride 为啥不用设置？ ub里面存储的是 alignCalW = 208， calW = 200， （208 - 200）* sizeof(half) / 32 = 0
    copyParams.dstStride = 0;
    int64_t localOffset = hPad1 * alignWidth;
    if (isAtomicAdd == true) {
        SetAtomicAdd<T>();
        DataCopyPad(yGm[offset], yLocal[localOffset], copyParams);
        SetAtomicNone();
    }else {
        DataCopyPad(yGm[offset], yLocal[localOffset], copyParams);
    }  
    outQueueY.FreeTensor(yLocal);
}

template<typename T> template<typename T1>
__aicore__ inline void ReflectionPad3dGrad<T>::TransoseSmall(LocalTensor<T1>& dstLocal, LocalTensor<T1>& srcLocal, const int32_t calH, const int32_t calW){
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = calH / 16;
    transDataParams.dstRepStride = (16 * sizeof(T1)) / 32;
    transDataParams.srcRepStride = (16 * calW * sizeof(T1)) / 32; 
    if (transDataParams.repeatTimes == 1) {
        transDataParams.dstRepStride = 0;
        transDataParams.srcRepStride = 0;
    }
    // 入参类型是LocalTensor地址值的调用方式，推荐使用
    uint64_t srcLocalList[16];
    uint64_t dstLocalList[16];
    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;
    if constexpr (std::is_same<T1, float>::value) {
        for (int i = 0; i < calW / 8; i ++) {
            for (int j = 0; j < 16; j ++) {
                srcLocalList[j] = (uint64_t)(srcLocal[srcOffset + calW * j].GetPhyAddr());
            }
            for (int j = 0; j < 8; j++ ) {
                dstLocalList[2 * j] = (uint64_t)(dstLocal[dstOffset + calH * j].GetPhyAddr());
                dstLocalList[2 * j + 1] = (uint64_t)(dstLocal[dstOffset + calH * j + 8].GetPhyAddr());
            }
            TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
            srcOffset += 8;
            dstOffset += 8 * calH;
        }
    } else {
        for (int i = 0; i < calW / 16; i ++) {
            for (int j = 0; j < 16; j ++) {
                srcLocalList[j] = (uint64_t)(srcLocal[srcOffset + calW * j].GetPhyAddr());
            }
            for (int j = 0; j < 16; j++ ) {
                dstLocalList[j] = (uint64_t)(dstLocal[dstOffset + calH * j].GetPhyAddr());
            }
            TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
            srcOffset += 16;
            dstOffset += 16 * calH;
        }
    }
}

template<typename T> template<typename T1>
__aicore__ inline void ReflectionPad3dGrad<T>::ComputeSmallBasic(LocalTensor<T1>& tLocal, LocalTensor<T1>& xLocal) {
    if (hPad1 > 0) {
        for(uint32_t i = 0; i < hPad1; i++) {
            auto srcLocal_1 = xLocal[i * alignWidth];
            auto srcLocal_2 = xLocal[(2 * hPad1 - i)  * alignWidth];
            Add(srcLocal_2, srcLocal_2, srcLocal_1, alignWidth);
        }
    }

    if (hPad2 > 0) {
        for(uint32_t i = 0; i < hPad2; i++) {
            auto srcLocal_1 = xLocal[(height - 1 - i) * alignWidth];
            auto srcLocal_2 = xLocal[(height - 2*hPad2 - 1 + i)  * alignWidth];
            Add(srcLocal_2, srcLocal_2, srcLocal_1, alignWidth);
        }
    }
    TransoseSmall<T1>(tLocal, xLocal, alignHeight, alignWidth);
    if (wPad1 > 0) {
        for(uint32_t i = 0; i < wPad1; i++) {
            auto srcLocal_1 = tLocal[i * alignHeight];
            auto srcLocal_2 = tLocal[(2 * wPad1 - i) * alignHeight];
            Add(srcLocal_2, srcLocal_2, srcLocal_1, alignHeight); 
        } 
    }

    if (wPad2 > 0) {
        for(uint32_t i = 0; i < wPad2; i++) {
            auto srcLocal_1 = tLocal[(width - 1 - i) * alignHeight];
            auto srcLocal_2 = tLocal[(width - 2 * wPad2 - 1 + i) * alignHeight];
            Add(srcLocal_2, srcLocal_2, srcLocal_1, alignHeight); 
        } 
    }

    // 平移
    if (wPad1 > 0) {
        for (uint32_t i = 0; i < width - wPad1; i++){
            auto srcLocal_1 = tLocal[i * alignHeight];
            auto srcLocal_2 = tLocal[(i + wPad1) * alignHeight];
            Muls(srcLocal_1, srcLocal_2, (T1)1.0, alignHeight);
        }
    }
} 

#endif  // REFLECTION_PAD3D_GRAD_SMALL_H