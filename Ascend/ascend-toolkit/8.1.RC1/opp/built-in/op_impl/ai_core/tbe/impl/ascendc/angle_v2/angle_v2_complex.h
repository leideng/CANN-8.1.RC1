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
 * \file angle_v2_complex.h
 * \brief
 */
#ifndef _ANGLE_V2_COMPLEX_H_
#define _ANGLE_V2_COMPLEX_H_

#include "angle_v2_base.h"

namespace AngleV2N {
template <typename yType>
class AngleV2Complex : public AngleV2Base<yType> {
 public:
  __aicore__ inline AngleV2Complex() {
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AngleV2TilingData* __restrict tilingData) {
    this->BaseMemberDataInit(tilingData);
    repeatTimes = (this->tileLength + this->mask - 1) / this->mask;

    uint32_t totalLengthAlignedCp64 = (this->totalLength + alignNumCp64 - 1) / alignNumCp64 * alignNumCp64;
    uint32_t diffLength = this->totalLengthAligned - totalLengthAlignedCp64;

    uint32_t current_offset = this->offset;
    complexTailLength = this->lastTileLength * COEFFICENT;
    if (GetBlockIdx() == this->formerNum + this->tailNum - 1) {
      if (this->offset < diffLength) {
        // back the compute elements when the address can not back
        complexTailLength =  this->lastTileLength * COEFFICENT - diffLength * COEFFICENT;
      } else {
        // address back to avoid data stampede
        current_offset = this->offset - diffLength;
      }
    }

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(x) + current_offset * COEFFICENT,
                        this->blockLength * COEFFICENT);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y) + current_offset, this->blockLength);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(yType) * COEFFICENT);
    pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(yType));

    pipe.InitBuffer(maskBuf1, this->tileLength * sizeof(uint8_t));
    pipe.InitBuffer(maskBuf2, this->tileLength * sizeof(uint8_t));
    pipe.InitBuffer(realBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(imagBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(zeroBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(oneBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(tempBuf1, this->tileLength * sizeof(yType));
    pipe.InitBuffer(tempBuf2, this->tileLength * sizeof(yType));
    pipe.InitBuffer(tempBuf3, this->tileLength * sizeof(yType));
    pipe.InitBuffer(tempBuf4, this->tileLength * sizeof(yType));
    pipe.InitBuffer(tempBuf5, this->tileLength * sizeof(yType));
  }

  __aicore__ inline void Process() {
    BufferGet();
    // loop count need to be doubled, due to double buffer
    uint32_t complexLength = this->tileLength * COEFFICENT;
#if (__CCE_AICORE__ >= 200)
    for (int32_t i = 0; i < this->tileNum; i++) {
      CopyInComplex(i * complexLength, complexLength);
      pipe_barrier(PIPE_MTE2);
      Compute(this->mask, repeatTimes, this->tileLength);
      pipe_barrier(PIPE_V);
      CopyOut(i * this->tileLength, this->tileLength);
    }

    if (this->lastTileLength > 0) {
      complexLength = this->lastTileLength * COEFFICENT;
      repeatTimes = (this->lastTileLength + this->mask - 1) / this->mask;
      CopyInComplex(this->blockLength * COEFFICENT - complexLength, complexTailLength);
      pipe_barrier(PIPE_MTE2);
      Compute(this->mask, repeatTimes, this->lastTileLength);
      pipe_barrier(PIPE_V);
      CopyOut(this->blockLength - this->lastTileLength, this->lastTileLength);
    }
#else
    for (int32_t i = 0; i < this->tileNum; i++) {
      CopyInComplex(i * complexLength, complexLength);
      pipe_barrier(PIPE_ALL);
      Compute(this->mask, repeatTimes, this->tileLength);
      pipe_barrier(PIPE_ALL);
      CopyOut(i * this->tileLength, this->tileLength);
    }

    if (this->lastTileLength > 0) {
      complexLength = this->lastTileLength * COEFFICENT;
      repeatTimes = (this->lastTileLength + this->mask - 1) / this->mask;
      CopyInComplex(this->blockLength * COEFFICENT - complexLength, complexTailLength);
      pipe_barrier(PIPE_ALL);
      Compute(this->mask, repeatTimes, this->lastTileLength);
      pipe_barrier(PIPE_ALL);
      CopyOut(this->blockLength - this->lastTileLength, this->lastTileLength);
    }
#endif
  }

 private:
  __aicore__ inline void BufferGet() {
    realLocal = realBuf.Get<yType>();
    imagLocal = imagBuf.Get<yType>();
    zeroTensor = zeroBuf.Get<yType>();
    oneTensor = oneBuf.Get<yType>();
    tempTensor1 = tempBuf1.Get<yType>();
    tempTensor2 = tempBuf2.Get<yType>();
    tempTensor3 = tempBuf3.Get<yType>();
    tempTensor4 = tempBuf4.Get<yType>();
    tempTensor5 = tempBuf5.Get<yType>();
    mask1 = maskBuf1.Get<uint8_t>();
    mask2 = maskBuf2.Get<uint8_t>();

    Duplicate(zeroTensor, static_cast<yType>(0.0), this->tileLength);
    Duplicate(oneTensor, static_cast<yType>(1.0), this->tileLength);
  }

  __aicore__ inline void SplitComplex(LocalTensor<yType> &realLocal, LocalTensor<yType> &imagLocal,
                                      LocalTensor<yType> &input, uint32_t calCount) {
#if (__CCE_AICORE__ >= 200)
    // SplitComplex for 910B
    GatherMaskParams params;
    uint64_t rsvdCnt = 0;
    uint32_t complexMask = calCount * 2;
    params.repeatTimes = 1;
    GatherMask(realLocal, input, GATHER_MASK_MODE_ONE, true, complexMask, params, rsvdCnt);
    GatherMask(imagLocal, input, GATHER_MASK_MODE_TWO, true, complexMask, params, rsvdCnt);
    pipe_barrier(PIPE_V);
#else
    // SplitComplex for 910
    for (int32_t i = 0; i < calCount; i++) {
      realLocal.SetValue(i, input.GetValue(COEFFICENT * i));
      imagLocal.SetValue(i, input.GetValue(COEFFICENT * i + 1));
    }
#endif
  }

  __aicore__ inline void CopyInComplex(int32_t coreOffset, uint32_t coreLength) {
    // alloc tensor from queue memory
    LocalTensor<yType> xLocal = inQueue.AllocTensor<yType>();
    // copy progress_th tile from global tensor to local tensor
    DataCopy(xLocal, xGm[coreOffset], coreLength);
    // enque input tensors to VECIN queue
    inQueue.EnQue(xLocal);
  }

  __aicore__ inline void Compute(uint64_t mask, uint8_t repeatTimes, uint32_t calCount) {
    // deque input tensors from VECIN queue
    LocalTensor<yType> input = inQueue.DeQue<yType>();
    LocalTensor<yType> result = outQueue.AllocTensor<yType>();

    // get the real and imag for input(complex)
    SplitComplex(realLocal, imagLocal, input, calCount);

    // calculate atan(imag / real)
    Div(tempTensor5, imagLocal, realLocal, calCount);
    pipe_barrier(PIPE_V);
#if (__CCE_AICORE__ >= 200)
    // implementation of Atan for 910B, Atan not support 910
    Atan<yType, false>(result, tempTensor5, calCount);
#else
    // implementation of Atan for 910
    AtanCompute(result, tempTensor5, tempTensor1, calCount);
#endif
    pipe_barrier(PIPE_V);

    // fix result by imag
    mask1 = imagLocal >= zeroTensor;
    Duplicate(tempTensor1, static_cast<yType>(1.0), calCount);
    Duplicate(tempTensor2, static_cast<yType>(-1.0), calCount);
    pipe_barrier(PIPE_V);
    // tempTensor1 = when imag >= 0 then 1 else -1
    this->DoSelect(tempTensor1, mask1, tempTensor1, tempTensor2, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    mask1 = realLocal < zeroTensor;
    // tempTensor2 = when (real < 0 and imag >=0) then 1; when (real < 0 and imag < 0) then -1; else 0
    this->DoSelect(tempTensor2, mask1, tempTensor1, zeroTensor, mask, repeatTimes);
    Duplicate(tempTensor3, static_cast<yType>(constData.const_pi_by_two), calCount);
    pipe_barrier(PIPE_V);
    // tempTensor1 = when imag >= 0 then pi / 2 else -pi / 2
    Mul(tempTensor1, tempTensor3, tempTensor1, calCount);
    pipe_barrier(PIPE_V);

    // fix result by real
    Duplicate(tempTensor3, static_cast<yType>(constData.const_pi), calCount);
    pipe_barrier(PIPE_V);
    // tempTensor2 = when (real < 0 and imag >=0) then pi; when (real < 0 and imag < 0) then -pi; else 0 
    Mul(tempTensor2, tempTensor3, tempTensor2, calCount);
    mask1 = realLocal == zeroTensor;
    pipe_barrier(PIPE_V);
    // result = when (x == 0 and imag >= 0) then pi / 2; when (x == 0 and imag < 0) then -pi / 2; else result
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);
    pipe_barrier(PIPE_V);

    // result = when (real < 0 and imag >= 0) then result + pi; when (real < 0 and imag < 0) then result - pi; else result
    Add(result, tempTensor2, result, calCount);
    pipe_barrier(PIPE_V);
    outQueue.EnQue<yType>(result);
    CornerProcess(mask, repeatTimes, calCount);
    // free input tensors for reuse
    inQueue.FreeTensor(input);
  }

  __aicore__ inline void CornerProcess(uint64_t mask, uint8_t repeatTimes, uint32_t calCount) {
    LocalTensor<yType> result = outQueue.DeQue<yType>();
  
    mask1 = imagLocal == zeroTensor;
    pipe_barrier(PIPE_V);
    this->DoSelect(tempTensor1, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    mask1 = imagLocal < zeroTensor;
    pipe_barrier(PIPE_V);
    // tempTensor2(signbit_imag) = when imag < 0 then 1 else 0
    this->DoSelect(tempTensor2, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    Mul(tempTensor2, tempTensor1, tempTensor2, calCount);
    pipe_barrier(PIPE_V);
    // tempTensor3 = when (imag = 0 and imag >= 0) then 1; when (imag = 0 and imag < 0) then -1; else 0
    Sub(tempTensor3, tempTensor1, tempTensor2, calCount);

    // real < 0 and imag = +0 --> pi
    mask1 = realLocal < zeroTensor;
    pipe_barrier(PIPE_V);
    // tempTensor1(signbit_real) = when real < 0 then 1 else 0
    this->DoSelect(tempTensor1, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    Sub(tempTensor4, oneTensor, tempTensor1, calCount);
    Mul(tempTensor5, tempTensor3, tempTensor1, calCount);
    pipe_barrier(PIPE_V);
    mask1 = tempTensor5 == oneTensor;
    pipe_barrier(PIPE_V);
    Duplicate(tempTensor5, static_cast<yType>(constData.const_pi), calCount);
    pipe_barrier(PIPE_V);
    // result = when (imag == +0 and real < 0) then pi else result
    this->DoSelect(result, mask1, tempTensor5, result, mask, repeatTimes);

    // real >= 0 and imag = +0 -->0
    Mul(tempTensor3, tempTensor4, tempTensor3, calCount);
    pipe_barrier(PIPE_V);
    mask1 = tempTensor3 == oneTensor;
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, zeroTensor, result, mask, repeatTimes);

    // real < 0 and imag = -0 --> -pi
    Mul(tempTensor1, tempTensor2, tempTensor1, calCount);
    pipe_barrier(PIPE_V);
    mask1 = tempTensor1 == oneTensor;
    pipe_barrier(PIPE_V);
    Duplicate(tempTensor1, static_cast<yType>(constData.const_neg_pi), calCount);
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);

    // real >= 0 and imag =-0 --> -0
    Mul(tempTensor4, tempTensor2, tempTensor4, calCount);
    pipe_barrier(PIPE_V);
    mask1 = tempTensor4 == oneTensor;
    Duplicate(tempTensor1, static_cast<yType>(float(-0.0)), calCount);
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);

    CornerProcessINFNAN(result, oneTensor, mask, repeatTimes, calCount);
    // enque the output tensor to VECOUT queue
    outQueue.EnQue<yType>(result);
  }

  __aicore__ inline void CornerProcessINFNAN(LocalTensor<yType> &result, LocalTensor<yType> &oneTensor, uint64_t mask,
                                             uint8_t repeatTimes, uint32_t calCount) {
    CornerProcessRealINF(result, oneTensor, mask, repeatTimes, calCount);
    CornerProcessRealNINF(result, oneTensor, mask, repeatTimes, calCount);
    CornerProcessNAN(result, mask, repeatTimes, calCount);
  }

  __aicore__ inline void CornerProcessRealINF(LocalTensor<yType> &result, LocalTensor<yType> &oneTensor, uint64_t mask,
                                              uint8_t repeatTimes, uint32_t calCount) {
    // real = inf and -inf < imag < 0 --> -0
    mask1 = imagLocal < zeroTensor;
    pipe_barrier(PIPE_V);
    this->DoSelect(tempTensor2, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    Duplicate(tempTensor3, static_cast<yType>(-INFINITY), calCount);
    pipe_barrier(PIPE_V);
    mask1 = imagLocal > tempTensor3;
    pipe_barrier(PIPE_V);
    this->DoSelect(tempTensor2, mask1, tempTensor2, zeroTensor, mask, repeatTimes);
    Duplicate(tempTensor4, static_cast<yType>(INFINITY), calCount);
    pipe_barrier(PIPE_V);
    mask2 = realLocal == tempTensor4;
    pipe_barrier(PIPE_V);
    this->DoSelect(tempTensor2, mask2, tempTensor2, zeroTensor, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    mask1 = tempTensor2 == oneTensor;
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);
    pipe_barrier(PIPE_V);

    // real = inf and imag = inf --> pi / 4
    mask1 = imagLocal == tempTensor4;
    pipe_barrier(PIPE_V);
    this->DoSelect(tempTensor1, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    mask1 = imagLocal == tempTensor3;
    pipe_barrier(PIPE_V);
    this->DoSelect(tempTensor2, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    this->DoSelect(tempTensor5, mask2, tempTensor1, zeroTensor, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    mask1 = tempTensor5 == oneTensor;
    pipe_barrier(PIPE_V);
    Duplicate(tempTensor5, static_cast<yType>(constData.const_pi_by_four), calCount);
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, tempTensor5, result, mask, repeatTimes);
    pipe_barrier(PIPE_V);
  
    // real = inf and imag = -inf --> -pi / 4
    this->DoSelect(tempTensor5, mask2, tempTensor2, zeroTensor, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    mask1 = tempTensor5 == oneTensor;
    pipe_barrier(PIPE_V);
    Duplicate(tempTensor5, static_cast<yType>(constData.const_neg_pi_by_four), calCount);
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, tempTensor5, result, mask, repeatTimes);
  }

  __aicore__ inline void CornerProcessRealNINF(LocalTensor<yType> &result, LocalTensor<yType> &oneTensor, uint64_t mask,
                                               uint8_t repeatTimes, uint32_t calCount) {
    // real = -inf and imag = inf --> 3pi / 4
    mask2 = realLocal == tempTensor3;
    pipe_barrier(PIPE_V);
    this->DoSelect(tempTensor3, mask2, tempTensor1, zeroTensor, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    mask1 = tempTensor3 == oneTensor;
    pipe_barrier(PIPE_V);
    Duplicate(tempTensor3, static_cast<yType>(constData.const_pi_by_three_quarters), calCount);
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, tempTensor3, result, mask, repeatTimes);
    pipe_barrier(PIPE_V);

    // real = -inf and imag = -inf --> -3pi / 4
    this->DoSelect(tempTensor3, mask2, tempTensor2, zeroTensor, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    mask1 = tempTensor3 == oneTensor;
    pipe_barrier(PIPE_V);
    Duplicate(tempTensor3, static_cast<yType>(constData.const_neg_pi_by_three_quarters), calCount);
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, tempTensor3, result, mask, repeatTimes);
    pipe_barrier(PIPE_V);

    // abs_real < inf and imag = inf --> pi / 2
    Abs(tempTensor3, realLocal, calCount);
    pipe_barrier(PIPE_V);
    mask2 = tempTensor3 < tempTensor4;
    pipe_barrier(PIPE_V);
    this->DoSelect(tempTensor1, mask2, tempTensor1, zeroTensor, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    mask1 = tempTensor1 == oneTensor;
    pipe_barrier(PIPE_V);
    Duplicate(tempTensor1, static_cast<yType>(constData.const_pi_by_two), calCount);
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);

    // abs_real < inf and imag = -inf --> -pi / 2
    this->DoSelect(tempTensor2, mask2, tempTensor2, zeroTensor, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    mask1 = tempTensor2 == oneTensor;
    Duplicate(tempTensor1, static_cast<yType>(constData.const_neg_pi_by_two), calCount);
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);
  }

  __aicore__ inline void CornerProcessNAN(LocalTensor<yType> &result, uint64_t mask, uint8_t repeatTimes,
                                          uint32_t calCount) {
    Duplicate(tempTensor1, static_cast<yType>(NAN), calCount);
    mask1 = realLocal == realLocal;
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, result, tempTensor1, mask, repeatTimes);
    pipe_barrier(PIPE_V);
    mask1 = imagLocal == imagLocal;
    pipe_barrier(PIPE_V);
    this->DoSelect(result, mask1, result, tempTensor1, mask, repeatTimes);
  }

  __aicore__ inline void CopyOut(int32_t coreOffset, uint32_t coreLength) {
    // deque output tensor from VECOUT queue
    LocalTensor<yType> result = outQueue.DeQue<yType>();
    // copy progress_th tile from local tensor to global tensor
    DataCopy(yGm[coreOffset], result, coreLength);
    // free output tensor for reuse
    outQueue.FreeTensor(result);
  }

  __aicore__ inline void Sign(LocalTensor<yType> &dst, LocalTensor<yType> &src, LocalTensor<yType> &denominator,
                              uint32_t calCount) {
    Muls(dst, src, static_cast<yType>(ATAN_FP32_MAX), calCount);
    pipe_barrier(PIPE_V);
    Abs(denominator, dst, calCount);
    pipe_barrier(PIPE_V);
    Adds(denominator, denominator, static_cast<yType>(ATAN_FP32_MIN), calCount);
    pipe_barrier(PIPE_V);
    Div(dst, dst, denominator, calCount);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void TaylorExpand(LocalTensor<yType> &dstTensor, LocalTensor<yType> &srcTensor,
                                      LocalTensor<yType> &squareTensor, int32_t expandLevel, uint32_t calCount) {
    // arctan(x) = x - x^3/3 + x^5/5 + ... + (-1)^k*x^(k*2+1)/( k*2+1)
    // The initial value of dstTensor is assigned as the coefficient of the last item of expansion.
    Mul(squareTensor, srcTensor, srcTensor, calCount);
    Mul(dstTensor, srcTensor, srcTensor, calCount);
    pipe_barrier(PIPE_V);
    Muls(dstTensor, dstTensor, factorList[expandLevel], calCount);
    pipe_barrier(PIPE_V);
    for (uint32_t i = expandLevel - 1; i > 0; --i) {
        // dst*x^2+ the previois expand factor
        Adds(dstTensor, dstTensor, factorList[i], calCount);
        pipe_barrier(PIPE_V);
        Mul(dstTensor, dstTensor, squareTensor, calCount);
        pipe_barrier(PIPE_V);
    }
    Adds(dstTensor, dstTensor, factorList[0], calCount);
    pipe_barrier(PIPE_V);
    Mul(dstTensor, dstTensor, srcTensor, calCount);
  }

  __aicore__ inline void AtanTransform(LocalTensor<yType> &dstTensor, LocalTensor<yType> &srcTensor,
                                       LocalTensor<yType> &tmpTensor, const float transFactor, uint32_t calCount) {
    // (x-y)/(1+xy)
    const float transFactorNeg = 0 - transFactor;

    // x*y
    Muls(dstTensor, srcTensor, transFactor, calCount);
    pipe_barrier(PIPE_V);
    // x*y + 1
    Adds(dstTensor, dstTensor, static_cast<yType>(1.0), calCount);
    // x=x-y
    Adds(tmpTensor, srcTensor, transFactorNeg, calCount);
    pipe_barrier(PIPE_V);
    // (x-y)/(1+xy)
    Div(dstTensor, tmpTensor, dstTensor, calCount);
    pipe_barrier(PIPE_V);
    // abs(x)
    Abs(dstTensor, dstTensor, calCount);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void AtanImpl(LocalTensor<yType> &dstTensor, LocalTensor<yType> &srcTensor,
                                  LocalTensor<yType> &tmpTensor, uint32_t calCount) {
    const float piByFour = 0.78539816339744830961566084581988;
    const float piByEight = 0.39269908169872415480783042290994;
    const float tanPiByEight = 0.4142135623730950;
    LocalTensor<yType> absTensor = tempTensor3;
    LocalTensor<yType> tmpTensor2 = tempTensor2;
    LocalTensor<yType> squareTensor = tempTensor4;

    Abs(absTensor, srcTensor, calCount);
    pipe_barrier(PIPE_V);

    // when x's value is too large the first caculator of TaylorExpand will be overflow. when epsilon is 0.0001,
    // the approximate value of `tan(pi/2 - 0.0001)` is 10000
    Mins(absTensor, absTensor, static_cast<yType>(10000), calCount);
    pipe_barrier(PIPE_V);

    // calucate data less than one
    // 1.x
    TaylorExpand(dstTensor, absTensor, squareTensor, TAYLOR_COUNT_FOUR, calCount);

    // 2.(x-tan(pi/8)) / (1 + x*tan(pi/8))
    AtanTransform(tmpTensor, absTensor, tmpTensor2, tanPiByEight, calCount); // tan(pi/8)
    TaylorExpand(tmpTensor2, tmpTensor, squareTensor, TAYLOR_COUNT_FOUR, calCount);
    pipe_barrier(PIPE_V);
    // pi/8 + atan((x-tan(pi/8)) / (1 + x*tan(pi/8)))
    Adds(tmpTensor2, tmpTensor2, piByEight, calCount);
    pipe_barrier(PIPE_V);
    Min(dstTensor, dstTensor, tmpTensor2, calCount);

    // calucate data more than one
    // (x-1)/(x+1)
    Adds(tmpTensor2, absTensor, static_cast<yType>(1.0), calCount);
    // x=x-y
    Adds(tmpTensor, absTensor, -static_cast<yType>(1.0), calCount);
    pipe_barrier(PIPE_V);
    // (x-y)/(1+xy)
    Div(tmpTensor, tmpTensor, tmpTensor2, calCount);
    pipe_barrier(PIPE_V);
    Abs(tmpTensor, tmpTensor, calCount); // take the absolute value
    pipe_barrier(PIPE_V);

    // 3. atan((x-1)/(x+1))
    TaylorExpand(tmpTensor2, tmpTensor, squareTensor, TAYLOR_COUNT_FOUR, calCount);
    pipe_barrier(PIPE_V);
    // pi/4 + atan((x-1)/(x+1))
    Adds(tmpTensor2, tmpTensor2, piByFour, calCount);
    pipe_barrier(PIPE_V);
    Min(dstTensor, dstTensor, tmpTensor2, calCount);

    // 4.reuse the Transform result in step 3, and calculate (x-tan(pi/8)) / (1 + x*tan(pi/8))
    AtanTransform(tmpTensor2, tmpTensor, squareTensor, tanPiByEight, calCount); // tan(pi/8)
    TaylorExpand(tmpTensor, tmpTensor2, squareTensor, TAYLOR_COUNT_SIX, calCount);
    pipe_barrier(PIPE_V);
    // pi/8 + atan((x-tan(pi/8)) / (1 + x*tan(pi/8)))
    Adds(tmpTensor, tmpTensor, piByEight, calCount);
    pipe_barrier(PIPE_V);
    Adds(tmpTensor, tmpTensor, piByFour, calCount);
    pipe_barrier(PIPE_V);
    Min(dstTensor, dstTensor, tmpTensor, calCount);
    pipe_barrier(PIPE_V);
}

  __aicore__ inline void AtanCompute(LocalTensor<yType> &dstTensor, LocalTensor<yType> &srcTensor,
                                     LocalTensor<yType> &tmpTensor, uint32_t calCount) {
    AtanImpl(dstTensor, srcTensor, tmpTensor, calCount);

    // get sign of src
    Sign(tmpTensor, srcTensor, tempTensor2, calCount);

    // dst = sign(x) * dst.
    Mul(dstTensor, dstTensor, tmpTensor, calCount);
    pipe_barrier(PIPE_V);
}

 private:
  TPipe pipe;
  ConstData constData;
  uint8_t repeatTimes;
  uint32_t complexTailLength = 64;

  const float factorList[7] = {1, -0.3333333333333333, 0.2, -0.14285714285714285, 0.1111111111111111,
                               -0.09090909090909091, 0.07692307692307693};
  const float ATAN_FP32_MAX = 4611686018427387904;   // 2^62
  const float ATAN_FP32_MIN = 2.168404344971009e-19; // 2^-62
  const uint8_t TAYLOR_COUNT_FOUR = 4; // x > 0 and x < tan(pi/8)
  const uint8_t TAYLOR_COUNT_SIX = 6; // x > tan(pi/8) and x < tan(pi/4)
  const uint8_t GATHER_MASK_MODE_ONE = 1;
  const uint8_t GATHER_MASK_MODE_TWO = 2;
  const uint32_t alignNumCp64 = 4;

  GlobalTensor<yType> xGm, yGm;

  TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

  TBuf<TPosition::VECCALC> maskBuf1, tempBuf1, tempBuf2, tempBuf3;
  TBuf<TPosition::VECCALC> maskBuf2, tempBuf4, tempBuf5, oneBuf, zeroBuf;
  TBuf<TPosition::VECCALC> realBuf, imagBuf;

  LocalTensor<yType> realLocal, imagLocal, zeroTensor, oneTensor;
  LocalTensor<yType> tempTensor1, tempTensor2, tempTensor3, tempTensor4, tempTensor5;
  LocalTensor<uint8_t> mask1, mask2;
};
} // AngleV2N
#endif  // _ANGLE_V2_COMPLEX_H_
