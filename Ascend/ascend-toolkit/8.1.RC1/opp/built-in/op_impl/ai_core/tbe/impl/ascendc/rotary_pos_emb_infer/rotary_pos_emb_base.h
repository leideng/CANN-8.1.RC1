/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file rotary_pos_emb_base.h
 * \brief
 */

#ifndef ROTARY_POS_EMB_BASE
#define ROTARY_POS_EMB_BASE

#include "simd.h"
#include "common_val.h"
#include "kernel_operator.h"

using AscendC::HardEvent;
using AscendC::Duplicate;

template <typename QkDtype, typename CosDtype, bool IF_COS_BROADCAST>
class RopeBase {
public:
    // QkDtype ：输入qk和输出qk的数据类型
    // CosDtype ：输入cos/sin的数据类型
    // IF_COS_BROADCAST ：cos sin是否已扩展
    // 构造函数
    __aicore__ inline RopeBase(RotaryPosEmbInferTilingData *tilingData, AscendC::TPipe *pipe)
        : pipe_(pipe), blockIdx_(AscendC::GetBlockIdx())
    {
        this->tilingData_ = tilingData;
        batchSize_ = (tilingData_->cosFormat == 0) ? 0 : ((tilingData_->batch + DEFAULT_REPEAT_STRIDE - 1) /
            DEFAULT_REPEAT_STRIDE) * DEFAULT_REPEAT_STRIDE;
        hiddenSize_ = tilingData_->hiddenSizeK > tilingData_->hiddenSizeQ ?
            tilingData_->hiddenSizeK : tilingData_->hiddenSizeQ;
        nlCoreRun_ = (tilingData_->ntokens + tilingData_->realCore - 1) / tilingData_->realCore;
        lCoreRun_ =  tilingData_->ntokens - (tilingData_->realCore - 1) * nlCoreRun_;
        headNum_ = tilingData_->headNumK > tilingData_->headNumQ ? tilingData_->headNumK : tilingData_->headNumQ;
        rotateStride_ = tilingData_->headDim / tilingData_->rotaryCoeff;
        dynamicRound_ = (blockIdx_ == tilingData_->realCore - 1) ? lCoreRun_ : nlCoreRun_;
        rotaryStrideOffset = (tilingData_->headDim == tilingData_->rotaryCoeff) ? 1 : rotateStride_;
        alignRotary_ = rotateStride_ % ELE_NUM_FP16;
        pipe_->InitBuffer(seqLenQueue_, 1, batchSize_ * sizeof(int32_t));
    }

    // 初始化Gm
    __aicore__ inline void RopeInitGm(__gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *cos, __gm__ uint8_t *sin,
        __gm__ uint8_t *seqLen, __gm__ uint8_t *outQ, __gm__ uint8_t *outK)
    {
        qGm_.SetGlobalBuffer((__gm__ QkDtype *)q);
        kGm_.SetGlobalBuffer((__gm__ QkDtype *)k);
        cosGm_.SetGlobalBuffer((__gm__ CosDtype *)cos);
        sinGm_.SetGlobalBuffer((__gm__ CosDtype *)sin);
        outQGm_.SetGlobalBuffer((__gm__ QkDtype *)outQ);
        outKGm_.SetGlobalBuffer((__gm__ QkDtype *)outK);
        seqLenGm_.SetGlobalBuffer((__gm__ int32_t *)seqLen);
    }

    template <typename T>
    __aicore__ inline void Copy2Ub(const AscendC::GlobalTensor<T> &src, const AscendC::LocalTensor<T> &dst,
                                   uint32_t copyLen)
    {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        if (g_coreType == AscendC::AIC) return;
#endif
        uint32_t blkSizeReal = BLK_SIZE / sizeof(T);
        if (copyLen % blkSizeReal != 0) {
            DataCopy(dst, src, {1, static_cast<uint16_t>((copyLen + blkSizeReal - 1) / blkSizeReal), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            DataCopy(dst, src, {1, static_cast<uint16_t>(copyLen / blkSizeReal), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    template <typename T>
    __aicore__ inline void Copy2Gm(const AscendC::LocalTensor<T> &src, const AscendC::GlobalTensor<T> &dst,
                                   uint32_t hiddenSizeLen)
    {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        if (g_coreType == AscendC::AIC) return;
#endif
        uint32_t blkSizeReal = BLK_SIZE / sizeof(T);
        if (hiddenSizeLen % blkSizeReal != 0) {
            DataCopy(dst, src, {1, static_cast<uint16_t>((hiddenSizeLen + blkSizeReal - 1) / blkSizeReal), 0, 0});
        } else {
            DataCopy(dst, src, {1, static_cast<uint16_t>(hiddenSizeLen / blkSizeReal), 0, 0});
        }
    }

    // 此函数用来复用unpad情況下的cos和sin
    // 例：cos[0~7] cos[0~3]用于第一个batch， cos[0~4]用于第二个batch
    __aicore__ inline void ExpandCosSin(const AscendC::LocalTensor<CosDtype> &tempBuf,
                                        const AscendC::GlobalTensor<CosDtype> &src,
                                        const AscendC::GlobalTensor<CosDtype> &extraGm)
    {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        if (g_coreType == AscendC::AIC) return;
#endif
        // cos or sin,[maxseqlen,headsize]-->[sumseqlen,hiddensize]
        AscendC::LocalTensor<int32_t> seqLenLocal = seqLenQueue_.AllocTensor<int32_t>();
        DataCopy(seqLenLocal, seqLenGm_, {1, static_cast<uint16_t>(batchSize_ * sizeof(int32_t) / 32), 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
        uint64_t rowsPerLoop = static_cast<uint64_t>((maxProcessNum_ - batchSize_ * NUM_TWO) / tilingData_->headDim);
        uint64_t cosoffset = 0;
        for (uint32_t perBatch = 0; perBatch < tilingData_->batch; perBatch++) {
            uint64_t rowsRepeat = seqLenLocal.GetValue(perBatch) / rowsPerLoop;
            int32_t rowsRemain = seqLenLocal.GetValue(perBatch) % rowsPerLoop;
            for (uint64_t j = 0; j < rowsRepeat; j++) {
                Copy2Ub(src[(j * rowsPerLoop) * tilingData_->headDim], tempBuf, rowsPerLoop * tilingData_->headDim);
                Copy2Gm(tempBuf, extraGm[(cosoffset + j * rowsPerLoop) * tilingData_->headDim],
                        rowsPerLoop * tilingData_->headDim);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            if (rowsRemain > 0) {
                Copy2Ub(src[(rowsRepeat * rowsPerLoop) * tilingData_->headDim], tempBuf,
                        rowsRemain * tilingData_->headDim);
                Copy2Gm(tempBuf, extraGm[(cosoffset + rowsRepeat * rowsPerLoop) * tilingData_->headDim],
                        rowsRemain * tilingData_->headDim);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            cosoffset += seqLenLocal.GetValue(perBatch);
        }
        seqLenQueue_.FreeTensor(seqLenLocal);
    }

    // 构建tensor -1 -1 -1 0 0 0
    // 构建tensor 0 0 0 1 1 1
    template<typename BUF_TYPE>
    __aicore__ inline void ExpandNeg(const AscendC::LocalTensor<BUF_TYPE> &tempBuf,
        uint32_t bufPos, uint32_t headNumTemp, uint32_t repeatTimeTemp)
    {
        if (tilingData_->headDim != tilingData_->rotaryCoeff) {
            if (alignRotary_ == 0) { // 对齐直接 -1 1
                for (uint32_t i = 0; i < rotateStride_; ++i) {
                    tempBuf.SetValue(negOne_ + i, (BUF_TYPE)-1);
                    tempBuf.SetValue(negOne_ + i + rotateStride_, (BUF_TYPE)1);
                }
                AscendC::SetFlag<HardEvent::S_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID1);
                for (uint32_t i = 1; i < headNumTemp * tilingData_->rotaryCoeff / NUM_TWO; ++i) {
                    DataCopy(tempBuf[negOne_ + rotateStride_ * NUM_TWO * i], tempBuf[negOne_],
                             {1, static_cast<uint16_t>(rotateStride_ * sizeof(BUF_TYPE) / ELE_NUM_FP16), 0, 0});
                }
            } else {
                for (uint32_t i = 0; i < rotateStride_; ++i) { // 非对齐 -1 0
                    tempBuf.SetValue(negOne_ + i, (BUF_TYPE)-1);
                    tempBuf.SetValue(negOne_ + i + rotateStride_, (BUF_TYPE)0);
                }
                AscendC::SetFlag<HardEvent::S_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID1);
                for (uint32_t i = 0; i < headNumTemp * tilingData_->rotaryCoeff / NUM_TWO; ++i) {
                    if ((rotateStride_ * NUM_TWO) *  sizeof(BUF_TYPE) % BLK_SIZE == 0) {
                        DataCopy(tempBuf[negOne_ + rotateStride_ * NUM_TWO * i], tempBuf[negOne_],
                                 {1, static_cast<uint16_t>(rotateStride_ * NUM_TWO * sizeof(BUF_TYPE) / ELE_NUM_FP16),
                                  0, 0});
                    } else {
                        for (uint32_t j = 0; j < rotateStride_ * NUM_TWO; j++) {
                            tempBuf.SetValue(negOne_ + rotateStride_ * NUM_TWO * i + j, tempBuf.GetValue(negOne_ + j));
                        }
                    }
                }
                
                AscendC::SetFlag<HardEvent::S_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID1);
                AscendC::PipeBarrier<PIPE_V>();
                adds_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[bufPos], tempBuf[negOne_], (BUF_TYPE)1, repeatTimeTemp,
                                                        1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
            }
        } else {
            Duplicate<BUF_TYPE, false>(tempBuf[negOne_], (BUF_TYPE)-1.0, AscendC::MASK_PLACEHOLDER,
                                       (uint8_t)repeatTimeTemp, 1, DEFAULT_REPEAT_STRIDE);
            AscendC::SetVectorMask<BUF_TYPE, AscendC::MaskMode::NORMAL>((uint64_t)0xaaaaaaaaaaaaaaaa,
                                                                        (uint64_t)0xaaaaaaaaaaaaaaaa);
            Duplicate<BUF_TYPE, false>(tempBuf[negOne_], (BUF_TYPE)0.0, AscendC::MASK_PLACEHOLDER,
                                       (uint8_t)repeatTimeTemp, 1, DEFAULT_REPEAT_STRIDE);
            AscendC::ResetMask();
            AscendC::PipeBarrier<PIPE_V>();
            adds_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[bufPos], tempBuf[negOne_], (BUF_TYPE)1, repeatTimeTemp, 1,
                                                    1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        }
    }

    // 从(tilingData_->headDim)->(heads*tilingData_->headDim)
    __aicore__ inline void CosSinCommonBroardcast(const AscendC::GlobalTensor<CosDtype> &extraGm,
        uint32_t z, const AscendC::LocalTensor<CosDtype> &tempBuf, uint32_t calcLen)
    {
        // 永远的先拷一次
        uint32_t cosOffset = blockIdx_ * nlCoreRun_ * tilingData_->headDim + z * tilingData_->headDim;
        uint32_t sinOffset = blockIdx_ * nlCoreRun_ * tilingData_->headDim + z * tilingData_->headDim;
        AscendC::SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
        AscendC::DataCopyParams copyParams = {
            1, static_cast<uint16_t>((tilingData_->headDim * sizeof(CosDtype) + BLK_SIZE - 1) / BLK_SIZE), 0, 0};
        DataCopy(tempBuf[cosPad_], cosGm_[cosOffset], copyParams);
        DataCopy(tempBuf[sinPad_], sinGm_[cosOffset], copyParams);
        if (tilingData_->cosFormat == 1) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID3);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
        if ((tilingData_->headDim * sizeof(CosDtype)) % BLK_SIZE != 0) {
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID3);
            // 补齐cos，从(tilingData_->headDim)->(heads*tilingData_->headDim)
            // headnum
            for (uint32_t i = 0; i < calcLen / tilingData_->headDim; ++i) {
                DataCopy(extraGm[offsetExtraGm_ + tilingData_->headDim * i], tempBuf[cosPad_], copyParams);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            Copy2Ub<CosDtype>(extraGm[offsetExtraGm_], tempBuf[cosPad_], calcLen);
            // 补齐sin，从(tilingData_->headDim)->(heads*tilingData_->headDim)
            for (uint32_t i = 0; i < calcLen / tilingData_->headDim; ++i) {
                DataCopy(extraGm[offsetExtraGm_ + tilingData_->headDim * i], tempBuf[sinPad_],
                         {1, static_cast<uint16_t>((tilingData_->headDim * sizeof(CosDtype) + BLK_SIZE - 1) / BLK_SIZE),
                          0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            Copy2Ub<CosDtype>(extraGm[offsetExtraGm_], tempBuf[sinPad_], calcLen);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
        } else {
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
            copyParams = {1, static_cast<uint16_t>(tilingData_->headDim * sizeof(CosDtype) / BLK_SIZE), 0, 0};
            for (uint32_t i = 1; i < calcLen / tilingData_->headDim; ++i) {
                DataCopy(tempBuf[cosPad_ + tilingData_->headDim * i], tempBuf[cosPad_], copyParams);
                DataCopy(tempBuf[sinPad_ + tilingData_->headDim * i], tempBuf[sinPad_], copyParams);
            }
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID3);
        }
    }

    // 满足 cos sin 多头输入
    template <typename BUF_TYPE>
    __aicore__ inline void CosSinBroadcast(const AscendC::GlobalTensor<uint8_t> &extraGm, uint32_t z,
                                           const AscendC::LocalTensor<BUF_TYPE> &tempBuf, uint32_t Calclen)
    {
        if constexpr (IF_COS_BROADCAST) {
            AscendC::DataCopyParams copyParams = {1, static_cast<uint16_t>(Calclen * sizeof(BUF_TYPE) / BLK_SIZE), 0,
                                                  0};
            DataCopy(tempBuf[cosPad_],
                     cosGm_[blockIdx_ * nlCoreRun_ * tilingData_->hiddenSizeQ + z * tilingData_->hiddenSizeQ],
                     copyParams);
            DataCopy(tempBuf[sinPad_],
                     sinGm_[blockIdx_ * nlCoreRun_ * tilingData_->hiddenSizeQ + z * tilingData_->hiddenSizeQ],
                     copyParams);
        } else {
            AscendC::GlobalTensor<CosDtype> extraGmCosDtype;
            extraGmCosDtype.SetGlobalBuffer((__gm__ CosDtype *)extraGm.GetPhyAddr());
            AscendC::LocalTensor<CosDtype> tempBufCosDtype = tempBuf.template ReinterpretCast<CosDtype>();
            CosSinCommonBroardcast(extraGmCosDtype, z, tempBufCosDtype, Calclen);
        }
    }

    // qk 公用函数
    template <typename BUF_TYPE>
    __aicore__ inline void QkComm(const AscendC::GlobalTensor<BUF_TYPE> &src,
                                  const AscendC::GlobalTensor<uint8_t> &extraGm1, uint32_t hiddenSizeTmp,
                                  const AscendC::LocalTensor<BUF_TYPE> &tempBuf, uint32_t headNumTemp)
    {
        uint16_t hiddenSizeBlk = static_cast<uint16_t>(hiddenSizeTmp / ELE_NUM_FP16);
        AscendC::SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
        DataCopy(tempBuf[oriPos_], src, {1, hiddenSizeBlk, 0, 0});
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
        if (alignRotary_ == 0) {
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
            AscendC::DataCopyParams copyParams = {static_cast<uint16_t>(headNumTemp * tilingData_->rotaryCoeff / 2),
                                                  static_cast<uint16_t>(rotaryStrideOffset / ELE_NUM_FP16),
                                                  static_cast<uint16_t>(rotaryStrideOffset / ELE_NUM_FP16),
                                                  static_cast<uint16_t>(rotaryStrideOffset / ELE_NUM_FP16)};
            DataCopy(tempBuf[removeBefore_ + rotaryStrideOffset], tempBuf[oriPos_], copyParams);
            DataCopy(tempBuf[removeBefore_], tempBuf[oriPos_ + rotaryStrideOffset], copyParams);
        } else {
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
            AscendC::GlobalTensor<BUF_TYPE> extraGm1BufType;
            extraGm1BufType.SetGlobalBuffer((__gm__ BUF_TYPE *)extraGm1.GetPhyAddr());
            AscendC::DataCopyParams copyParams = {1, static_cast<uint16_t>(hiddenSizeBlk), 0, 0};
            // ub -> workspace[0~hiddensize]
            DataCopy(extraGm1BufType[offsetExtraGm_], tempBuf[oriPos_], copyParams);
            // ub -> workspace[hiddensize ~ 2 * hiddensize]
            DataCopy(extraGm1BufType[offsetExtraGm_ + hiddenSizeTmp], tempBuf[oriPos_], copyParams);
            // workspace[rotary ~ hiddensize + rotary] -> ub[hiddensize ~ 2 * hiddensize]
            AscendC::PipeBarrier<PIPE_ALL>();
            DataCopy(tempBuf[removeBefore_], extraGm1BufType[offsetExtraGm_ + rotateStride_], copyParams);
            // gm[hiddensize - rotary ~ 2 * hiddensize - rotary] -> ub[2 *hiddensize ~ 3 * hiddensize]
            DataCopy(tempBuf[padBefore_], extraGm1BufType[offsetExtraGm_ + hiddenSizeTmp - rotateStride_], copyParams);
        }
    }

    // 主体计算逻辑
    template<typename BUF_TYPE>
    __aicore__ inline void CalcRope(const AscendC::LocalTensor<BUF_TYPE> &tempBuf, uint32_t repeatTimes1,
        uint32_t oriPosTemp, uint32_t removeTemp, uint32_t padTemp,
        uint32_t posTemp, uint32_t res)
    {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        if (g_coreType == AscendC::AIC) return;
#endif
        mul_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[oriPosTemp], tempBuf[cosPad_], tempBuf[oriPosTemp], repeatTimes1,
                                               1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
                                               DEFAULT_REPEAT_STRIDE);
        mul_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[padTemp], tempBuf[posTemp], tempBuf[padTemp], repeatTimes1, 1, 1,
                                               1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();

        mul_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[removeTemp], tempBuf[sinPad_], tempBuf[removeTemp], repeatTimes1,
                                               1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
                                               DEFAULT_REPEAT_STRIDE);
        mul_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[padTemp], tempBuf[sinPad_], tempBuf[padTemp], repeatTimes1, 1, 1,
                                               1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();

        mul_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[removeTemp], tempBuf[negOne_], tempBuf[removeTemp], repeatTimes1,
                                               1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
                                               DEFAULT_REPEAT_STRIDE);
        add_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[padTemp], tempBuf[oriPosTemp], tempBuf[padTemp], repeatTimes1, 1,
                                               1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
                                               DEFAULT_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();

        add_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[res], tempBuf[removeTemp], tempBuf[padTemp], repeatTimes1, 1, 1,
                                               1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 主体计算逻辑
    template <typename BUF_TYPE>
    __aicore__ inline void CalcRopeAlign(const AscendC::LocalTensor<BUF_TYPE> &tempBuf, uint32_t repeatTimes1,
                                         uint32_t oriPosTemp, uint32_t removeTemp, uint32_t padTemp)
    {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        if (g_coreType == AscendC::AIC) return;
#endif

#if (__CCE_AICORE__ == 300)
        AscendC::Mul(
            tempBuf[oriPosTemp], tempBuf[cosPad_], tempBuf[oriPosTemp], (uint64_t)this->repeatSize_, repeatTimes1,
            AscendC::BinaryRepeatParams(1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE));
        AscendC::Mul(
            tempBuf[removeTemp], tempBuf[negOne_], tempBuf[removeTemp], (uint64_t)this->repeatSize_, repeatTimes1,
            AscendC::BinaryRepeatParams(1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(
            tempBuf[removeTemp], tempBuf[removeTemp], tempBuf[sinPad_], (uint64_t)this->repeatSize_, repeatTimes1,
            AscendC::BinaryRepeatParams(1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(
            tempBuf[padTemp], tempBuf[removeTemp], tempBuf[oriPosTemp], (uint64_t)this->repeatSize_, repeatTimes1,
            AscendC::BinaryRepeatParams(1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE));
        AscendC::PipeBarrier<PIPE_V>();
#else
        mul_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[oriPosTemp], tempBuf[cosPad_], tempBuf[oriPosTemp], repeatTimes1,
                                               1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
                                               DEFAULT_REPEAT_STRIDE);
        mul_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[removeTemp], tempBuf[negOne_], tempBuf[removeTemp], repeatTimes1,
                                               1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
                                               DEFAULT_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();

        mul_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[removeTemp], tempBuf[sinPad_], tempBuf[removeTemp], repeatTimes1,
                                               1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
                                               DEFAULT_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();

        add_v<ArchType::ASCEND_V220, BUF_TYPE>(tempBuf[padTemp], tempBuf[removeTemp], tempBuf[oriPosTemp], repeatTimes1,
                                               1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
                                               DEFAULT_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
#endif
    }

    __aicore__ inline void SliceCalculation(uint32_t sliceSizeTmp,
        uint32_t& sliceTimeQ, uint32_t& lastSliceSizeQ,
        uint32_t& sliceTimeK, uint32_t& lastSliceSizeK)
    {
        // 判断是否需要切块计算
        if (tilingData_->hiddenSizeQ > sliceSizeTmp && sliceSizeTmp != 0) {
            sliceTimeQ = (tilingData_->hiddenSizeQ + sliceSizeTmp - 1) / sliceSizeTmp; // 向上取整
            lastSliceSizeQ = tilingData_->hiddenSizeQ - (sliceTimeQ - 1) * sliceSizeTmp; // 1024
        } else {
            sliceTimeQ = 1;
            lastSliceSizeQ = tilingData_->hiddenSizeQ;
        }

        if (this->tilingData_->hiddenSizeK > sliceSizeTmp && sliceSizeTmp != 0) {
            sliceTimeK = (this->tilingData_->hiddenSizeK + sliceSizeTmp - 1) / sliceSizeTmp; // 向上取整
            lastSliceSizeK = this->tilingData_->hiddenSizeK - (sliceTimeK - 1) * sliceSizeTmp;
        } else {
            sliceTimeK = 1;
            lastSliceSizeK = this->tilingData_->hiddenSizeK;
        }
    }

public:
    RotaryPosEmbInferTilingData *tilingData_ = nullptr;
    AscendC::GlobalTensor<QkDtype> qGm_;
    AscendC::GlobalTensor<QkDtype> kGm_;
    AscendC::GlobalTensor<CosDtype> cosGm_;
    AscendC::GlobalTensor<CosDtype> sinGm_;
    AscendC::GlobalTensor<int32_t> seqLenGm_;
    AscendC::GlobalTensor<QkDtype> outQGm_;
    AscendC::GlobalTensor<QkDtype> outKGm_;
    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> seqLenQueue_;

    uint32_t cosPad_{0}; // broadcast的cos在uB中的位置
    uint32_t sinPad_{0}; // broadcast的sin在uB中的位置
    uint32_t negOne_{0}; // -1 -1 -1 0 0 0在uB中的位置
    uint32_t oriPos_{0}; // q,k在uB中的位置
    uint32_t padBefore_{0}; // 保存qk[-x : hiddensize - x]
    uint32_t removeBefore_{0}; // 保存qk[x : hiddensize + x]
    uint32_t repeatSize_{0}; // 一拍做几个元素
    uint32_t maxProcessNum_{0}; // 最大处理元素个数
    uint32_t repeatTimesQ_{0}; // q重复次数
    uint32_t repeatTimesK_{0}; // k重复次数
    uint32_t hiddenSizeAlign_{0}; // 对齐后的hiddensize
    uint32_t repeatTimes_{0}; // 对齐后重复次数
    uint32_t headNum_{0}; // 几个头
    uint32_t hiddenSize_{0}; // hiddensizeQ,K的最大值
    uint32_t nlCoreRun_{0}; // 非最后一个核需要跑几次
    uint32_t lCoreRun_{0};  // 最后一个核需要跑几次
    uint32_t batchSize_{0}; // batch向上取整
    uint32_t rotateStride_{0}; // headdim / 旋转系数
    uint64_t offsetExtraGm_{0}; // 使用workspace需要的offset
    uint32_t dynamicRound_{0}; // 每个核做几轮
    uint32_t alignHalfHeadDim_{0}; // headDim / 旋转系数 * 2 是否对齐
    uint32_t rotaryStrideOffset{0}; // 每次旋转长度
    uint32_t alignRotary_; // 旋转距离是否对齐
    uint32_t syncOffset_;
    uint32_t blockIdx_;
};

#endif