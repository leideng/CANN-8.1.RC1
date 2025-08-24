/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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
 * \file mat_mul_nd2nz_stress_detect.h
 * \brief
 */
#ifndef __OP_MM_STRESS_DETECT_ND2NZ_H__
#define __OP_MM_STRESS_DETECT_ND2NZ_H__

namespace MatmulStressDetect {

using namespace AscendC;
using namespace matmul;
#if defined(__CCE_KT_TEST__)
using namespace std;
#endif

constexpr uint32_t VNCHW_SIZE = 16;
constexpr uint64_t BLOCK_SIZE_BYTE = 32;

constexpr uint64_t REPEAT_TIMES_MAX = 255;
constexpr uint64_t SINGLE_COPY_SIZE = 256;
constexpr uint64_t ALIGNED_H = 16;

constexpr uint32_t M_BLOCK_NUM_ELE_LIST[16] = {1, 16, 8, 16, 4, 16, 8, 16, 2, 16, 8, 16, 4, 16, 8, 16};
constexpr uint32_t GCD_LIST[16] = {16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1};
const TransDataTo5HDParams PARA_ONE(false, false, 1, 0, 0);

enum class ND2NZ_DB_TYPE { IN_OUTPUT, OUTPUT, NO_DB_REUSE_OUTPUT };

template <class T>
__aicore__ inline void Copy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint32_t count) {
    constexpr uint32_t copyLen = SINGLE_COPY_SIZE / sizeof(T);
    const CopyRepeatParams para(1, 1, 8, 8);
    uint32_t repeatTimes = count / copyLen;
    uint32_t tail = count % copyLen;
    uint32_t offset = repeatTimes * copyLen;
    Copy(dstLocal, srcLocal, copyLen, repeatTimes, para);
    if (tail != 0) {
        Copy(dstLocal[offset], srcLocal[offset], tail, 1, para);
    }
}

__aicore__ inline int32_t Align2(uint32_t x, uint32_t divisor) {
    uint32_t remainder = x & (divisor - 1);  // 计算m与divisor的模数
    if (remainder == 0) {
        return x;  // 如果m已经能被2^n整除，直接返回m
    }
    return (x + divisor - remainder);  // 否则找到
}

template <class T>
class KernelND2NZ {
   public:
    __aicore__ inline KernelND2NZ(){};
    __aicore__ inline void CopyIn(uint64_t progress, LocalTensor<T>& dstLocal);
    __aicore__ inline void CopyOutMM(uint64_t progress, LocalTensor<T>& srcLocal);
    template <bool HAS_FLAG = true>
    __aicore__ inline void PadD(uint64_t progress, LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
                                int eventIn = EVENT_ID0, int eventOut = EVENT_ID0);
    template <bool HAS_FLAG = true>
    __aicore__ inline void PadDMain(uint64_t progress, LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
                                    int eventIn = EVENT_ID0, int eventOut = EVENT_ID0);
    template <bool HAS_FLAG = true>
    __aicore__ inline void PadDOne(uint64_t progress, LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
                                   int eventIn = EVENT_ID0, int eventOut = EVENT_ID0);
    template <bool HAS_FLAG = true>
    __aicore__ inline void PadDAligned(uint64_t progress, LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
                                       int eventIn = EVENT_ID0, int eventOut = EVENT_ID0);
    __aicore__ inline bool SetBufBMM();
    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src, uint32_t height, uint32_t width, uint32_t batch,
                                TBuf<TPosition::VECCALC>& ubBuffer, uint32_t usedCoreNum);
    template <ND2NZ_DB_TYPE TYPE, bool noZero = false>
    __aicore__ inline bool SetBufMM();

    __aicore__ inline bool ProcessBMM();
    __aicore__ inline bool ProcessMM();
    __aicore__ inline void ProcessInOutDB();
    __aicore__ inline void ProcessNoDBReuse();
    __aicore__ inline void ProcessOutDBReuse();

   private:
    __aicore__ inline void CopyOutDirect(uint64_t gmOutOffset, uint32_t startPad, uint16_t total, uint64_t progress);
    __aicore__ inline void CopyOutPageInit(uint64_t& gmOutOffset, uint32_t startPad, uint32_t& bufOffset);
    __aicore__ inline void CopyOutMakePage(uint32_t nLoop, uint32_t& bufOffset);
    __aicore__ inline void CopyOutPageMainImp(uint64_t& gmOutOffset, uint32_t nLoop, uint32_t& bufOffset);
    __aicore__ inline void CopyOutPageMain(uint64_t& gmOutOffset, uint32_t mPage, uint32_t startPad, uint32_t total,
                                           uint32_t& bufOffset, uint64_t progress);
    __aicore__ inline void CopyOutPageEnd(uint64_t gmOutOffset, uint32_t res, uint32_t& bufOffset);
    __aicore__ inline void CopyOutPage(uint64_t gmOutOffset, uint32_t mPage, uint32_t total, uint32_t startPad,
                                       uint64_t progress);
    __aicore__ inline void CopyOutBatchReform(uint64_t gmOutOffset, uint32_t mPage, uint32_t total, uint32_t startPad,
                                              uint64_t progress);
    __aicore__ inline void ComputeBMM(uint64_t progress);

   private:
    TBuf<TPosition::VECCALC>* ubPtr_;
    GlobalTensor<T> srcGM;
    GlobalTensor<T> dstGM;
    LocalTensor<T> inBuf_;
    LocalTensor<T> inBuf2_;
    LocalTensor<T> midBuf_;
    LocalTensor<T> outBuf_;
    LocalTensor<T> outBuf2_;
    LocalTensor<T> zeroBuf_;
    uint32_t padSize_;
    uint32_t height_;
    uint32_t hAligned_;
    uint32_t width_;
    uint32_t batch_;
    uint32_t wTail_;
    uint32_t hBuffer_;
    uint32_t nFullProgress_;
    uint32_t heightTotalTail_;
    uint16_t hPad_;
    uint32_t blockDim_;
    uint32_t blockIdx_;
    uint32_t hBlockNum_;
    uint32_t copyInSize_;
    uint64_t c0_;
    uint32_t copyInRepeat_;
    uint16_t widthBlockTotal_;
    bool noPadD_;
};

template <class T>
__aicore__ inline void KernelND2NZ<T>::CopyIn(uint64_t progress, LocalTensor<T>& dstLocal) {
    uint64_t curCopyInSize = progress == nFullProgress_ ? Align2(heightTotalTail_ * width_, c0_) : copyInSize_;
    uint64_t gmInOffset = copyInSize_ * progress;
    DataCopy(dstLocal, srcGM[gmInOffset], curCopyInSize);
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::CopyOutMM(uint64_t progress, LocalTensor<T>& srcLocal) {
    uint64_t oneColSizeGM = Align2(height_ * batch_, ALIGNED_H) * c0_;
    uint64_t oneColSize = hBuffer_ * c0_;
    uint32_t copyOutSize = progress == nFullProgress_ ? Align2(heightTotalTail_, ALIGNED_H) * c0_ : oneColSize;
    if (progress == nFullProgress_) {
        for (uint32_t i = 0; i < widthBlockTotal_; i++) {
            Duplicate(srcLocal[oneColSize * i + heightTotalTail_ * c0_], T(0), padSize_);
        }
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    for (uint32_t i = 0; i < widthBlockTotal_; i++) {
        DataCopy(dstGM[oneColSizeGM * i + oneColSize * progress], srcLocal[oneColSize * i], copyOutSize);
    }
}

template <class T>
template <bool HAS_FLAG>
__aicore__ inline void KernelND2NZ<T>::PadD(uint64_t progress, LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
                                            int eventIn, int eventOut) {
    if (width_ == 1) {
        PadDOne<HAS_FLAG>(progress, dstLocal, srcLocal, eventIn, eventOut);
    } else if (wTail_ == 0) {
        PadDAligned<HAS_FLAG>(progress, dstLocal, srcLocal, eventIn, eventOut);
    } else {
        PadDMain<HAS_FLAG>(progress, dstLocal, srcLocal, eventIn, eventOut);
    }
}

template <class T>
template <bool HAS_FLAG>
__aicore__ inline void KernelND2NZ<T>::PadDMain(uint64_t progress, LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
                                                int eventIn, int eventOut) {
    uint32_t widthBlock = width_ / c0_;
    uint64_t dstLocalList[VNCHW_SIZE];
    uint64_t srcLocalList[VNCHW_SIZE];

    for (uint32_t i = 0; i < VNCHW_SIZE; i++) {
        srcLocalList[i] = (i * hBlockNum_ * width_ * sizeof(T) + srcLocal.GetPhyAddr());
        dstLocalList[i] = (i * BLOCK_SIZE_BYTE + midBuf_.GetPhyAddr());
    }

    if (copyInRepeat_ == 1) {
        TransDataTo5HD<T>(dstLocalList, srcLocalList, PARA_ONE);
    } else {
        TransDataTo5HDParams para(false, false, copyInRepeat_, VNCHW_SIZE, 1);
        TransDataTo5HD<T>(dstLocalList, srcLocalList, para);
    }

    if constexpr (HAS_FLAG) {
        SetFlag<HardEvent::V_MTE2>(eventIn);
        WaitFlag<HardEvent::MTE3_V>(eventOut);
    }

    for (uint32_t i = 0; i < widthBlock; i++) {
        if constexpr (sizeof(T) == sizeof(half)) {
            for (uint32_t j = 0; j < VNCHW_SIZE; j++) {
                srcLocalList[j] = (j * BLOCK_SIZE_BYTE + midBuf_.GetPhyAddr() + i * VNCHW_SIZE * BLOCK_SIZE_BYTE);
                dstLocalList[j] =
                    (j * BLOCK_SIZE_BYTE * hBlockNum_ + dstLocal.GetPhyAddr() + hBuffer_ * i * BLOCK_SIZE_BYTE);
            }
        }
        if constexpr (sizeof(T) == sizeof(float)) {
            for (uint32_t j = 0; j < VNCHW_SIZE; j++) {
                srcLocalList[j] =
                    ((2 * (j % 8) + j / 8) * BLOCK_SIZE_BYTE + midBuf_.GetPhyAddr() + i * VNCHW_SIZE * BLOCK_SIZE_BYTE);
                dstLocalList[j] = ((j / 2 + 8 * (j % 2)) * BLOCK_SIZE_BYTE * hBlockNum_ + dstLocal.GetPhyAddr() +
                                   hBuffer_ * i * BLOCK_SIZE_BYTE);
            }
        }
        if (hBlockNum_ == 1) {
            TransDataTo5HD<T>(dstLocalList, srcLocalList, PARA_ONE);
        } else {
            uint64_t srcRepStride = width_ * sizeof(T) / 2;
            TransDataTo5HDParams para(false, false, hBlockNum_, 1, srcRepStride);
            TransDataTo5HD<T>(dstLocalList, srcLocalList, para);
        }
    }

    constexpr uint32_t copyLen = SINGLE_COPY_SIZE / sizeof(T);

    if (width_ > c0_ && wTail_ != 0) {
        uint16_t dstBlockStride = sizeof(T) * width_ / 2;

        if (8 * dstBlockStride > UINT8_MAX){
            for (int i = 0; i < hBlockNum_ / 8; i++) {
                Duplicate(midBuf_[8 * dstBlockStride * c0_ * i], T(0), copyLen, 1, dstBlockStride, 8 * dstBlockStride);
            }
        }
        else {
            Duplicate(midBuf_, T(0), copyLen, hBlockNum_ / 8, dstBlockStride, 8 * dstBlockStride);
        }

        Duplicate(midBuf_[c0_ * dstBlockStride * (hBlockNum_ / 8) * 8], T(0), (copyLen / 8) * (hBlockNum_ % 8), 1,
                dstBlockStride, 0);
        }

    if constexpr (sizeof(T) == sizeof(half)) {
        for (uint32_t j = 0; j < wTail_; j++) {
            srcLocalList[j] = (j * BLOCK_SIZE_BYTE + midBuf_.GetPhyAddr() + widthBlock * VNCHW_SIZE * BLOCK_SIZE_BYTE);
        }
        if (width_ > c0_) {
            for (uint32_t j = wTail_; j < VNCHW_SIZE; j++) {
                srcLocalList[j] = midBuf_.GetPhyAddr();
            }
        } else {
            for (uint32_t j = wTail_; j < VNCHW_SIZE; j++) {
                srcLocalList[j] = zeroBuf_.GetPhyAddr();
            }
        }
        for (uint32_t j = 0; j < VNCHW_SIZE; j++) {
            dstLocalList[j] =
                (j * BLOCK_SIZE_BYTE * hBlockNum_ + dstLocal.GetPhyAddr() + hBuffer_ * widthBlock * BLOCK_SIZE_BYTE);
        }
    }
    if constexpr (sizeof(T) == sizeof(float)) {
        for (uint32_t j = 0; j < wTail_; j++) {
            srcLocalList[j] =
                ((2 * j) * BLOCK_SIZE_BYTE + midBuf_.GetPhyAddr() + widthBlock * VNCHW_SIZE * BLOCK_SIZE_BYTE);
            srcLocalList[j + 8] =
                ((2 * j + 1) * BLOCK_SIZE_BYTE + midBuf_.GetPhyAddr() + widthBlock * VNCHW_SIZE * BLOCK_SIZE_BYTE);
        }
        if (width_ > c0_) {
            for (uint32_t j = wTail_; j < 8; j++) {
                srcLocalList[j] = midBuf_.GetPhyAddr();
                srcLocalList[j + 8] = midBuf_.GetPhyAddr();
            }
        } else {
            for (uint32_t j = wTail_; j < 8; j++) {
                srcLocalList[j] = zeroBuf_.GetPhyAddr();
                srcLocalList[j + 8] = zeroBuf_.GetPhyAddr();
            }
        }

        for (uint32_t j = 0; j < VNCHW_SIZE; j++) {
            dstLocalList[j] = ((j / 2 + 8 * (j % 2)) * BLOCK_SIZE_BYTE * hBlockNum_ + dstLocal.GetPhyAddr() +
                               hBuffer_ * widthBlock * BLOCK_SIZE_BYTE);
        }
    }
    uint16_t dstBlockStride = sizeof(T) * width_ / 2;
    TransDataTo5HDParams para(false, false, hBlockNum_, 1, dstBlockStride);
    TransDataTo5HD<T>(dstLocalList, srcLocalList, para);
}

template <class T>
template <bool HAS_FLAG>
__aicore__ inline void KernelND2NZ<T>::PadDOne(uint64_t progress, LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
                                               int eventIn, int eventOut) {
    uint64_t dstLocalList[VNCHW_SIZE];
    uint64_t srcLocalList[VNCHW_SIZE];

    if constexpr (HAS_FLAG) {
        WaitFlag<HardEvent::MTE3_V>(eventOut);
    }

    if constexpr (sizeof(T) == sizeof(half)) {
        for (uint32_t i = 0; i < VNCHW_SIZE; i++) {
            srcLocalList[i] = zeroBuf_.GetPhyAddr();
            dstLocalList[i] = (i * BLOCK_SIZE_BYTE + dstLocal.GetPhyAddr());
        }

        srcLocalList[0] = srcLocal.GetPhyAddr();
    }

    if constexpr (sizeof(T) == sizeof(float)) {
        for (uint32_t i = 0; i < VNCHW_SIZE; i++) {
            srcLocalList[i] = zeroBuf_.GetPhyAddr();
        }
        srcLocalList[0] = srcLocal.GetPhyAddr();
        srcLocalList[8] = srcLocal.GetPhyAddr() + BLOCK_SIZE_BYTE;

        for (uint32_t i = 0; i < 8; i++) {
            dstLocalList[2 * i] = dstLocal.GetPhyAddr() + BLOCK_SIZE_BYTE * i;
            dstLocalList[2 * i + 1] = dstLocal.GetPhyAddr() + BLOCK_SIZE_BYTE * i + 8 * BLOCK_SIZE_BYTE;
        }
    }

    if (copyInRepeat_ == 1) {
        TransDataTo5HD<T>(dstLocalList, srcLocalList, PARA_ONE);
    } else {
        TransDataTo5HDParams para(false, false, copyInRepeat_, VNCHW_SIZE, sizeof(T) / 2);
        TransDataTo5HD<T>(dstLocalList, srcLocalList, para);
    }

    if constexpr (HAS_FLAG) {
        SetFlag<HardEvent::V_MTE2>(eventIn);
    }
}

template <class T>
template <bool HAS_FLAG>
__aicore__ inline void KernelND2NZ<T>::PadDAligned(uint64_t progress, LocalTensor<T>& dstLocal,
                                                   LocalTensor<T>& srcLocal, int eventIn, int eventOut) {
    if constexpr (HAS_FLAG) {
        WaitFlag<HardEvent::MTE3_V>(eventOut);
    }
    uint32_t repeatTimes = hBlockNum_ * 2;
    int nLoop = repeatTimes / REPEAT_TIMES_MAX;
    int loopTail = repeatTimes % REPEAT_TIMES_MAX;
    for (int i = 0; i < width_ / c0_; i++) {
        for (int j = 0; j < nLoop; j++) {
            Copy(dstLocal[hBlockNum_ * c0_ * ALIGNED_H * i + REPEAT_TIMES_MAX * j * 8 * c0_],
                 srcLocal[c0_ * i + REPEAT_TIMES_MAX * j * 8 * width_], SINGLE_COPY_SIZE / sizeof(T), REPEAT_TIMES_MAX,
                 {1, uint16_t(width_ / c0_), 8, uint16_t(8 * width_ / c0_)});
        }
        Copy(dstLocal[hBlockNum_ * c0_ * ALIGNED_H * i + REPEAT_TIMES_MAX * nLoop * 8 * c0_],
             srcLocal[c0_ * i + REPEAT_TIMES_MAX * nLoop * 8 * width_], SINGLE_COPY_SIZE / sizeof(T), loopTail,
             {1, uint16_t(width_ / c0_), 8, uint16_t(8 * width_ / c0_)});
    }
    if constexpr (HAS_FLAG) {
        SetFlag<HardEvent::V_MTE2>(eventIn);
    }
}

template <class T>
__aicore__ inline bool KernelND2NZ<T>::SetBufBMM() {
    uint32_t hTotal = height_ * batch_;
    uint32_t wAligned = Align2(width_, c0_);

    uint32_t hMax = TOTAL_UB_SIZE / sizeof(T) / (width_ + width_ + width_ + wAligned);
    // hBlockNumEle表示最少要几行连续数据才能32B对齐
    uint32_t hBlockNumEle = M_BLOCK_NUM_ELE_LIST[wTail_] * 2 / sizeof(T);
    hBlockNumEle = width_ == 1 ? 1 : hBlockNumEle;
    hBlockNumEle = hBlockNumEle == 0 ? 1 : hBlockNumEle;
    // gcd是c0_和width_的最大公约数
    uint32_t gcd = GCD_LIST[wTail_];
    if constexpr (sizeof(T) == sizeof(float)) {
        gcd = wTail_ == 0 ? 8 : gcd;
    }
    // hEle表示最小载入行数，为满足vnchwconv的要求，要乘个16
    uint32_t hEle = hBlockNumEle * ALIGNED_H;
    // eleNum是在ub_buffer和外轴限制的基础上，最多可载入几倍的hEle
    uint32_t eleNum = (hTotal + hEle - 1) / hEle;
    uint32_t eleNumTmp = hMax / hEle;
    eleNum = min(eleNumTmp, eleNum);
    eleNum = eleNum * hBlockNumEle > REPEAT_TIMES_MAX ? (REPEAT_TIMES_MAX / hBlockNumEle) : eleNum;

    if (eleNum == 0) {
        return false;
    }

    copyInRepeat_ = eleNum * width_ / gcd;

    hBuffer_ = eleNum * hEle;
    copyInSize_ = hBuffer_ * width_;
    // 16 * (eleNum * hBlockNumEle)*width_，计算地址偏移时使用
    hBlockNum_ = eleNum * hBlockNumEle;
    nFullProgress_ = hTotal / hBuffer_;
    heightTotalTail_ = hTotal % hBuffer_;

    midBuf_ = ubPtr_->Get<T>()[0];
    zeroBuf_ = ubPtr_->Get<T>()[copyInSize_];
    inBuf_ = ubPtr_->Get<T>()[copyInSize_ * 2];
    outBuf_ = ubPtr_->Get<T>()[copyInSize_ * 3];
    // 清零可以去掉，mad使用实际的大小计算，就不需要清零
    Duplicate(zeroBuf_, T(0), copyInSize_);

    PipeBarrier<PIPE_ALL>();
    return true;
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::Init(GM_ADDR dst, GM_ADDR src, uint32_t height, uint32_t width, uint32_t batch,
                                            TBuf<TPosition::VECCALC>& ubBuffer, uint32_t usedCoreNum) {
    height_ = height;
    width_ = width;
    batch_ = batch;
    uint32_t hTotal = height_ * batch_;

    blockDim_ = usedCoreNum;
    blockIdx_ = GetBlockIdx();

    c0_ = BLOCK_SIZE_BYTE / sizeof(T);

    srcGM.SetGlobalBuffer((__gm__ T*)src);
    dstGM.SetGlobalBuffer((__gm__ T*)dst);
    ubPtr_ = &ubBuffer;

    noPadD_ = (width_ == c0_);

    uint32_t batchTail = height_ % ALIGNED_H;
    hPad_ = batchTail == 0 ? 0 : ALIGNED_H - batchTail;

    padSize_ = hPad_ * c0_;

    hAligned_ = Align2(height_, ALIGNED_H);

    uint32_t widthBlock = width_ / c0_;
    wTail_ = width_ & (c0_ - 1);

    widthBlockTotal_ = wTail_ ? widthBlock + 1 : widthBlock;
}

template <class T>
template <ND2NZ_DB_TYPE TYPE, bool noZero>
__aicore__ inline bool KernelND2NZ<T>::SetBufMM() {
    uint32_t hTotal = height_ * batch_;
    uint32_t wAligned = Align2(width_, c0_);

    uint32_t ubTotalWidth = 0;

    if constexpr (TYPE == ND2NZ_DB_TYPE::IN_OUTPUT) {
        ubTotalWidth = 3 * width_ + 2 * wAligned;
    }
    if constexpr (TYPE == ND2NZ_DB_TYPE::OUTPUT) {
        ubTotalWidth = width_ + 2 * wAligned;
    }
    if constexpr (TYPE == ND2NZ_DB_TYPE::NO_DB_REUSE_OUTPUT) {
        ubTotalWidth = width_ + wAligned;
    }

    if (!(width_ > c0_) && !noZero) {
        ubTotalWidth += width_;
    }

    uint32_t hMax = TOTAL_UB_SIZE / sizeof(T) / ubTotalWidth;

    uint32_t hBlockNumEle = M_BLOCK_NUM_ELE_LIST[wTail_] * 2 / sizeof(T);
    hBlockNumEle = width_ == 1 ? 1 : hBlockNumEle;
    hBlockNumEle = hBlockNumEle == 0 ? 1 : hBlockNumEle;
    uint32_t gcd = GCD_LIST[wTail_];
    if constexpr (sizeof(T) == sizeof(float)) {
        gcd = wTail_ == 0 ? 8 : gcd;
    }

    uint32_t hEle = hBlockNumEle * ALIGNED_H;
    uint32_t eleNum = (hTotal + hEle - 1) / hEle;
    uint32_t eleNumTmp = hMax / hEle;
    eleNum = min(eleNumTmp, eleNum);
    eleNum = eleNum * hBlockNumEle > REPEAT_TIMES_MAX ? (REPEAT_TIMES_MAX / hBlockNumEle) : eleNum;

    if (eleNum == 0) {
        return false;
    }

    copyInRepeat_ = eleNum * width_ / gcd;
    hBuffer_ = eleNum * hEle;
    copyInSize_ = hBuffer_ * width_;
    hBlockNum_ = eleNum * hBlockNumEle;
    nFullProgress_ = hTotal / hBuffer_;
    heightTotalTail_ = hTotal % hBuffer_;

    uint32_t ubTail = 0;

    midBuf_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];

    ubTail += copyInSize_ * sizeof(T);

    if ((width_ < c0_) && !noZero) {
        zeroBuf_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];
        ubTail += copyInSize_ * sizeof(T);
    }
    if (TYPE == ND2NZ_DB_TYPE::IN_OUTPUT) {
        inBuf_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];
        ubTail += copyInSize_ * sizeof(T);
        inBuf2_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];
        ubTail += copyInSize_ * sizeof(T);
    }

    if (TYPE == ND2NZ_DB_TYPE::IN_OUTPUT || TYPE == ND2NZ_DB_TYPE::OUTPUT ||
        TYPE == ND2NZ_DB_TYPE::NO_DB_REUSE_OUTPUT) {
        outBuf_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];
        ubTail += hBuffer_ * wAligned * sizeof(T);
    }
    if (TYPE == ND2NZ_DB_TYPE::IN_OUTPUT || TYPE == ND2NZ_DB_TYPE::OUTPUT) {
        outBuf2_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];
        ubTail += hBuffer_ * wAligned * sizeof(T);
    }
    if ((width_ < c0_) && !noZero) {
        Duplicate(zeroBuf_, T(0), copyInSize_);
    }

    PipeBarrier<PIPE_ALL>();
    return true;
}

template <class T>
__aicore__ inline bool KernelND2NZ<T>::ProcessBMM() {
    if (SetBufBMM()) {
        uint32_t nLoop = heightTotalTail_ ? nFullProgress_ + 1 : nFullProgress_;
        for (int32_t i = blockIdx_; i < nLoop; i += blockDim_) {
            ComputeBMM(i);

            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        PipeBarrier<PIPE_ALL>();
        return true;
    }
    return false;
}

template <class T>
__aicore__ inline bool KernelND2NZ<T>::ProcessMM() {
    if (width_ == 1) {
        SetBufMM<ND2NZ_DB_TYPE::IN_OUTPUT, false>();
        ProcessInOutDB();
        return true;
    } else if (width_ % c0_ == 0) {
        if (SetBufMM<ND2NZ_DB_TYPE::IN_OUTPUT, true>()) {
            ProcessInOutDB();
            return true;
        }
        return false;
    } else if (SetBufMM<ND2NZ_DB_TYPE::OUTPUT>()) {
        ProcessOutDBReuse();
        return true;
    } else if (SetBufMM<ND2NZ_DB_TYPE::NO_DB_REUSE_OUTPUT>()) {
        ProcessNoDBReuse();
        return true;
    }
    return false;
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::ProcessInOutDB() {
    uint32_t nLoop = heightTotalTail_ ? nFullProgress_ + 1 : nFullProgress_;
    uint32_t j = 0;
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    for (int32_t i = blockIdx_; i < nLoop; i += blockDim_, j++) {
        if (j % 2 == 1) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CopyIn(i, inBuf_);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            PadD<true>(i, outBuf_, inBuf_, EVENT_ID0, EVENT_ID0);

            CopyOutMM(i, outBuf_);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        } else {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            CopyIn(i, inBuf2_);

            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            PadD<true>(i, outBuf2_, inBuf2_, EVENT_ID1, EVENT_ID1);

            CopyOutMM(i, outBuf2_);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        }
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::ProcessOutDBReuse() {
    uint32_t nLoop = heightTotalTail_ ? nFullProgress_ + 1 : nFullProgress_;
    uint32_t j = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    for (int32_t i = blockIdx_; i < nLoop; i += blockDim_, j++) {
        if (j % 2 == 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            CopyIn(i, outBuf_);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            PadDMain<false>(i, outBuf_, outBuf_);
            CopyOutMM(i, outBuf_);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        } else {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            CopyIn(i, outBuf2_);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            PadDMain<false>(i, outBuf2_, outBuf2_);
            CopyOutMM(i, outBuf2_);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::ProcessNoDBReuse() {
    uint32_t nLoop = heightTotalTail_ ? nFullProgress_ + 1 : nFullProgress_;

    for (int32_t i = blockIdx_; i < nLoop; i += blockDim_) {
        CopyIn(i, outBuf_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        PadDMain<false>(i, outBuf_, outBuf_);

        CopyOutMM(i, outBuf_);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::CopyOutDirect(uint64_t gmOutOffset, uint32_t startPad, uint16_t total,
                                                     uint64_t progress) {
    uint32_t start = startPad - hPad_;
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // 处理上个核心的尾部没完成的batch
    if (start > total) {
        if (hAligned_ - total <= UINT16_MAX) {
            DataCopy(dstGM[gmOutOffset], outBuf_,
                     {widthBlockTotal_, total, 0, static_cast<uint16_t>(hAligned_ - total)});
        } else {
            for (uint16_t i = 0; i < widthBlockTotal_; i++) {
                DataCopy(dstGM[gmOutOffset + hAligned_ * c0_ * i], outBuf_[total * c0_ * i], {1, total, 0, 0});
            }
        }
        return;
    } else if (start == total) {
        DataCopy(dstGM[gmOutOffset], outBuf_,
                 {widthBlockTotal_, uint16_t(start), uint16_t(hBuffer_ - start), uint16_t(hAligned_ - start)});
        DataCopy(dstGM[gmOutOffset + start * c0_], zeroBuf_, {widthBlockTotal_, hPad_, 0, uint16_t(height_)});
        return;
    }

    if (startPad != hAligned_) {
        DataCopy(dstGM[gmOutOffset], outBuf_,
                 {widthBlockTotal_, uint16_t(start), uint16_t(hBuffer_ - start), uint16_t(hAligned_ - start)});
        DataCopy(dstGM[gmOutOffset + start * c0_], zeroBuf_, {widthBlockTotal_, hPad_, 0, uint16_t(height_)});
        gmOutOffset += startPad * c0_ + (widthBlockTotal_ - 1) * hAligned_ * c0_;

    } else {
        start = 0;
    }
    // 处理完整的batch
    uint32_t nLoop = (total - start) / height_;
    uint16_t res = (total - start) % height_;

    if (height_ <= total - start) {
        for (int i = 0; i < nLoop; i++) {
            DataCopy(dstGM[gmOutOffset], outBuf_[start * c0_ + height_ * c0_ * i],
                     {widthBlockTotal_, uint16_t(height_), uint16_t(hBuffer_ - height_), hPad_});
            DataCopy(dstGM[gmOutOffset + height_ * c0_], zeroBuf_, {widthBlockTotal_, hPad_, 0, uint16_t(height_)});
            gmOutOffset += hAligned_ * c0_ * widthBlockTotal_;
        }
    }
    // 处理尾部余下的batch
    if (res) {
        if (hAligned_ - total <= UINT16_MAX) {
            DataCopy(dstGM[gmOutOffset], outBuf_[start * c0_ + height_ * c0_ * nLoop],
                     {widthBlockTotal_, res, uint16_t(hBuffer_ - res), uint16_t(hAligned_ - res)});
        } else {
            for (uint16_t i = 0; i < widthBlockTotal_; i++) {
                DataCopy(dstGM[gmOutOffset + hAligned_ * c0_ * i],
                         outBuf_[start * c0_ + height_ * c0_ * nLoop + total * c0_ * i], {1, res, 0, 0});
            }
        }
    }
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::CopyOutPageInit(uint64_t& gmOutOffset, uint32_t startPad, uint32_t& bufOffset) {
    uint32_t start = startPad - hPad_;
    uint32_t startSize = start * c0_;

    for (int k = 0; k < widthBlockTotal_; k++) {
        if (start > 0) {
            Copy(midBuf_[startPad * c0_ * k], outBuf_[hBuffer_ * c0_ * k], startSize);
            Duplicate(midBuf_[startPad * c0_ * k + startSize], T(0), padSize_);
        } else {
            Duplicate(midBuf_[startPad * c0_ * k], T(0), startPad * c0_);
        }
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(dstGM[gmOutOffset], midBuf_, {widthBlockTotal_, uint16_t(startPad), 0, uint16_t(hAligned_ - startPad)});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

    bufOffset = startSize;
    gmOutOffset += startPad * c0_ + (widthBlockTotal_ - 1) * hAligned_ * c0_;
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::CopyOutMakePage(uint32_t nLoop, uint32_t& bufOffset) {
    for (int j = 0; j < nLoop; j++) {
        for (int k = 0; k < widthBlockTotal_; k++) {
            Copy(midBuf_[c0_ * hAligned_ * (k + widthBlockTotal_ * j)],
                 outBuf_[bufOffset + hBuffer_ * c0_ * k + c0_ * height_ * j], height_ * c0_);
            Duplicate(midBuf_[height_ * c0_ + c0_ * hAligned_ * (k + widthBlockTotal_ * j)], T(0), padSize_);
        }
    }
    bufOffset += c0_ * height_ * nLoop;
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::CopyOutPageMainImp(uint64_t& gmOutOffset, uint32_t nLoop, uint32_t& bufOffset) {
    CopyOutMakePage(nLoop, bufOffset);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(dstGM[gmOutOffset], midBuf_, hAligned_ * widthBlockTotal_ * c0_ * nLoop);

    gmOutOffset += hAligned_ * widthBlockTotal_ * c0_ * nLoop;
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::CopyOutPageMain(uint64_t& gmOutOffset, uint32_t mPage, uint32_t startPad,
                                                       uint32_t total, uint32_t& bufOffset, uint64_t progress) {
    uint32_t mPage2 = mPage / widthBlockTotal_;
    uint32_t nLoopIn = mPage2 / hAligned_;
    uint32_t mFinal = ((total - startPad + hPad_) / height_ + 1) * hPad_ + total;
    uint32_t nFull = (startPad == hAligned_) ? mFinal / hAligned_ : (mFinal - startPad) / hAligned_;
    uint32_t nLoopOut = nFull / nLoopIn;

    for (int i = 0; i < nLoopOut; i++) {
        CopyOutPageMainImp(gmOutOffset, nLoopIn, bufOffset);
    }
    uint32_t nLoopTail = nFull % nLoopIn;

    CopyOutPageMainImp(gmOutOffset, nLoopTail, bufOffset);
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::CopyOutPageEnd(uint64_t gmOutOffset, uint32_t res, uint32_t& bufOffset) {
    for (int k = 0; k < widthBlockTotal_; k++) {
        Copy(midBuf_[c0_ * res * k], outBuf_[bufOffset + hBuffer_ * c0_ * k], res * c0_);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(dstGM[gmOutOffset], midBuf_, {uint16_t(widthBlockTotal_), uint16_t(res), 0, uint16_t(hAligned_ - res)});
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::CopyOutPage(uint64_t gmOutOffset, uint32_t mPage, uint32_t total,
                                                   uint32_t startPad, uint64_t progress) {
    uint32_t bufOffset = 0;
    uint32_t start = startPad - hPad_;

    if (startPad != hAligned_) {
        CopyOutPageInit(gmOutOffset, startPad, bufOffset);
    } else {
        start = 0;
    }

    uint32_t res = (total - start) % height_;

    CopyOutPageMain(gmOutOffset, mPage, startPad, total, bufOffset, progress);

    if (res) {
        CopyOutPageEnd(gmOutOffset, res, bufOffset);
    }
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::CopyOutBatchReform(uint64_t gmOutOffset, uint32_t mPage, uint32_t total,
                                                          uint32_t startPad, uint64_t progress) {
    uint32_t mPage2 = mPage / widthBlockTotal_;

    if (hAligned_ > mPage2) {
        CopyOutDirect(gmOutOffset, startPad, total, progress);
        return;
    }
    CopyOutPage(gmOutOffset, mPage, total, startPad, progress);
}

template <class T>
__aicore__ inline void KernelND2NZ<T>::ComputeBMM(uint64_t progress) {
    if (noPadD_) {
        CopyIn(progress, outBuf_);
    } else {
        CopyIn(progress, inBuf_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (width_ == 1) { // 内轴为1的场景可以不存在，在tiling侧修改trans属性，将1变成外轴
            PadDOne<false>(progress, outBuf_, inBuf_);
        } else if (wTail_ == 0) { // 内轴32B对齐，大块搬入，再重排，当前实现可能有问题
            PadDAligned<false>(progress, outBuf_, inBuf_);
        } else {
            PadDMain<false>(progress, outBuf_, inBuf_);
        }
    }
    PipeBarrier<PIPE_ALL>();

    uint64_t gmOutOffset =
        (hBuffer_ * progress) / height_ * hAligned_ * widthBlockTotal_ * c0_ + ((hBuffer_ * progress) % height_) * c0_;
    uint32_t total = (progress == nFullProgress_) ? heightTotalTail_ : hBuffer_;
    uint32_t startPad = hAligned_ - (progress * hBuffer_) % height_;
    uint32_t mPage = (hBuffer_ * width_) / c0_ / ALIGNED_H * ALIGNED_H;

    CopyOutBatchReform(gmOutOffset, mPage, total, startPad, progress);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

#if defined(__DAV_C220_VEC__)
template <class T>
__aicore__ inline bool Nd2nzVnchw(GlobalTensor<T>& dst, GlobalTensor<T>& src, uint32_t height, uint32_t width,
                                  uint32_t batch, TBuf<TPosition::VECCALC>& ubBuffer, uint32_t usedCoreNum) {
    KernelND2NZ<T> op;
    op.Init((GM_ADDR)dst[0].GetPhyAddr(), (GM_ADDR)src[0].GetPhyAddr(), height, width, batch, ubBuffer, usedCoreNum);
    if (batch == 1) {
        return op.ProcessMM();
    } else if (batch > 1) {
        return op.ProcessBMM();
    }
    return false;
}

template <>
__aicore__ inline bool Nd2nzVnchw(GlobalTensor<bfloat16_t>& dst, GlobalTensor<bfloat16_t>& src, uint32_t height, uint32_t width,
                                  uint32_t batch, TBuf<TPosition::VECCALC>& ubBuffer, uint32_t usedCoreNum)
{
    GlobalTensor<half> dstGlobalTrans;
    GlobalTensor<half> srcGlobalTrans;
    dstGlobalTrans.SetGlobalBuffer((__gm__ half*)dst.GetPhyAddr(0));
    srcGlobalTrans.SetGlobalBuffer((__gm__ half*)src.GetPhyAddr(0));
    return Nd2nzVnchw(dstGlobalTrans, srcGlobalTrans, height, width, batch, ubBuffer, usedCoreNum);
}
#endif
}

#endif