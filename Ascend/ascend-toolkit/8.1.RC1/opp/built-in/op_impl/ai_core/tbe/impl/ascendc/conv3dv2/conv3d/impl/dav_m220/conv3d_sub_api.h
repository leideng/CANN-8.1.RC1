/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
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
 * \file conv3d_sub_api.h
 * \brief
 */

#ifndef CONV3D_SUB_API_H
#define CONV3D_SUB_API_H

#include "conv3d_mte1_sub_api.h"

using namespace AscendC;
using namespace conv;
using namespace conv3d;

namespace Conv3dFunc {

template <class Intf, typename DataTypeT>
class LoadChannelWiseL1Tools {
public:
    __aicore__ inline LoadChannelWiseL1Tools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void SetN(uint64_t n)
    {
        currentNL0_ = n;
    }

    __aicore__ inline void LoadChannelWiseL1(
        const LocalTensor<DataTypeT> &tensorL1, const GlobalTensor<DataTypeT> &tensorGm)
    {
        PreProcess();
        uint64_t srcDValue =
            self_->ctx.biasFullLoadFlag ? AlignB(self_->ctx.singleCoreCo, BLOCK_L0_N) : AlignB(currentNL0_, BLOCK_L0_N);
        if (currentNL0_ != srcDValue) {
            // 非16对齐场景补0, 尾块
            InitConstValueParams<DataTypeT> initConstValueParams;
            initConstValueParams.repeatTimes = 1;
            initConstValueParams.blockNum = srcDValue * sizeof(DataTypeT) / AL1_BLOCK_SIZE;
            initConstValueParams.dstGap = 0;
            initConstValueParams.initValue = 0;
            InitConstValue<DataTypeT>(tensorL1, initConstValueParams);
            ASC_OP_LOGD("[LoadChannelWiseL1] initConstValueParams.blockNum %d.\n", initConstValueParams.blockNum);
        }
        if (currentNL0_ <= MAX_UINT16) {
            SetNd2NzParams(currentNL0_, srcDValue);
            DataCopy<DataTypeT>(tensorL1, tensorGm[tensorGmOffset], nd2NzParams);
        } else {
            biasBlockNum = ONE_BLK_SIZE / self_->ctx.sizeOfBias;
            limitLen = (MAX_UINT16 / biasBlockNum) * biasBlockNum;
            uint16_t num = currentNL0_ / limitLen;

            ASC_OP_LOGD("[LoadChannelWiseL1] num %d limitLen %d tensorGmOffset %d.\n", num, limitLen, tensorGmOffset);
            SetNd2NzParams(limitLen, limitLen);
            nd2NzParams.ndNum = num;
            nd2NzParams.srcNdMatrixStride = limitLen;
            nd2NzParams.dstNzMatrixStride = limitLen;
            DataCopy<DataTypeT>(tensorL1, tensorGm[tensorGmOffset], nd2NzParams);

            uint16_t tail = currentNL0_ % limitLen;
            if (tail) {
                uint64_t offset = num * limitLen;
                tensorGmOffset += offset;
                ASC_OP_LOGD("[LoadChannelWiseL1] offset %d tensorGmOffset %d tail %d.\n",
                            offset, tensorGmOffset, tail);
                SetNd2NzParams(tail, tail);
                DataCopy<DataTypeT>(tensorL1[offset], tensorGm[tensorGmOffset], nd2NzParams);
            }
        }
    }

private:
    __aicore__ inline void PreProcess()
    {
        tensorGmOffset =
            self_->ctx.nBL1Iter * self_->ctx.conv3dTiling->nBL1 + self_->ctx.nBL0Iter * self_->ctx.conv3dTiling->nL0;
        currentNL0_ = self_->ctx.biasFullLoadFlag ? self_->ctx.singleCoreCo : currentNL0_;
        ASC_OP_LOGD("tensorGmOffset %d currentNL0_ %d \n", tensorGmOffset, currentNL0_);
    }

    __aicore__ inline void SetNd2NzParams(uint64_t dValue, uint64_t srcDValue)
    {
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = 1;
        nd2NzParams.dValue = dValue;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = srcDValue;
        nd2NzParams.dstNzC0Stride = 1;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 1;
        ASC_OP_LOGD("[LoadChannelWiseL1] nd2NzParams.dValue %d nd2NzParams.srcDValue %d.\n",
            nd2NzParams.dValue,
            nd2NzParams.srcDValue);
    }

private:
    Intf *self_ = nullptr;
    uint64_t tensorGmOffset;
    uint64_t currentNL0_;
    uint64_t biasBlockNum = 0;
    uint64_t limitLen = 0;
    Nd2NzParams nd2NzParams;
};

template <class Intf>
class LoadBiasBtTools {
public:
    __aicore__ inline LoadBiasBtTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        tilingNBL1_ = self_->ctx.conv3dTiling->nBL1;
    }

    __aicore__ inline void SetN(uint64_t n)
    {
        currentNL0_ = n;
    }

    __aicore__ inline void LoadBiasBt()
    {
        uint32_t offset = 0;
        if (self_->ctx.conv3dTiling->biasFullLoadFlag) {
            offset = self_->ctx.nBL1Iter * tilingNBL1_ + self_->ctx.nBL0Iter * self_->ctx.conv3dTiling->nL0;
        }
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = CeilDIV(currentNL0_ * sizeof(typename Intf::BiasT), BT_BLOCK_SIZE);
        DataCopy<typename Intf::L0cT, typename Intf::BiasT>(
            self_->ctx.biasBT, self_->ctx.biasL1[offset], dataCopyParams);
    }

private:
    Intf *self_ = nullptr;
    uint64_t tilingNBL1_;
    uint64_t currentNL0_ = 0;
};

template <class Intf>
class MMadTools {
public:
    __aicore__ inline MMadTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void SetMN(uint64_t m, uint64_t n)
    {
        currentML0_ = m;
        currentNL0_ = n;
    }

    __aicore__ inline void Mad()
    {
        MmadParams mmadParams;
        mmadParams.m = currentML0_;
        mmadParams.n = currentNL0_;
        if (self_->ctx.kIter != self_->ctx.maxKL0Iter) {
            mmadParams.k = self_->ctx.singleCoreKL0;
        } else {
            mmadParams.k = self_->ctx.kL0Tail;
        }

        if (!self_->ctx.enableBias) {
            if (self_->ctx.kIter == 0) {
                mmadParams.cmatrixInitVal = true;
                mmadParams.cmatrixSource = false;
                mmadParams.isBias = false;
            } else {
                mmadParams.cmatrixInitVal = false;
                mmadParams.cmatrixSource = false;
                mmadParams.isBias = false;
            }
        } else {
            if (self_->ctx.kIter == 0) {
                mmadParams.cmatrixInitVal = false;
                mmadParams.cmatrixSource = true;
                mmadParams.isBias = true;
            } else {
                mmadParams.cmatrixInitVal = false;
                mmadParams.cmatrixSource = false;
                mmadParams.isBias = false;
            }
        }
        ASC_OP_LOGD(
            "[Mad] mmadParams.cmatrixInitVal %d, mmadParams.cmatrixSource %d, mmadParams.isBias %d, mmadParams.k %d, "
            "mmadParams.n %d, mmadParams.m %d.\n",
            mmadParams.cmatrixInitVal,
            mmadParams.cmatrixSource,
            mmadParams.isBias,
            mmadParams.k,
            mmadParams.n,
            mmadParams.m);
        Mmad<typename Intf::L0cT, typename Intf::FmapT, typename Intf::WeightT>(
            self_->ctx.cl0, self_->ctx.al0, self_->ctx.bl0, mmadParams);
    }

private:
    Intf *self_ = nullptr;
    uint64_t currentML0_ = 0;
    uint64_t currentNL0_ = 0;
};

template <class Intf>
class CopyOutTools {
public:
    __aicore__ inline CopyOutTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        tilingNBL1_ = self_->ctx.conv3dTiling->nBL1;
        tilingMAL1_ = self_->ctx.conv3dTiling->mAL1;
        valueHoWo_ = self_->ctx.orgHo * self_->ctx.orgWo;
    }

    __aicore__ inline void SetMN(uint64_t m, uint64_t n)
    {
        currentML0_ = m;
        currentNL0_ = n;
    }

    __aicore__ inline void SetFixpipeIntriParams(FixpipeParamsV220 &intriParams)
    {
        if (self_->ctx.nBL1Iter == self_->ctx.maxNBL1Iter && self_->ctx.nBL0Iter == self_->ctx.maxNL0Iter) {
            intriParams.nSize = currentNL0_;
        } else {
            intriParams.nSize = self_->ctx.conv3dTiling->nL0;
        }

        if (self_->ctx.mAL1Iter == self_->ctx.maxMAL1Iter && self_->ctx.mAL0Iter == self_->ctx.maxML0Iter) {
            intriParams.mSize = self_->ctx.singleCoreM - self_->ctx.mAL1Iter * self_->ctx.conv3dTiling->mAL1 -
                                self_->ctx.mAL0Iter * self_->ctx.conv3dTiling->mL0;
            intriParams.srcStride = AlignB(self_->ctx.mAL0Tail, BLOCK_L0_M);
        } else {
            intriParams.mSize = self_->ctx.conv3dTiling->mL0;
            intriParams.srcStride = self_->ctx.conv3dTiling->mL0;
        }

        intriParams.dstStride = valueHoWo_;
        intriParams.quantPre = GetQuantPre();
        intriParams.reluEn = false;

        if constexpr (AscendC::IsSameType<typename Intf::L0cT, float>::value &&
                      AscendC::IsSameType<typename Intf::OutputT, float>::value) {
            intriParams.isChannelSplit = true;
        }

        ASC_OP_LOGD("[CopyOut] intriParams.nSize %d, intriParams.mSize %d, intriParams.srcStride %d, "
                    "intriParams.dstStride %d, intriParams.quantPre %d, intriParams.reluEn %d.\n",
            intriParams.nSize,
            intriParams.mSize,
            intriParams.srcStride,
            intriParams.dstStride,
            intriParams.quantPre,
            intriParams.reluEn);
    }

    __aicore__ inline void CopyOut(const GlobalTensor<typename Intf::OutputT> &output)
    {
        FixpipeParamsV220 intriParams;
        SetFixpipeIntriParams(intriParams);
        uint64_t offset = CalcFixpipeOffset();
        ASC_OP_LOGD("[CopyOut] offset %d.\n", offset);
        Fixpipe<typename Intf::OutputT, typename Intf::L0cT, CFG_NZ>(output[offset], self_->ctx.cl0, intriParams);
    }

private:
    __aicore__ inline uint64_t CalcFixpipeOffset()
    {
        uint64_t offsetCout = tilingNBL1_ * self_->ctx.nBL1Iter + self_->ctx.conv3dTiling->nL0 * self_->ctx.nBL0Iter;
        uint64_t offsetM = tilingMAL1_ * self_->ctx.mAL1Iter + self_->ctx.conv3dTiling->mL0 * self_->ctx.mAL0Iter;
        // 当前每次只出一个dout
        uint64_t offsetDout = self_->ctx.dOutIter;
        if constexpr (Intf::groupConvType) {
            return offsetDout * AlignB(self_->ctx.conv3dTiling->orgCo, self_->ctx.cin0) * valueHoWo_ +
                   self_->ctx.groupOptIter * self_->ctx.orgCoAlignK0 * valueHoWo_ + offsetCout * valueHoWo_ +
                   offsetM * self_->ctx.cin0;
        } else {
            return offsetDout * self_->ctx.orgCoAlignK0 * valueHoWo_ + offsetCout * valueHoWo_ +
                   offsetM * self_->ctx.cin0;
        }
    }

    __aicore__ inline QuantMode_t GetQuantPre()
    {
        if constexpr (AscendC::IsSameType<typename Intf::L0cT, float>::value &&
                      AscendC::IsSameType<typename Intf::OutputT, float>::value) {
            return QuantMode_t::NoQuant;
        }

        if constexpr (AscendC::IsSameType<typename Intf::L0cT, int32_t>::value &&
                      AscendC::IsSameType<typename Intf::OutputT, half>::value) {
            return QuantMode_t::VDEQF16;
        }

        if constexpr (AscendC::IsSameType<typename Intf::L0cT, float>::value &&
                      AscendC::IsSameType<typename Intf::OutputT, bfloat16_t>::value) {
            return QuantMode_t::F322BF16;
        }

        return QuantMode_t::F322F16;
    }

private:
    Intf *self_ = nullptr;
    uint64_t tilingNBL1_ = 0;
    uint64_t tilingMAL1_ = 0;
    uint64_t valueHoWo_ = 0;
    uint64_t currentML0_ = 0;
    uint64_t currentNL0_ = 0;
};

};  // namespace Conv3dFunc

#endif  // __CONV3D_SUB_API_H__
