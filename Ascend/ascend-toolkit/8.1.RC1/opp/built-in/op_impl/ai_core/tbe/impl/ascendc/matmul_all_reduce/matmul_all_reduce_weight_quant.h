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
 * \file matmul_all_reduce_weight_quant.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_WEIGHT_QUANT_H
#define MATMUL_ALL_REDUCE_WEIGHT_QUANT_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "common.h"
#if (FORMAT_X2 == FORMAT_FRACTAL_NZ)
#include "../weight_quant_batch_matmul_v2/weight_quant_batch_matmul_v2_custom_weight_nz.h"
#define WEIGH_QUANT_MATMUL_CLASS_NAME WeightQuantBatchMatmulV2::WeightQuantBatchMatmulV2CustomWeightNzKernel
#else
#include "../weight_quant_batch_matmul_v2/weight_quant_batch_matmul_v2_custom.h"
#define WEIGH_QUANT_MATMUL_CLASS_NAME WeightQuantBatchMatmulV2::WeightQuantBatchMatmulV2CustomKernel
#endif
#include "matmul_all_reduce_base.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
using WeightQuantBatchMatmulV2::QuantType;
template <typename xType, typename wType, typename yType, class mmType>
class MatmulAllReduceWeightQuant: public MatmulAllReduceBase<xType, yType, Mc2CoreType::ON_CUBE_AND_VECTOR> {
public:
    __aicore__ inline MatmulAllReduceWeightQuant(
            MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs, ArnGmAddrs *arnAddrs, MC2TilingHeader *tilingData,
            TPipe *tPipe): MatmulAllReduceBase<xType, yType, Mc2CoreType::ON_CUBE_AND_VECTOR>(
                    addrs, quantAddrs, arnAddrs, tilingData, tPipe) {
        mc2TilingData_ = (WeightQuantMatmulAllReduceTilingData *)tilingData;
        this->tileInfo_.mmTiling = &mc2TilingData_->tilematmulTiling.matmulTiling;
        this->tailInfo_.mmTiling = &mc2TilingData_->tailmatmulTiling.matmulTiling;
    }

    __aicore__ inline void Process() {
#if (ORIG_DTYPE_X1 == DT_BF16)
        this->PreProcForBiasOnVector();
#endif
        InnerProcess(false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            InnerProcess(true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }

        this->HcclFinalize();
    }

protected:
    __aicore__ inline void InnerProcess(bool tailFlag, uint32_t tileCnt, const MC2TileInfo &tileInfo) {
        const WeightQuantBatchMatmulV2TilingData *tiling = (tailFlag ?
                &mc2TilingData_->tailmatmulTiling : &mc2TilingData_->tilematmulTiling);
        for (uint32_t i = 0U; i < tileCnt; ++i) {
            if (this->addFlag_ || i == 0U) {
                this->tPipe_->Reset();
                mmOp_.Init(this->addrs_->aGM, this->addrs_->bGM, this->quantAddrs_->antiquantScaleGM,
                           this->quantAddrs_->antiquantOffsetGM, nullptr, nullptr, this->addrs_->biasGM,
                           this->addrs_->cGM, this->addrs_->workspaceGM, tiling, this->tPipe_);
            } else {
                mmOp_.UpdateGlobalAddr(this->addrs_->aGM, this->addrs_->bGM, this->quantAddrs_->antiquantScaleGM,
                                       this->quantAddrs_->antiquantOffsetGM, nullptr, nullptr, this->addrs_->biasGM,
                                       this->addrs_->cGM, this->addrs_->workspaceGM);
            }
            mmOp_.Process();
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset);
        }
    }

private:
    WeightQuantMatmulAllReduceTilingData *mc2TilingData_;
    mmType mmOp_;
};

#define INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(bTransFlag, quantType, offsetFlag)                              \
    do {                                                                                                    \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantMatmulAllReduceTilingData, tilingData, tilingGM);            \
        using opType = WEIGH_QUANT_MATMUL_CLASS_NAME<                                                       \
                DTYPE_X1, DTYPE_X2, DTYPE_BIAS_FOR_MC2, DTYPE_Y,                                            \
                false, bTransFlag, quantType, offsetFlag, QuantType::NONE>;                                 \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM};                                \
        QuantGmAddrs quantAddrs = {antiquantScaleGM, antiquantOffsetGM, nullptr, nullptr};                  \
        MatmulAllReduceWeightQuant<DTYPE_X1, DTYPE_X2, DTYPE_Y, opType> op(                                 \
                &addrs, &quantAddrs, nullptr, (MC2TilingHeader *)&tilingData, &tPipe);                      \
        op.Init();                                                                                          \
        op.Process();                                                                                       \
    } while (0)
}  // namespace MatmulAllReduceImpl
#endif  // MATMUL_ALL_REDUCE_WEIGHT_QUANT_H