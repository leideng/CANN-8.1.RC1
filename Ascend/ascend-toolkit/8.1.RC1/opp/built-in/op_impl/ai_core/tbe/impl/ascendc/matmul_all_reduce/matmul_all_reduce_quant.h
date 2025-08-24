/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file matmul_all_reduce_quant.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_QUANT_H
#define MATMUL_ALL_REDUCE_QUANT_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "common.h"
#ifdef MC2_QUANT_BF16
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_bf16.h"
#else
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3.h"
#endif
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_pertoken.h"
#include "matmul_all_reduce_base.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename xType, typename wType, typename yType, class mmType, Mc2CoreType coreType, bool pertokenFlag>
class MatmulAllReduceQuantBF16: public MatmulAllReduceBase<xType, yType, coreType> {
public:
    __aicore__ inline MatmulAllReduceQuantBF16(
            MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs, ArnGmAddrs *arnAddrs, MC2TilingHeader *tilingData,
            TPipe *tPipe): MatmulAllReduceBase<xType, yType, coreType>(addrs, quantAddrs, arnAddrs, tilingData, tPipe) {
        mc2TilingData_ = (QuantMatmulAllReduceTilingData *)tilingData;
        this->tileInfo_.mmTiling = &mc2TilingData_->tilematmulTiling.matmulTiling;
        this->tailInfo_.mmTiling = &mc2TilingData_->tailmatmulTiling.matmulTiling;
    }

    __aicore__ inline void Process(mmType &opTile, mmType &opTail) {
        InnerProcess(opTile, false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            InnerProcess(opTail, true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }

        this->HcclFinalize();
    }

protected:
    __aicore__ inline void InnerProcess(mmType &mmOp, bool tailFlag, uint32_t tileCnt, const MC2TileInfo &tileInfo) {
        const QuantBatchMatmulV3TilingData *tiling = (tailFlag ?
                &mc2TilingData_->tailmatmulTiling : &mc2TilingData_->tilematmulTiling);
        const uint64_t pertokenOffset = sizeof(float) * tiling->matmulTiling.M;
        for (uint32_t i = 0U; i < tileCnt; ++i) {
            if (this->addFlag_ || i == 0U) {
                this->tPipe_->Reset();
                if constexpr (pertokenFlag) {
                    mmOp.Init(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM, this->quantAddrs_->dequantGM,
                              this->quantAddrs_->pertokenGM, this->addrs_->cGM, this->addrs_->workspaceGM, tiling,
                              this->tPipe_);
                } else {
                    mmOp.Init(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM, this->quantAddrs_->dequantGM,
                              this->addrs_->cGM, this->addrs_->workspaceGM, tiling, this->tPipe_);
                }
            } else {
                if constexpr (pertokenFlag) {
                    mmOp.UpdateGlobalAddr(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM,
                                          this->quantAddrs_->dequantGM, this->quantAddrs_->pertokenGM,
                                          this->addrs_->cGM, this->addrs_->workspaceGM);
                } else {
                    mmOp.UpdateGlobalAddr(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM,
                                          this->quantAddrs_->dequantGM, this->addrs_->cGM, this->addrs_->workspaceGM);
                }
            }
            mmOp.Process();
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset);
            this->quantAddrs_->pertokenGM += pertokenOffset;
        }
    }

private:
    QuantMatmulAllReduceTilingData *mc2TilingData_;
};

#define REG_MM_OBJ(opTile, opTail)                                  \
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), opTile.mm,      \
        &(tilingData.tilematmulTiling.matmulTiling),                \
        opTail.mm, &(tilingData.tailmatmulTiling.matmulTiling))

#define REG_NO_MM_OBJ(opTile, opTail)

#define INVOKE_MC2_QUANT_910_OP_IMPL(templateClass, coreType, regObjCb, pertokenFlag, ...)                  \
    do {                                                                                                    \
        GET_TILING_DATA_WITH_STRUCT(QuantMatmulAllReduceTilingData, tilingData, tilingGM);                  \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM};                                \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, dequantGM, pertokenGM};                                \
        using opType = templateClass<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, __VA_ARGS__>;                           \
        opType opTile, opTail;                                                                              \
        regObjCb(opTile, opTail);                                                                           \
        MatmulAllReduceQuantBF16<DTYPE_X1, DTYPE_X2, DTYPE_Y, opType, coreType, pertokenFlag> op(           \
                &addrs, &quantAddrs, nullptr, (MC2TilingHeader *)&tilingData, &tPipe);                      \
        op.Init();                                                                                          \
        op.Process(opTile, opTail);                                                                         \
    } while(0)
}  // namespace MatmulAllReduceImpl
#endif  // MATMUL_ALL_REDUCE_QUANT_H