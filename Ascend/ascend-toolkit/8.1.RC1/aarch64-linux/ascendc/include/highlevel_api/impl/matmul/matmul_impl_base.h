/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_impl_base.h
 * \brief
 */
#ifndef IMPL_MATMUL_MATMUL_IMPL_BASE_H
#define IMPL_MATMUL_MATMUL_IMPL_BASE_H

#include "policy/matmul_policy.h"
#include "policy/matmul_private_modules.h"
#include "utils/matmul_module.h"
#include "utils/matmul_param.h"
#include "utils/matmul_utils.h"

namespace AscendC {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG = CFG_NORM,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImplBase
: MATMUL_IMPORT_MODULE(Context)
, MATMUL_IMPORT_MODULE(CubeOutBuffer)
, MATMUL_IMPORT_MODULE(CopyCubeOut)
, MATMUL_IMPORT_MODULE(CopyCubeInA)
, MATMUL_IMPORT_MODULE(CopyCubeInB)
, MATMUL_IMPORT_MODULE(CubeInBufferA)
, MATMUL_IMPORT_MODULE(CubeInBufferB)
, MATMUL_IMPORT_MODULE(Scheduler)
, MATMUL_IMPORT_MODULE(BiasScheduler)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsA)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsB)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsB)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulQuantProcessor)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoA)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulSubBlockInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
, MATMUL_IMPORT_MODULE_PRIVATE(LocalWorkspace)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulUserDefineInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(LoadToA2)
, MATMUL_IMPORT_MODULE_PRIVATE(LoadToB2)
, MATMUL_IMPORT_MODULE_PRIVATE(TBufPoolL0)
, MATMUL_IMPORT_MODULE_PRIVATE(MmadCompute)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyBiasIn)
, MATMUL_IMPORT_MODULE_PRIVATE(LoadBias2C2)
, MATMUL_IMPORT_MODULE_PRIVATE(C1Buffer)
, MATMUL_IMPORT_MODULE_PRIVATE(C2Buffer)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulUnitFlag)
, MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeOutUtils)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulAntiQuantProcessor)
, MATMUL_IMPORT_MODULE_PRIVATE(PartialOutUtils)
{
public:
    using AType = A_TYPE;
    using BType = B_TYPE;
    using CType = C_TYPE;
    using BiasType = BIAS_TYPE;
private:
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;
    using SrcT = typename A_TYPE::T;
    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline MatmulImplBase() {};
    __aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe = nullptr);
    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK);
    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc = 0);
    __aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK);
    __aicore__ inline void SetTail(int tailM = -1, int tailN = -1, int tailK = -1);
    __aicore__ inline void SetTensorA(const GlobalTensor<SrcAT>& gm, bool isTransposeA = false);
    __aicore__ inline void SetTensorB(const GlobalTensor<SrcBT>& gm, bool isTransposeB = false);
    __aicore__ inline void SetBias(const GlobalTensor<BiasT>& biasGlobal);
    __aicore__ inline void SetSelfDefineData(const uint64_t dataPtr);
    __aicore__ inline void SetSparseIndex(const GlobalTensor<uint8_t>& indexGlobal);
    __aicore__ inline void SetUserDefInfo(const uint64_t tilingPtr);
    __aicore__ inline void SetQuantScalar(const uint64_t quantScalar);
    __aicore__ inline void SetQuantVector(const GlobalTensor<uint64_t>& quantTensor);
    __aicore__ inline void SetTensorA(const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA = false);
    __aicore__ inline void SetTensorAWithCopy(const GlobalTensor<SrcAT>& gm, const LocalTensor<SrcAT>& leftMatrix,
        bool isTransposeA = false);
    __aicore__ inline void SetTensorB(const LocalTensor<SrcBT>& rightMatrix, bool isTransposeB = false);
    __aicore__ inline void SetTensorA(SrcAT aScalar);
    __aicore__ inline void SetTensorB(SrcBT bScalar);
    __aicore__ inline void SetTensorBWithCopy(const GlobalTensor<SrcBT>& gm, const LocalTensor<SrcBT>& rightMatrix,
        bool isTransposeB = false);
    __aicore__ inline void SetBias(const LocalTensor<BiasT>& inputBias);
    __aicore__ inline void DisableBias();
    __aicore__ inline void ClearBias();
    template <bool sync = true> __aicore__ inline bool Iterate(bool enPartialSum = false);

    template <bool sync = true>
    __aicore__ inline void GetTensorC(const LocalTensor<DstT>& co2Local, uint8_t enAtomic = 0, bool enSequentialWrite = false);
    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0, bool enSequentialWrite = false);
    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<DstT> &gm, const LocalTensor<DstT> &co2Local,
        uint8_t enAtomic = 0, bool enSequentialWrite = false);
    template <bool isTurnOnDebug = true>
    __aicore__ inline MatrixOffset GetOffsetC();
    __aicore__ inline void End();
    __aicore__ inline void SetHF32(bool enableHF32 = false, int32_t transMode = 0);
    __aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx);
    __aicore__ inline uint8_t GetSubBlockIdx();
    template <class T> __aicore__ inline void SetWorkspace(__gm__ const T* addr, int size)
    {
        MATMUL_MODULE(PartialOutUtils)->SetWorkspace(addr);
    }

    template <class T> __aicore__ inline void SetWorkspace(GlobalTensor<T>& addr)
    {
        ASSERT(addr.GetSize() > 0);
        SetWorkspace(addr.GetPhyAddr(), addr.GetSize() * sizeof(T));
    }

    __aicore__ inline void SetLocalWorkspace(const LocalTensor<uint8_t>& tmpBuffer)
    {
        MATMUL_MODULE(LocalWorkspace)->Init(tmpBuffer);
    }

public:
    MATMUL_ALLOW_USING(CubeOutBuffer);
    MATMUL_ALLOW_USING(CubeInBufferA);
    MATMUL_ALLOW_USING(CubeInBufferB);
    MATMUL_ALLOW_USING(CopyCubeInA);
    MATMUL_ALLOW_USING(CopyCubeInB);
    MATMUL_ALLOW_USING(CopyCubeOut);
    MATMUL_ALLOW_USING(Context);
    MATMUL_ALLOW_USING(Scheduler);
    MATMUL_ALLOW_USING(BiasScheduler);

    MATMUL_ALLOW_USING_PRIVATE(LoadToA2);
    MATMUL_ALLOW_USING_PRIVATE(LoadToB2);
    MATMUL_ALLOW_USING_PRIVATE(TBufPoolL0);
    MATMUL_ALLOW_USING_PRIVATE(MmadCompute);
    MATMUL_ALLOW_USING_PRIVATE(MatmulQuantProcessor);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsA);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsB);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsA);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsB);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyWrapperA);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyWrapperB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulTensorInfoA);
    MATMUL_ALLOW_USING_PRIVATE(MatmulTensorInfoB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulSubBlockInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(LocalWorkspace);
    MATMUL_ALLOW_USING_PRIVATE(MatmulUserDefineInfo);
    MATMUL_ALLOW_USING_PRIVATE(CopyBiasIn);
    MATMUL_ALLOW_USING_PRIVATE(LoadBias2C2);
    MATMUL_ALLOW_USING_PRIVATE(C1Buffer);
    MATMUL_ALLOW_USING_PRIVATE(C2Buffer);
    MATMUL_ALLOW_USING_PRIVATE(MatmulUnitFlag);
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeOutUtils);
    MATMUL_ALLOW_USING_PRIVATE(MatmulAntiQuantProcessor);
    MATMUL_ALLOW_USING_PRIVATE(PartialOutUtils);

    template <InputTypeTag TAG>
    using CubeInBuffer = typename AscendC::Conditional<TAG == InputTypeTag::A, CubeInBufferA, CubeInBufferB>::type;

    template <InputTypeTag TAG>
    using CopyCubeInParams =
        typename AscendC::Conditional<TAG == InputTypeTag::A, CopyCubeInParamsA, CopyCubeInParamsB>::type;

    template <InputTypeTag TAG>
    using MatmulTensorInfo =
        typename AscendC::Conditional<TAG == InputTypeTag::A, MatmulTensorInfoA, MatmulTensorInfoB>::type;

    template <InputTypeTag TAG>
    using DataCopyUtils = typename AscendC::Conditional<TAG == InputTypeTag::A, DataCopyUtilsA, DataCopyUtilsB>::type;

    template <InputTypeTag TAG>
    using DataCopyWrapper =
        typename AscendC::Conditional<TAG == InputTypeTag::A, DataCopyWrapperA, DataCopyWrapperB>::type;

    using CallBack = MM_CB;

private:
    template <typename, typename> friend struct DfxProxy;
    using IMPL = MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(LocalWorkspace);
    MATMUL_USE_MODULE(PartialOutUtils);

    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(CopyCubeOut);
    MATMUL_USE_MODULE(Scheduler);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(LoadToA2);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulAntiQuantProcessor);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);

private:
    template <class A_TYPE_, class B_TYPE_, class C_TYPE_, class BIAS_TYPE_, const auto &MM_CFG_, class MM_CB_,
        MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY_)>
    friend __aicore__ inline void SetTPipe(
        MatmulImpl<A_TYPE_, B_TYPE_, C_TYPE_, BIAS_TYPE_, MM_CFG_, MM_CB_, MATMUL_POLICY_> &mm, TPipe* tpipe);

protected:
    typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulVersion(MM_CFG)>::PARAMS var;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto &MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void SetTPipe(MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY> &mm,
    TPipe* tpipe)
{
    mm.var.tpipe_ = tpipe;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetSelfDefineData(
    const uint64_t dataPtr)
{
    MATMUL_MODULE(MatmulUserDefineInfo)->SetSelfDefineData(dataPtr);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetSparseIndex(const GlobalTensor<uint8_t>& indexGlobal)
{
#if __CCE_AICORE__ == 220
    if constexpr (DoMatmulMDL(MM_CFG) && HasSparseIndex<B_TYPE>()) {
        MATMUL_MODULE(CopyCubeInB)->SetSparseIndex(indexGlobal);
    }
#endif
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetUserDefInfo(
    const uint64_t tilingPtr)
{
    MATMUL_MODULE(MatmulUserDefineInfo)->SetUserDefineInfo(tilingPtr);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetQuantScalar(
    const uint64_t quantScalar)
{
    MatmulQuantProcessor::SetQuantScalar(quantScalar);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetQuantVector(
    const GlobalTensor<uint64_t>& quantTensor)
{
    MatmulQuantProcessor::SetQuantVector(quantTensor);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto &MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline uint8_t MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::GetSubBlockIdx()
{
    return MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::Init(
    const TCubeTiling* __restrict cubeTiling, TPipe* tpipe)
{
    auto tpipePtr = GetTPipePtr();
    MATMUL_MODULE(Scheduler)->Init(cubeTiling, tpipePtr);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetOrgShape(
    int orgM, int orgN, int orgK)
{
    SetOrgShape(orgM, orgN, orgK, orgK);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetOrgShape(
    int orgM, int orgN, int orgKa, int orgKb, int orgKc)
{
    MATMUL_MODULE(MatmulShapeInfo)->CheckOrgShape(orgM, orgN, orgKa, orgKb, orgKc);
    MATMUL_MODULE(MatmulShapeInfo)->SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetSingleShape(
    int singleM, int singleN, int singleK)
{
    SetTail(singleM, singleN, singleK);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetHF32(bool enableHF32,
    int32_t transMode)
{
    ASCENDC_ASSERT((transMode == 0 || transMode == 1),
                   { KERNEL_LOG(KERNEL_ERROR, "transMode is %d , which should only be 0 / 1", transMode); });
    if (unlikely(enableHF32)) {
        SetHF32Mode(1);
    } else {
        SetHF32Mode(0);
    }
    if (unlikely(transMode == 1)) {
        SetHF32TransMode(1);
    } else {
        SetHF32TransMode(0);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto &MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetSubBlockIdx(uint8_t subBlockIdx)
{
    MATMUL_MODULE(MatmulSubBlockInfo)->SetSubBlockIdx(subBlockIdx);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::End()
{
    MATMUL_MODULE(Scheduler)->End();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTail(
    int tailM, int tailN, int tailK)
{
    auto shapeInfo = MATMUL_MODULE(MatmulShapeInfo);
    shapeInfo->CheckTailShape(tailM, tailN, tailK);
    if constexpr (ToMatmulConfig(MM_CFG).intraBlockPartSum) {
        int singleM = (MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx() == 0) ? shapeInfo->GetSingleCoreM()
                          : shapeInfo->template GetSingleCoreM<true>();
        int singleN = (MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx() == 0) ? shapeInfo->GetSingleCoreN()
                          : shapeInfo->template GetSingleCoreN<true>();
        int singleK = (MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx() == 0) ? shapeInfo->GetSingleCoreK()
                          : shapeInfo->template GetSingleCoreK<true>();
        singleM = (tailM != -1) ? tailM : singleM;
        singleN = (tailN != -1) ? tailN : singleN;
        singleK = (tailK != -1) ? tailK : singleK;
        shapeInfo->SetSingleShape(singleM, singleN, singleK);
        MATMUL_MODULE(MLoop)->SetSingleShape(singleM);
        MATMUL_MODULE(NLoop)->SetSingleShape(singleN);
        MATMUL_MODULE(KLoop)->SetSingleShape(singleK);
    } else {
        if ((tailM != -1) && (tailM != shapeInfo->GetSingleCoreM())) {
            shapeInfo->SetSingleCoreM(tailM);
            MATMUL_MODULE(MLoop)->SetSingleShape(shapeInfo->GetSingleCoreM());
        }
        if ((tailN != -1) && (tailN != shapeInfo->GetSingleCoreN())) {
            shapeInfo->SetSingleCoreN(tailN);
            MATMUL_MODULE(NLoop)->SetSingleShape(shapeInfo->GetSingleCoreN());
        }
        if ((tailK != -1) && (tailK != shapeInfo->GetSingleCoreK())) {
            shapeInfo->SetSingleCoreK(tailK);
            MATMUL_MODULE(KLoop)->SetSingleShape(shapeInfo->GetSingleCoreK());
        }
    }

    shapeInfo->CheckSpecificShape();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorA(
    const GlobalTensor<SrcAT>& gm, bool isTransposeA)
{
    MATMUL_MODULE(CopyCubeInA)->SetInput(gm, isTransposeA);
    MATMUL_MODULE(Scheduler)->Reset();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorA(
    const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA)
{
    MATMUL_MODULE(CopyCubeInA)->SetInput(leftMatrix, isTransposeA);
    MATMUL_MODULE(Scheduler)->Reset();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorA(SrcAT aScalar)
{
    MATMUL_MODULE(LoadToA2)->SetScalar(aScalar);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorB(SrcBT bScalar)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "It is not allowed to set matrix B with scalar."); });
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorAWithCopy(
    const GlobalTensor<SrcAT>& gm, const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA)
{
#if (__CCE_AICORE__ < 220)
    event_t eventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMte3);
    struct DataCopyParams param;
    param.blockLen = leftMatrix.GetSize() / AscendCUtils::GetC0Count(sizeof(SrcT));
    DataCopy(gm, leftMatrix, param);
    SetTensorA(gm, isTransposeA);
#else
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "not supported on Ascend910B1."); });
#endif
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorBWithCopy(
    const GlobalTensor<SrcBT>& gm, const LocalTensor<SrcBT>& rightMatrix, bool isTransposeB)
{
#if (__CCE_AICORE__ < 220)
    event_t eventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMte3);
    struct DataCopyParams param;
    param.blockLen = rightMatrix.GetSize() / AscendCUtils::GetC0Count(sizeof(SrcBT));
    DataCopy(gm, rightMatrix, param);
    SetTensorB(gm, isTransposeB);
#else
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "not supported on Ascend910B1."); });
#endif
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorB(
    const GlobalTensor<SrcBT>& gm, bool isTransposeB)
{
    MATMUL_MODULE(CopyCubeInB)->SetInput(gm, isTransposeB);
    MATMUL_MODULE(Scheduler)->Reset();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorB(
    const LocalTensor<SrcBT>& rightMatrix, bool isTransposeB)
{
    MATMUL_MODULE(CopyCubeInB)->SetInput(rightMatrix, isTransposeB);
    MATMUL_MODULE(Scheduler)->Reset();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
          MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetBias(
    const GlobalTensor<BiasT>& biasGlobal)
{
    MATMUL_MODULE(BiasScheduler)->SetInput(biasGlobal);
    MATMUL_MODULE(BiasScheduler)->SetBias(true);
    MATMUL_MODULE(Scheduler)->Reset();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
          MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetBias(
    const LocalTensor<BiasT>& inputBias)
{
    MATMUL_MODULE(BiasScheduler)->SetInput(inputBias);
    MATMUL_MODULE(BiasScheduler)->SetBias(true);
    MATMUL_MODULE(Scheduler)->Reset();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::DisableBias()
{
    MATMUL_MODULE(BiasScheduler)->SetBias(false);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::ClearBias()
{
    DisableBias();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
template <bool isTurnOnDebug>
__aicore__ inline MatrixOffset MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::GetOffsetC()
{
    if constexpr (isTurnOnDebug) {
        static_assert(!isTurnOnDebug, "unsupported!");
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
template <bool sync>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::GetTensorC(
    const LocalTensor<DstT>& co2Local, uint8_t enAtomic, bool enSequentialWrite)
{
    static_assert(ToMatmulConfig(MM_CFG).scheduleType != ScheduleType::OUTER_PRODUCT, "Unsupported scheduleType");
    MATMUL_MODULE(Scheduler)->GetResult(co2Local, enAtomic, enSequentialWrite);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
template <bool sync>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::GetTensorC(
    const GlobalTensor<DstT>& gm, uint8_t enAtomic, bool enSequentialWrite)
{
    MATMUL_MODULE(Scheduler)->GetResult(gm, enAtomic, enSequentialWrite);
}

#if __CCE_AICORE__ < 220
// v100, v200
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
template <bool sync>
__aicore__ inline void MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::GetTensorC(
    const GlobalTensor<DstT> &gm, const LocalTensor<DstT> &co2Local, uint8_t enAtomic, bool enSequentialWrite)
{
    MATMUL_MODULE(Scheduler)->GetResult(gm, co2Local, enAtomic, enSequentialWrite);
}
#endif

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
template <bool sync>
__aicore__ inline bool MatmulImplBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::Iterate(bool enPartialSum)
{
    return MATMUL_MODULE(Scheduler)->ScheduleOnce(enPartialSum);
}

} // namespace AscendC
#endif // _MATMUL_IMPL_BASE_H_
