/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file kernel_operator_reg_others_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_REG_OTHERS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_REG_OTHERS_IMPL_H

#ifndef ASCENDC_CPU_DEBUG
namespace AscendC {
template <typename T> __aicore__ inline void HsetFlag(pipe_t pipe, pipe_t tpipe, T eventID, mem_t memory, bool v)
{
    hset_flag(pipe, tpipe, eventID, memory, v);
}

template <typename T> __aicore__ inline void HwaitFlag(pipe_t pipe, pipe_t tpipe, T eventID, mem_t memory, bool v)
{
    hwait_flag(pipe, tpipe, eventID, memory, v);
}

__aicore__ inline int64_t GetThreadDim()
{
    return get_thread_dim();
}

__aicore__ inline int64_t GetThreadId()
{
    return get_thread_id();
}

__aicore__ inline int64_t GetSubBlockDim()
{
    return get_subblockdim();
}

__aicore__ inline int64_t GetSubBlockId()
{
    return get_subblockid();
}

__aicore__ inline int64_t GetStackPhyBase()
{
    return get_stack_phy_base();
}

__aicore__ inline int64_t GetStAtomicCfg()
{
    return get_st_atomic_cfg();
}

__aicore__ inline int64_t GetRsvdCnt()
{
    return get_rsvd_cnt();
}

__aicore__ inline int64_t GetMaxMinCnt()
{
    return get_max_min_cnt();
}

__aicore__ inline int64_t GetIcachePrlSt()
{
    return get_icache_prl_st();
}

__aicore__ inline int64_t GetFpc()
{
    return get_fpc();
}

__aicore__ inline void GetCmpmask(__ubuf__ void *dst)
{
    get_cmpmask(dst);
}

template <typename T> __aicore__ inline void DcPreload(__gm__ uint64_t *address, T offset)
{
    dc_preload(address, offset);
}

template <typename T> __aicore__ inline uint64_t ld_dev(T *src, int16_t offset)
{
    return ld_dev(src, offset);
}

__aicore__ inline void SetAippSpr0(uint64_t config)
{
    set_aipp_spr_0(config);
}

__aicore__ inline void SetAippSpr1(uint64_t config)
{
    set_aipp_spr_1(config);
}

__aicore__ inline void SetAippSpr10(uint64_t config)
{
    set_aipp_spr_10(config);
}

__aicore__ inline void SetAippSpr11(uint64_t config)
{
    set_aipp_spr_11(config);
}

__aicore__ inline void SetAippSpr12(uint64_t config)
{
    set_aipp_spr_12(config);
}

__aicore__ inline void SetAippSpr13(uint64_t config)
{
    set_aipp_spr_13(config);
}

__aicore__ inline void SetAippSpr14(uint64_t config)
{
    set_aipp_spr_14(config);
}

__aicore__ inline void SetAippSpr15(uint64_t config)
{
    set_aipp_spr_15(config);
}

__aicore__ inline void SetAippSpr16(uint64_t config)
{
    set_aipp_spr_16(config);
}

__aicore__ inline void SetAippSpr17(uint64_t config)
{
    set_aipp_spr_17(config);
}

__aicore__ inline void SetAippSpr18(uint64_t config)
{
    set_aipp_spr_18(config);
}

__aicore__ inline void SetAippSpr19(uint64_t config)
{
    set_aipp_spr_19(config);
}

__aicore__ inline void SetAippSpr2(uint64_t config)
{
    set_aipp_spr_2(config);
}

__aicore__ inline void SetAippSpr20(uint64_t config)
{
    set_aipp_spr_20(config);
}

__aicore__ inline void SetAippSpr21(uint64_t config)
{
    set_aipp_spr_21(config);
}

__aicore__ inline void SetAippSpr22(uint64_t config)
{
    set_aipp_spr_22(config);
}

__aicore__ inline void SetAippSpr23(uint64_t config)
{
    set_aipp_spr_23(config);
}

__aicore__ inline void SetAippSpr24(uint64_t config)
{
    set_aipp_spr_24(config);
}

__aicore__ inline void SetAippSpr3(uint64_t config)
{
    set_aipp_spr_3(config);
}

__aicore__ inline void SetAippSpr4(uint64_t config)
{
    set_aipp_spr_4(config);
}

__aicore__ inline void SetAippSpr5(uint64_t config)
{
    set_aipp_spr_5(config);
}

__aicore__ inline void SetAippSpr6(uint64_t config)
{
    set_aipp_spr_6(config);
}

__aicore__ inline void SetAippSpr7(uint64_t config)
{
    set_aipp_spr_7(config);
}

__aicore__ inline void SetAippSpr8(uint64_t config)
{
    set_aipp_spr_8(config);
}

__aicore__ inline void SetAippSpr9(uint64_t config)
{
    set_aipp_spr_9(config);
}

__aicore__ inline void SetDataExp0(uint64_t config)
{
    set_data_exp_0(config);
}

template <typename T> __aicore__ inline void SetDeqscale(T config)
{
    set_deqscale(config);
}

__aicore__ inline void SetFftsBaseAddr(uint64_t config)
{
    set_ffts_base_addr(config);
}

__aicore__ inline int64_t GetFftsBaseAddr()
{
    return get_ffts_base_addr();
}

__aicore__ inline void SetMovPadVal(uint64_t config)
{
    set_mov_pad_val(config);
}

__aicore__ inline void SetPcieRdCtrl(uint64_t config)
{
    set_pcie_rd_ctrl(config);
}

template <typename T> __aicore__ inline void StDev(T src, __gm__ T *dst, int16_t offset)
{
    st_dev(src, dst, offset);
}

__aicore__ inline void WaitFlagDev(int64_t flagID)
{
    wait_flag_dev(flagID);
}

__aicore__ inline void FftsCrossCoreSync(pipe_t pipe, uint64_t config)
{
    ffts_cross_core_sync(pipe, config);
}

__aicore__ inline int64_t Clz(uint64_t in)
{
    return clz(in);
}

__aicore__ inline uint64_t FakeOverFlowStatus1()
{
    return fake_overflow_status_1();
}

__aicore__ inline int64_t GetArchVer()
{
    return get_arch_ver();
}

__aicore__ inline int64_t GetConditionFlag()
{
    return get_condition_flag();
}

__aicore__ inline int64_t GetCoreid()
{
    return get_coreid();
}

__aicore__ inline int64_t GetCtrl()
{
    return get_ctrl();
}

__aicore__ inline uint64_t GetImm(uint64_t imm0_15)
{
    return get_imm(imm0_15);
}

__aicore__ inline uint64_t GetImm(uint64_t imm0_15, uint64_t imm16_31)
{
    return get_imm(imm0_15, imm16_31);
}

__aicore__ inline uint64_t GetImm(uint64_t imm0_15, uint64_t imm16_31, uint64_t imm32_47)
{
    return get_imm(imm0_15, imm16_31, imm32_47);
}

__aicore__ inline uint64_t GetImm(uint64_t imm0_15, uint64_t imm16_31, uint64_t imm32_47, uint64_t imm48_63)
{
    return get_imm(imm0_15, imm16_31, imm32_47, imm48_63);
}

__aicore__ inline int64_t GetL2InMain()
{
    return get_l2_in_main();
}

__aicore__ inline int64_t GetL2VAddrBase()
{
    return get_l2_vaddr_base();
}

__aicore__ inline int64_t GetLpcnt()
{
    return get_lpcnt();
}

__aicore__ inline uint64_t GetOverflowStatus()
{
    return get_overflow_status();
}

__aicore__ inline int64_t GetParaBase()
{
    return get_para_base();
}

__aicore__ inline int64_t GetStatus()
{
    return get_status();
}

__aicore__ inline uint64_t SbitSet0(uint64_t x, int64_t idx)
{
    return sbitset0(x, idx);
}

__aicore__ inline void SetConditionFlag(uint64_t config)
{
    set_condition_flag(config);
}

__aicore__ inline void Set_ctrl(uint64_t config)
{
    set_ctrl(config);
}

template <typename T> __aicore__ inline void SetFlag(pipe_t pipe, pipe_t tpipe, T pipeID)
{
    set_flag(pipe, tpipe, pipeID);
}

template <typename T> __aicore__ inline void WaitFlag(pipe_t pipe, pipe_t tpipe, T pipeID)
{
    wait_flag(pipe, tpipe, pipeID);
}

__aicore__ inline void SetLpcnt(uint64_t config)
{
    set_lpcnt(config);
}

__aicore__ inline int64_t Sff0(uint64_t in)
{
    return sff0(in);
}

__aicore__ inline void VldVaReg(ub_addr8_t dst, __ubuf__ uint64_t *src, vpart_t config)
{
    vld_va_reg(dst, src, config);
}
} // namespace AscendC
#endif
#endif // ASCENDC_MODULE_OPERATOR_REG_OTHERS_IMPL_H