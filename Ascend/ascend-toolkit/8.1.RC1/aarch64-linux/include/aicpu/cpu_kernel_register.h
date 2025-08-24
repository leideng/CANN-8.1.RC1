/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of kernel register
 */

#ifndef CPU_KERNEL_REGISTAR_H
#define CPU_KERNEL_REGISTAR_H

#include <map>
#include <string>

#include "cpu_kernel.h"
#include "cpu_context.h"

namespace aicpu {
class AICPU_VISIBILITY CpuKernelRegister {
public:
    /*
     * get instance.
     * @return CpuKernelRegister &: CpuKernelRegister instance
     */
    static CpuKernelRegister &Instance();

    /*
     * get cpu kernel.
     * param opType: the op type of kernel
     * @return shared_ptr<CpuKernel>: cpu kernel ptr
     */
    std::shared_ptr<CpuKernel> GetCpuKernel(const std::string &opType);

    /*
     * get all cpu kernel registered op types.
     * @return std::vector<string>: all cpu kernel registered op type
     */
    std::vector<std::string> GetAllRegisteredOpTypes() const;

    /*
     * run cpu kernel.
     * param ctx: context of kernel
     * @return uint32_t: 0->success other->failed
     */
    uint32_t RunCpuKernel(CpuKernelContext &ctx);

    /*
     * run cpu kernel async
     * @param ctx: context of kernel
     * @param wait_type : event notify type
     * @param wait_id : event notify id
     * @param cb : callback function
     * @return uint32_t: 0->success other->failed
    */
    uint32_t RunCpuKernelAsync(CpuKernelContext &ctx,
                               const uint8_t wait_type,
                               const uint32_t wait_id,
                               std::function<uint32_t()> cb);

    // CpuKernel registration function to register different types of kernel to the factory
    class Registerar {
    public:
        Registerar(const std::string &type, const KERNEL_CREATOR_FUN &fun);
        ~Registerar() = default;

        Registerar(const Registerar &) = delete;
        Registerar(Registerar &&) = delete;
        Registerar &operator=(const Registerar &) = delete;
        Registerar &operator=(Registerar &&) = delete;
    };

protected:
    CpuKernelRegister() = default;
    ~CpuKernelRegister() = default;

    CpuKernelRegister(const CpuKernelRegister &) = delete;
    CpuKernelRegister(CpuKernelRegister &&) = delete;
    CpuKernelRegister &operator=(const CpuKernelRegister &) = delete;
    CpuKernelRegister &operator=(CpuKernelRegister &&) = delete;

    // register creator, this function will call in the constructor
    void Register(const std::string &type, const KERNEL_CREATOR_FUN &fun);

private:
    std::map<std::string, KERNEL_CREATOR_FUN> creatorMap_; // kernel map
};
}
#endif // CPU_KERNEL_REGISTAR_H
