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
 * \file kernel_print_lock.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_PRINT_LOCK_H
#define ASCENDC_KERNEL_PRINT_LOCK_H
#include <pthread.h>

namespace AscendC {
class KernelPrintLock {
public:
    int Lock();
    int Unlock();
    void UnInit();
    static void FreeLock();
    static KernelPrintLock* CreateLock();
    static KernelPrintLock* GetLock()
    {
        if (printLock == nullptr) {
            printLock = CreateLock();
        }
        return printLock;
    }
private:
    KernelPrintLock();
    ~KernelPrintLock();

private:
    pthread_rwlock_t lock;
    pthread_rwlockattr_t attr;
    inline void Init();
    static KernelPrintLock* printLock;
};
}

#endif // ASCENDC_KERNEL_PRINT_LOCK_H
