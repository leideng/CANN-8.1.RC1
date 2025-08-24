/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024.
 * Description: a high performance way to acquire cpu and memory stat
 * Author: Pang LiYuan
 * Create: 2024-04-07
 */
#ifndef __LINUX_RTOS_HPSTAT_SHARED_H
#define __LINUX_RTOS_HPSTAT_SHARED_H

#include <rtos_hpstat_offsets.h>
#include <linux/types.h>

struct rtos_hpstat_shared_info {
#if RTOS_HP_OBTAIN_STAT_MEM_IS_ENABLED == 1
	_Atomic(long) vm_zone_stat[ITEMS_NR_VM_ZONE_STAT] __attribute__((__aligned__(ASM_SMP_CACHE_BYTES)));
	_Atomic(long) vm_numa_stat[ITEMS_NR_VM_NUMA_STAT] __attribute__((__aligned__(ASM_SMP_CACHE_BYTES)));
	_Atomic(long) vm_node_stat[ITEMS_NR_VM_NODE_STAT] __attribute__((__aligned__(ASM_SMP_CACHE_BYTES)));
	_Atomic(unsigned long) rtos_totalreserve_pages;
#endif
#if RTOS_HP_OBTAIN_STAT_CPU_IS_ENABLED == 1
	_Atomic(__u64) cpu_idle_timeset[ITEMS_CPU_IDLE_TIMESET];
#endif
};

#endif /* __LINUX_RTOS_HPSTAT_SHARED_H */
