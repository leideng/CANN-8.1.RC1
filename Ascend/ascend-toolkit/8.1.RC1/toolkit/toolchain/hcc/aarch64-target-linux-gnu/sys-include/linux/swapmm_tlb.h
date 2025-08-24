/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Copyright @ Huawei Technologies Co., Ltd. 2020-2020. ALL rights reserved.
 * Description: Header file for swapmm_tlb module.
 */

#ifndef _LINUX_SWAPMM_H
#define _LINUX_SWAPMM_H

#include <linux/ioctl.h>

#define SWAPMM_MMAP_MADV 0x1ead6bb6dead1000
#define SWAPMM_BIT_GET_KADDR 0x01u
#define SWAPMM_BIT_RELIABLE 0x02u
#define SWAPMM_ALIGN_4K (4 * 1024)
#define SWAPMM_ALIGN_2M (2 * 1024 * 1024)

/* struct swapmm_phy_area - user defined physical mapping area.
 */
struct swapmm_phy_area {
	unsigned long virt_addr;
	unsigned long phy_addr;
	unsigned long size;
};

/* struct swapmm_pin_area - used by set the memory area which can
 * be recovered.
 * @offset: swapmm offset, do nothing, can be used as index
 */
struct swapmm_pin_area {
	unsigned long virt_start;
	unsigned long virt_end;
	unsigned long offset;
};

/* struct swapmm_madv_ospages - used by map and unmap the memory with os pages
 * @virt_addr: users guarantee the addr is available, must be align with pagealign
 * @len: must be an integer multiple of pagealign
 * @pagealign: must be 4K or 2M
 * @flag: reserved to control some behavior, now just one flag named SWAPMM_BIT_GET_KADDR
 *        to get physical addr in buf when do willneed
 * @numa_node: numa of physical memory needed when do willneed action, defautl 0,
 * 	  must be smaller than the MAX NUMA NODES
 * @buflen: the length of buf, must >= (len / pagealign * sizeof(unsigned long)), only needed
 * 	  by willneed action
 * @buf: the buf to get physical addr when flaged with SWAPMM_BIT_GET_KADDR, the alloc/release
 *       is guaranteed by the user, only needed by willneed action
 */
struct swapmm_madv_ospages {
	unsigned long virt_addr;
	unsigned long len;
	unsigned long pagealign;
	unsigned long flag;
	unsigned int numa_node;
	unsigned int buflen;
	void *buf;
};

/* struct swapmm_madv_swapmove - used by move pages from ori virt addr to dest virt addr
 * @vaddr_ori: original vaddr, users guarantee the addr is available, must be align with pagealign
 * @vaddr_dst: destination vaddr, users guarantee the addr is available, must be align with
 * 	  pagealign
 * @len: must be an integer multiple of pagealign
 * @pagealign: must be 4K or 2M
 * @flag: reserved to control some behavior
 */
struct swapmm_madv_swapmove {
	unsigned long vaddr_ori;
	unsigned long vaddr_dst;
	unsigned long len;
	unsigned long pagealign;
	unsigned long flag;
};

/* struct swapmm_mem_mirror_apply - used by memory mirror to add or delete CE address
 * @addr: physical or virtual address; virtual address is for current process
 * @is_virtual: 1 for virtual address, 0 for physical address
 */
struct swapmm_mem_mirror_apply {
	unsigned long addr;
	int is_virtual;
};

/* struct swapmm_mem_mirror_query - used by memory mirror to get mirrored CE address
 * @buflen: the length of buf area
 * @written_count: number of written data, used by memory mirror module
 * @buf: the buf is an area to get returned infomation, memory controlled by user
 */
struct swapmm_mem_mirror_query {
	unsigned int buflen;
	unsigned int written_count;
	void *buf;
};

/* SWAPMM ioctl parameters */
#define SWAPMM_MAGIC 0x55

enum _SWAPMM_IOCTL_CMD {
	_SWAPMM_PHY_MAPPING = 1,
	_SWAPMM_PIN_MAPPING,
	_SWAPMM_RESTORE_MAPPING,
	_SWAPMM_MADV_WILLNEED,
	_SWAPMM_MADV_DONTNEED,
	_SWAPMM_MADV_SWAPMOVE,
	_SWAPMM_MEM_MIRROR_ADD,
	_SWAPMM_MEM_MIRROR_DEL,
	_SWAPMM_MEM_MIRROR_QUERY,
	_SWAPMM_BIND_ASID,
	_SWAPMM_IOC_MAX_NR,
};

#define SWAPMM_PHY_MAPPING		\
	_IOW(SWAPMM_MAGIC, _SWAPMM_PHY_MAPPING, struct swapmm_phy_area)
#define SWAPMM_PIN_MAPPING		\
	_IOW(SWAPMM_MAGIC, _SWAPMM_PIN_MAPPING, struct swapmm_pin_area)
#define SWAPMM_RESTORE_MAPPING	\
	_IOW(SWAPMM_MAGIC, _SWAPMM_RESTORE_MAPPING, struct swapmm_pin_area)
#define SWAPMM_MADV_WILLNEED        \
	_IOW(SWAPMM_MAGIC, _SWAPMM_MADV_WILLNEED, struct swapmm_madv_ospages)
#define SWAPMM_MADV_DONTNEED        \
	_IOW(SWAPMM_MAGIC, _SWAPMM_MADV_DONTNEED, struct swapmm_madv_ospages)
#define SWAPMM_MADV_SWAPMOVE        \
	_IOW(SWAPMM_MAGIC, _SWAPMM_MADV_SWAPMOVE, struct swapmm_madv_swapmove)
#define SWAPMM_MEM_MIRROR_ADD		\
	_IOW(SWAPMM_MAGIC, _SWAPMM_MEM_MIRROR_ADD, struct swapmm_mem_mirror_apply)
#define SWAPMM_MEM_MIRROR_DEL		\
	_IOW(SWAPMM_MAGIC, _SWAPMM_MEM_MIRROR_DEL, struct swapmm_mem_mirror_apply)
#define SWAPMM_MEM_MIRROR_QUERY		\
	_IOW(SWAPMM_MAGIC, _SWAPMM_MEM_MIRROR_QUERY, struct swapmm_mem_mirror_query)
#define SWAPMM_BIND_ASID _IOW(SWAPMM_MAGIC, _SWAPMM_BIND_ASID, unsigned long)

#endif /* _LINUX_SWAPMM_H */
