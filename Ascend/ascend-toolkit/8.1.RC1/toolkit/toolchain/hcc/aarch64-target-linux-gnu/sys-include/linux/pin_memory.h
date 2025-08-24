/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Copyright @ Huawei Technologies Co., Ltd. 2020-2020. ALL rights reserved.
 * Description: Header file for pin memory module.
 */

#ifndef _LINUX_PIN_MEMORY_DEV_H
#define _LINUX_PIN_MEMORY_DEV_H

#include <linux/ioctl.h>

#define MAX_PIN_MEM_AREA_NUM  16

struct _pin_mem_area {
	unsigned long virt_start;
	unsigned long virt_end;
};

struct pin_mem_area_set {
	unsigned int pid;
	unsigned int area_num;
	struct _pin_mem_area mem_area[MAX_PIN_MEM_AREA_NUM];
};

#define PIN_MEM_MAGIC 0x59
#define _SET_PIN_MEM_AREA     1
#define _CLEAR_PIN_MEM_AREA   2
#define _REMAP_PIN_MEM_AREA   3
#define _FINISH_PIN_MEM_DUMP  4
#define _INIT_PAGEMAP_READ    5
#define _DUMP_SPECIAL_PAGES   6
#define _RETORE_SPECIAL_PAGES 7


#define SET_PIN_MEM_AREA        _IOW(PIN_MEM_MAGIC, _SET_PIN_MEM_AREA, struct pin_mem_area_set)
#define CLEAR_PIN_MEM_AREA      _IOW(PIN_MEM_MAGIC, _CLEAR_PIN_MEM_AREA, int)
#define REMAP_PIN_MEM_AREA      _IOW(PIN_MEM_MAGIC, _REMAP_PIN_MEM_AREA, int)
#define FINISH_PIN_MEM_DUMP     _IOW(PIN_MEM_MAGIC, _FINISH_PIN_MEM_DUMP, int)
#define INIT_PAGEMAP_READ       _IOW(PIN_MEM_MAGIC, _INIT_PAGEMAP_READ, int)
#define DUMP_SPECIAL_PAGES      _IOW(PIN_MEM_MAGIC, _DUMP_SPECIAL_PAGES, int)
#define RETORE_SPECIAL_PAGES    _IOW(PIN_MEM_MAGIC, _RETORE_SPECIAL_PAGES, int)


#endif /* _LINUX_PIN_MEMORY_DEV_H */
