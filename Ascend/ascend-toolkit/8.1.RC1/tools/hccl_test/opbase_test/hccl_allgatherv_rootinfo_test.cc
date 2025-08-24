#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <hccl/hccl_types.h>
#include "hccl_allgatherv_rootinfo_test.h"
#include "hccl_opbase_rootinfo_base.h"
#include "hccl_check_buf_init.h"

hccl::HcclTest* init_opbase_ptr(hccl::HcclTest* opbase)
{
    opbase = new hccl::HcclOpBaseAllgatherVTest();
    opbase->need_ranksize_alignment = true;

    return opbase;
}

void delete_opbase_ptr(hccl::HcclTest* opbase)
{
    delete opbase;
    opbase = nullptr;
    return;
}

namespace hccl
{
HcclOpBaseAllgatherVTest::HcclOpBaseAllgatherVTest() : HcclOpBaseTest()
{
    
    host_buf = nullptr;
    recv_buff_temp = nullptr;
    check_buf = nullptr;
    send_buff = nullptr;
    recv_buff = nullptr;
    recv_counts = nullptr;
    recv_disp = nullptr;
}

HcclOpBaseAllgatherVTest::~HcclOpBaseAllgatherVTest()
{
}

void HcclOpBaseAllgatherVTest::malloc_recv_buf()
{
    recv_counts = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    recv_disp = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    for(int i = 0; i < rank_size; ++i)
    {
        recv_counts[i] = data->count;
        recv_disp[i] = i * recv_counts[i];
    }
    return;
}

void HcclOpBaseAllgatherVTest::free_recv_buf()
{
    free(recv_counts);
    free(recv_disp);
}

int HcclOpBaseAllgatherVTest::init_buf_val()
{
    //初始化校验内存
    ACLCHECK(aclrtMallocHost((void**)&check_buf, malloc_kSize * rank_size));
    for(int i=0; i < rank_size; ++i)
    {
        hccl_host_buf_init((char*)check_buf + data->count * data->type_size * i, data->count, dtype, i+1);
    }

    return 0;
}

int HcclOpBaseAllgatherVTest::check_buf_result()
{
    //获取输出内存
    ACLCHECK(aclrtMallocHost((void**)&recv_buff_temp, malloc_kSize * rank_size));
    ACLCHECK(aclrtMemcpy((void*)recv_buff_temp, malloc_kSize * rank_size, (void*)recv_buff, malloc_kSize * rank_size, ACL_MEMCPY_DEVICE_TO_HOST));
    int ret = 0;
    switch(dtype)
    {
        case HCCL_DATA_TYPE_FP32:
            ret = check_buf_result_float((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
            ret = check_buf_result_int8((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
            ret = check_buf_result_int32((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_BFP16:
            ret = check_buf_result_half((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_FP64:
            ret = check_buf_result_int64((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        case HCCL_DATA_TYPE_UINT64:
            ret = check_buf_result_u64((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        default:
            ret++;
            printf("no match datatype\n");
            break;
    }
    if(ret != 0)
    {
        check_err++;
    }
    return 0;
}

void HcclOpBaseAllgatherVTest::cal_execution_time(float time)
{
    double total_time_us              = time * 1000;
    double average_time_us            = total_time_us / iters;
    double algorithm_bandwith_GBytes_s = malloc_kSize * rank_size / average_time_us * B_US_TO_GB_S;

    print_execution_time(average_time_us, algorithm_bandwith_GBytes_s);
    return;
}

size_t HcclOpBaseAllgatherVTest::init_malloc_Ksize_by_data()
{
    data->count = (data->count + rank_size - 1) / rank_size;
    return  data->count * data->type_size;
}
void HcclOpBaseAllgatherVTest::init_send_recv_size_by_data(size_t &send_bytes, size_t &recv_bytes)
{
    send_bytes = malloc_kSize;
    recv_bytes = malloc_kSize * rank_size;
}
int HcclOpBaseAllgatherVTest::hccl_op_base_test()  // 主函数
{
    if (op_flag != 0 && rank_id == root_rank) {
        printf("Warning: The -o,--op <sum/prod/min/max> option does not take effect. Check the cmd parameter.\n");
    }

    //申请recv_counts和recv_disp
    malloc_recv_buf();

    is_initdata_overflow();
    //初始化输入内存
    ACLCHECK(aclrtMallocHost((void**)&host_buf, malloc_kSize));
    hccl_host_buf_init((char*)host_buf, data->count, dtype, rank_id+1);
    ACLCHECK(aclrtMemcpy((void*)send_buff, malloc_kSize, (void*)host_buf, malloc_kSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 准备校验内存
    if (check == 1) {
        ACLCHECK(init_buf_val());
    }

    //执行集合通信操作
    for(int j = 0; j < warmup_iters; ++j) {
        HCCLCHECK(HcclAllGatherV((void *)send_buff, data->count, (void*)recv_buff, recv_counts, recv_disp, (HcclDataType)dtype, hccl_comm, stream));
    }

    ACLCHECK(aclrtRecordEvent(start_event, stream));

    for(int i = 0; i < iters; ++i) {
        HCCLCHECK(HcclAllGatherV((void *)send_buff, data->count, (void*)recv_buff, recv_counts, recv_disp, (HcclDataType)dtype, hccl_comm, stream));
    }
    //等待stream中集合通信任务执行完成
    ACLCHECK(aclrtRecordEvent(end_event, stream));

    ACLCHECK(aclrtSynchronizeStream(stream));

    float time;
    ACLCHECK(aclrtEventElapsedTime(&time, start_event, end_event));

    if (check == 1) {
        ACLCHECK(check_buf_result()); // 校验计算结果
    }

    cal_execution_time(time);
    free_recv_buf();
    return 0;
}
}