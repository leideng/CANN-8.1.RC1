#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <hccl/hccl_types.h>
#include "hccl_reducescatterv_rootinfo_test.h"
#include "hccl_opbase_rootinfo_base.h"
#include "hccl_check_buf_init.h"

hccl::HcclTest* init_opbase_ptr(hccl::HcclTest* opbase)
{
    opbase = new hccl::HcclOpBaseReducescatterVTest();
    opbase->need_ranksize_alignment = true;

    return opbase;
}

void delete_opbase_ptr(hccl::HcclTest* opbase)
{
    delete opbase;
    opbase = nullptr;
    return;
}

namespace hccl {
HcclOpBaseReducescatterVTest::HcclOpBaseReducescatterVTest() : HcclOpBaseTest()
{
    
    host_buf = nullptr;
    recv_buff_temp = nullptr;
    check_buf = nullptr;
    send_buff = nullptr;
    recv_buff = nullptr;
    send_counts = nullptr;
    send_disp = nullptr;
}

HcclOpBaseReducescatterVTest::~HcclOpBaseReducescatterVTest()
{
}

void HcclOpBaseReducescatterVTest::malloc_send_buf()
{
    send_counts = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    send_disp = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    for(int i = 0; i < rank_size; ++i)
    {
        send_counts[i] = data->count;
        send_disp[i] = i * send_counts[i];
    }
    return;
}

void HcclOpBaseReducescatterVTest::free_send_buf()
{
    free(send_counts);
    free(send_disp);
}

int HcclOpBaseReducescatterVTest::init_buf_val()
{
    //初始化校验内存
    ACLCHECK(aclrtMallocHost((void**)&check_buf, malloc_kSize));
    if(op_type == HCCL_REDUCE_PROD || dtype == HCCL_DATA_TYPE_INT8) {
        hccl_reduce_check_buf_init((char*)check_buf, data->count, dtype, op_type, val, rank_size);
    } else {
        hccl_reduce_check_buf_init(check_buf, data->count, dtype, op_type, rank_id + 1, rank_size);
    }
    return 0;
}

int HcclOpBaseReducescatterVTest::check_buf_result()
{
    //获取输出内存
    ACLCHECK(aclrtMallocHost((void**)&recv_buff_temp, malloc_kSize));
    ACLCHECK(aclrtMemcpy((void*)recv_buff_temp, malloc_kSize, (void*)recv_buff, malloc_kSize, ACL_MEMCPY_DEVICE_TO_HOST));

    int ret = 0;
    switch(dtype)
    {
        case HCCL_DATA_TYPE_FP32:
            ret = check_buf_result_float((char*)recv_buff_temp, (char*)check_buf, data->count);
            break;
        case HCCL_DATA_TYPE_INT8:
            ret = check_buf_result_int8((char*)recv_buff_temp, (char*)check_buf, data->count);
            break;
        case HCCL_DATA_TYPE_INT32:
            ret = check_buf_result_int32((char*)recv_buff_temp, (char*)check_buf, data->count);
            break;
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_BFP16:
            ret = check_buf_result_half((char*)recv_buff_temp, (char*)check_buf, data->count);
            break;
        case HCCL_DATA_TYPE_INT64:
            ret = check_buf_result_int64((char*)recv_buff_temp, (char*)check_buf, data->count);
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

void HcclOpBaseReducescatterVTest::cal_execution_time(float time)
{
    double total_time_us              = time * 1000;
    double average_time_us            = total_time_us / iters;
    double algorithm_bandwith_GBytes_s = malloc_kSize * rank_size / average_time_us * B_US_TO_GB_S;

    print_execution_time(average_time_us, algorithm_bandwith_GBytes_s);
    return;
}

size_t HcclOpBaseReducescatterVTest::init_malloc_Ksize_by_data()
{
    data->count = (data->count + rank_size - 1) / rank_size;
    return data->count * data->type_size;
}

void HcclOpBaseReducescatterVTest::init_send_recv_size_by_data(size_t &send_bytes, size_t &recv_bytes)
{
    send_bytes = malloc_kSize * rank_size;
    recv_bytes = malloc_kSize;
}
int HcclOpBaseReducescatterVTest::hccl_op_base_test() //主函数
{
    is_data_overflow();
    //申请sendcounts和send_disp
    malloc_send_buf();

    //初始化输入内存
    ACLCHECK(aclrtMallocHost((void**)&host_buf, malloc_kSize * rank_size));
    if(op_type == HCCL_REDUCE_PROD || dtype == HCCL_DATA_TYPE_INT8) {
        hccl_host_buf_init((char*)host_buf, data->count * rank_size, dtype, val);
    } else {
        for(int i = 0; i < rank_size; ++i)
        {
            hccl_host_buf_init(((char*)host_buf + i * malloc_kSize), data->count, dtype, i + 1);//+ i * malloc_kSize 跳到下一块内存中，写数据
        }
    }

    ACLCHECK(aclrtMemcpy((void*)send_buff, malloc_kSize * rank_size, (void*)host_buf, malloc_kSize * rank_size, ACL_MEMCPY_HOST_TO_DEVICE));

    // 准备校验内存
    if (check == 1) {
        ACLCHECK(init_buf_val());
    }

    //执行集合通信操作
    for (int j = 0; j < warmup_iters; ++j) {
        HCCLCHECK(HcclReduceScatterV((void *)send_buff, (const void *)send_counts, (const void *)send_disp,
            (void*)recv_buff, data->count, (HcclDataType)dtype, (HcclReduceOp)op_type, hccl_comm, stream));
    }

    ACLCHECK(aclrtRecordEvent(start_event, stream));

    for (int i = 0; i < iters; ++i) {
        HCCLCHECK(HcclReduceScatterV((void *)send_buff, (const void *)send_counts, (const void *)send_disp,
            (void*)recv_buff, data->count, (HcclDataType)dtype, (HcclReduceOp)op_type, hccl_comm, stream));
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
    free_send_buf();
    return 0;
}

void HcclOpBaseReducescatterVTest::is_data_overflow()
{
        if (op_type == HCCL_REDUCE_PROD) {
        if (dtype == HCCL_DATA_TYPE_FP16 && rank_size >= 16) {
            no_verification();
        }
        if (dtype == HCCL_DATA_TYPE_FP32 && rank_size >= 128) {
            no_verification();
        }
        if (dtype == HCCL_DATA_TYPE_INT8 && rank_size >= 7) {
            no_verification();
        }
        if (dtype == HCCL_DATA_TYPE_INT32 && rank_size >= 31) {
            no_verification();
        }
        if (dtype == HCCL_DATA_TYPE_INT64 && rank_size >= 63) {
            no_verification();
        }
    } else if (op_type == HCCL_REDUCE_SUM) {
        if(dtype == HCCL_DATA_TYPE_INT8 && REDUCESCATTERV_SUM_RESULE_OVERFLOW(rank_size,HCCL_DATA_TYPE_INT8)) {
            no_verification();
        }
        else if(dtype == HCCL_DATA_TYPE_INT16 && REDUCESCATTERV_SUM_RESULE_OVERFLOW(rank_size,HCCL_DATA_TYPE_INT16))
        {
            no_verification();
        }
        else if(dtype == HCCL_DATA_TYPE_INT32 && REDUCESCATTERV_SUM_RESULE_OVERFLOW(rank_size,HCCL_DATA_TYPE_INT32))
        {
            no_verification();
        }
        else if(dtype == HCCL_DATA_TYPE_FP16 && REDUCESCATTERV_SUM_RESULE_OVERFLOW(rank_size,HCCL_DATA_TYPE_FP16))
        {
            no_verification();
        }
        else if(dtype == HCCL_DATA_TYPE_BFP16 && REDUCESCATTERV_SUM_RESULE_OVERFLOW(rank_size,HCCL_DATA_TYPE_BFP16))
        {
            no_verification();
        }
        else if(dtype == HCCL_DATA_TYPE_FP32 && REDUCESCATTERV_SUM_RESULE_OVERFLOW(rank_size,HCCL_DATA_TYPE_FP32))
        {
            no_verification();
        }
    }
    return;
}
}
