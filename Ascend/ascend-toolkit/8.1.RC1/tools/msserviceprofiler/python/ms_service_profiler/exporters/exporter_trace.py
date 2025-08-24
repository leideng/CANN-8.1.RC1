# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

import json
import os
import threading
from ms_service_profiler.exporters.base import ExporterBase
from ms_service_profiler.utils.file_open_check import ms_open
from ms_service_profiler.plugins.plugin_req_status import ReqStatus
from ms_service_profiler.utils.log import logger


class ExporterTrace(ExporterBase):
    name = "trace"

    @classmethod
    def initialize(cls, args):
        cls.args = args

    @classmethod
    def export(cls, data) -> None:
        cpu_data_df, memory_data_df = data['cpu_data_df'], data['memory_data_df']
        tids = set(str(x) for x in set(data["tx_data_df"]["tid"])) if "tid" in data["tx_data_df"] else {}
        all_data_df = data['tx_data_df'].copy()
        if 'pid_label_map' in data:
            pid_label_map = data['pid_label_map']
        else:
            pid_label_map = None
        all_data_df['domain'] = all_data_df['domain'].replace('PDSplit', 'PDCommunication')
        msprof_data_df = data['msprof_data']
        cann_data = [load_single_prof(pf, tids) for pf in msprof_data_df]
        output = cls.args.output_path
        trace_data = create_trace_events(all_data_df, cpu_data_df, memory_data_df, pid_label_map)
        merged_data = merge_json_data(trace_data, cann_data)
        save_trace_data_into_json(merged_data, output)


def load_single_prof(pf, tids):
    try:
        with open(pf, 'r', encoding='utf-8') as file:
            trace_events = json.load(file)
    except FileNotFoundError:
        logger.warning(f"The msprof.json file was not found. Please check the file path.")
        return {"traceEvents": []}
    except json.JSONDecodeError:
        logger.warning(
            "%r is not in a valid JSON format, " \
            "which might be normal and probably because this file stores 'mstx' data only",
            pf
        )
        return {"traceEvents": []}

    return {"traceEvents": trace_events}


def find_cann_pid(trace_events):
    for event in trace_events:
        if event.get("name") == "process_name":
            args = event.get("args", {})
            if args.get("name") == "CANN":
                return event.get("pid")
    return None


def merge_json_data(trace_data, msprof_data_df):
    for item in msprof_data_df:
        events = item.get("traceEvents", [])
        trace_data["traceEvents"].extend(events)
    return trace_data


def write_trace_data_to_file(trace_data, output):
    with ms_open(output, "w") as f:
        json.dump(trace_data, f, ensure_ascii=False)
    logger.info("Written trace data successfully.")


def save_trace_data_into_json(trace_data, output):
    file_path = os.path.join(output, 'chrome_tracing.json')

    # 创建并启动新线程来执行写文件操作
    try:
        write_thread = threading.Thread(target=write_trace_data_to_file, args=(trace_data, file_path))
        write_thread.start()
        logger.info("Start to write trace data...")
    except Exception as e:
        logger.error(f"Failed to write trace data to file: {e}")


def add_flow_event(flow_event_df):
    flow_event_df.loc[:, 'rid'] = flow_event_df['rid'].str.split(',')
    exploded_df = flow_event_df.explode('rid')
    exploded_df['tid'] = exploded_df['domain']
    if 'PDCommunication' in flow_event_df['domain'].values:
        exploded_df['ph'] = [
            's' if 'httpReq' in name else ('f' if ('httpRes' in name and 'receiveToken=' in message) else 't')
            for name, message in zip(exploded_df['name'], exploded_df['message'])
        ]
    else:
        exploded_df['ph'] = [
            's' if 'httpReq' in name else ('f' if 'httpRes' in name else 't')
            for name in exploded_df['name']
        ]
    exploded_df['bp'] = ['b' if 'httpRes' in name else '' for name in exploded_df['name']]
    exploded_df['name'] = 'flow_' + exploded_df['rid']
    exploded_df['ts'] = exploded_df['start_time']
    exploded_df['id'] = exploded_df['rid']
    exploded_df['cat'] = exploded_df['rid']
    exploded_df['pid'] = exploded_df['pid']
    flow_trace_events = exploded_df[['name', 'ph', 'ts', 'id', 'cat', 'pid', 'tid']].to_dict(orient='records')
    return flow_trace_events


def create_trace_events(all_data_df, cpu_data_df, memory_data_df, pid_label_map=None):
    metric_event = ['npu', 'KVCache', 'PullKVCache']

    # 普通事件
    valid_name_df = all_data_df[all_data_df['name'].notna() & (~all_data_df['domain'].isin(metric_event))]
    trace_events = add_trace_events(valid_name_df)

    # metric事件
    cpu_trace_events = add_cpu_events(cpu_data_df)
    trace_events.extend(cpu_trace_events)

    mem_trace_events = add_mem_events(memory_data_df)
    trace_events.extend(mem_trace_events)

    npu_trace_events = add_npu_events(all_data_df[all_data_df['name'] == 'npu'])
    trace_events.extend(npu_trace_events)

    kv_trace_events = add_kvcache_events(all_data_df[all_data_df['domain'] == 'KVCache'])
    trace_events.extend(kv_trace_events)

    pull_kvcache_events = add_pull_kvcache_events(all_data_df[all_data_df['domain'] == 'PullKVCache'])
    trace_events.extend(pull_kvcache_events)

    # flow事件
    flow_event_df = valid_name_df[valid_name_df['rid'].notna()]
    flow_trace_events = add_flow_event(flow_event_df)
    trace_events.extend(flow_trace_events)
    trace_events = sort_trace_events_by_tid(trace_events)
    if pid_label_map is not None:
        trace_events.extend(sort_trace_events_by_pid(pid_label_map))

    trace_data = {"traceEvents": trace_events}
    return trace_data


def sort_trace_events_by_pid(pid_label_map):
    pid_sorting_meta = []
    pid_sorting = []
    for pid, item in pid_label_map.items():
        host_name = item.get("hostname", "")
        dp = item.get("dp", -1)
        pid_sorting.append((pid, host_name, dp))
    
    pid_sorting.sort(key=lambda x: (x[2], x[1]))

    for index, item in enumerate(pid_sorting):
        pid, host_name, dp = item
        pid_sorting_meta.append(dict(
            name="process_sort_index",
            ph="M",
            pid=pid,
            args=dict(sort_index=index))
        )
        if dp == -1:
            labels = [host_name]
        else:
            labels = [host_name, f"dp{int(dp)}"]
        pid_sorting_meta.append(dict(
            name="process_labels",
            ph="M",
            pid=pid,
            args=dict(labels=','.join(labels)))
        )
    
    return pid_sorting_meta


def sort_trace_events_by_tid(trace_events):
    req_status_names = list(ReqStatus.__members__.keys())
    # mindie 330将BatchScheduler打点修改为batchFrameworkProcessing，此处做新旧版本的兼容处理
    tid_sorting_order = ['http', 'Queue'] + req_status_names + \
        ['BatchSchedule', 'modelExec', 'batchFrameworkProcessing']
    main_pid = 0
    for event_info in trace_events:
        if event_info.get("cat") in tid_sorting_order:
            main_pid = event_info.get("pid")
            break
    tid_sorting_meta = [dict(
        name="thread_sort_index",
        ph="M",
        pid=main_pid,
        tid=tid,
        args=dict(sort_index=index)) for index, tid in enumerate(tid_sorting_order)]

    def get_tid_sorting_key(event):
        if 'tid' in event and event['tid'] in tid_sorting_order:
            return tid_sorting_order.index(event['tid'])
        else:
            return len(tid_sorting_order)

    # 排序 trace_events
    sorted_trace_events = sorted(trace_events, key=get_tid_sorting_key)

    sorted_trace_events.extend(tid_sorting_meta)
    return sorted_trace_events


def add_trace_events(valid_name_df):
    trace_event_df = valid_name_df.copy()

    # trace事件
    trace_event_df['ph'] = ['I' if during_time == 0 else 'X' for during_time in valid_name_df['during_time']]
    trace_event_df['ts'] = valid_name_df['start_time']
    trace_event_df['tid'] = valid_name_df['domain']
    trace_event_df['dur'] = valid_name_df['during_time']
    args_list = []
    for start, end, batch_type, batch_size, res_list, rid, message, tid, in zip(
            valid_name_df['start_datetime'],
            valid_name_df['end_datetime'],
            valid_name_df['batch_type'],
            valid_name_df['batch_size'],
            valid_name_df['res_list'],
            valid_name_df['rid'],
            valid_name_df['message'],
            valid_name_df['tid']
    ):
        args_dict = dict(**{k: v for k, v in message.items() if k not in ["domain", "name", "type", "rid"]}, **{
            'start_datetime': start,
            'end_datetime': end,
            'tid': tid
        })
        if batch_size is not None:
            args_dict.update({
                'batch_type': batch_type,
                'batch_size': batch_size,
            })
        if res_list is not None:
            args_dict.update({"res_list": res_list})
        if batch_size is None and rid != res_list:
            args_dict.update({"rid": rid})
        args_list.append(args_dict)
    trace_event_df['args'] = args_list
    trace_events = trace_event_df[['name', 'ph', 'ts', 'dur', 'pid', 'tid', 'args']].to_dict(orient='records')
    return trace_events


def add_cpu_events(cpu_data_df):
    if cpu_data_df is None or cpu_data_df.shape[0] == 0:
        return []
    cpu_trace_df = cpu_data_df.copy()
    cpu_trace_df['name'] = 'CPU Usage'
    cpu_trace_df['ph'] = 'C'
    cpu_trace_df['ts'] = cpu_data_df['start_time']
    cpu_trace_df['pid'] = 1
    cpu_trace_df['tid'] = 'CPU Usage'
    cpu_trace_df['args'] = [{'CPU Usage': usage} for usage in cpu_data_df['usage']]
    cpu_trace_events = cpu_trace_df[['name', 'ph', 'ts', 'pid', 'tid', 'args']].to_dict(orient='records')
    return cpu_trace_events


def add_mem_events(df):
    if df is None or df.shape[0] == 0:
        return []
    df = df.copy()
    df['name'] = 'Memory Usage'
    df['ph'] = 'C'
    df['ts'] = df['start_time']
    df['pid'] = 1
    df['tid'] = 'Memory Usage'
    df['args'] = [{'Memory Usage': usage} for usage in df['usage']]
    events = df[['name', 'ph', 'ts', 'pid', 'tid', 'args']].to_dict(orient='records')
    return events


def add_npu_events(npu_data_df):
    if npu_data_df is None or npu_data_df.shape[0] == 0:
        return []
    npu_trace_df = npu_data_df.copy()
    npu_trace_df['name'] = 'NPU Usage'
    npu_trace_df['ph'] = 'C'
    npu_trace_df['ts'] = npu_data_df['start_time']
    npu_trace_df['pid'] = 1
    npu_trace_df['tid'] = 'NPU Usage'
    npu_trace_df['args'] = [{'Usage': usage} for usage in npu_data_df['usage=']]
    npu_trace_events = npu_trace_df[['name', 'ph', 'ts', 'pid', 'tid', 'args']].to_dict(orient='records')
    return npu_trace_events


def add_kvcache_events(kv_data_df):
    if 'deviceBlock=' not in kv_data_df:
        return []
    kv_trace_df = kv_data_df.copy()
    if "scope#dp" in kv_trace_df:
        kv_trace_df['name'] = kv_trace_df['domain'] + '-dp' + kv_trace_df["scope#dp"].astype(int,
                                                                                            errors='ignore').astype(str)
    else:
        kv_trace_df['name'] = kv_trace_df['domain']
    kv_trace_df['ph'] = 'C'
    kv_trace_df['ts'] = kv_data_df['start_time']
    kv_trace_df['tid'] = kv_data_df['domain']
    kv_trace_df['args'] = [{'Device Block': usage} for usage in kv_data_df['deviceBlock=']]
    kv_trace_events = kv_trace_df[['name', 'ph', 'ts', 'pid', 'tid', 'args']].to_dict(orient='records')
    return kv_trace_events


def add_pull_kvcache_events(df):
    if df is None or df.shape[0] == 0:
        return []
    df_all_device = df.copy()
    all_events = []

    for rank in df_all_device['rank'].unique():
        df = df_all_device[df_all_device['rank'] == rank].copy().reset_index(drop=True)
        df['pid'] = "PullKVCache"
        df['name'] = df['domain']
        df['ph'] = 'X'
        df['ts'] = df['start_time']
        df['tid'] = f"Prefill Device {rank}"
        df['dur'] = df['during_time']
        args = ['rank', 'rid', 'block_tables', 'seq_len', \
                'during_time', 'start_datetime', 'end_datetime', 'start_time', 'end_time']
        df['args'] = df[[arg for arg in args if arg in df.columns]].to_dict(orient='records')
        events = df[['name', 'ph', 'ts', 'pid', 'tid', 'args', 'dur']].to_dict(orient='records')
        all_events.extend(events)

        df_decode = df.copy()
        df['ph'] = 'X'
        df_decode['ts'] = df_decode['end_time']
        df_decode['dur'] = 1
        df_decode['tid'] = f"Decode Device {rank}"
        events = df_decode[['name', 'ph', 'ts', 'pid', 'tid', 'args', 'dur']].to_dict(orient='records')
        all_events.extend(events)

        for i, _ in df.iterrows():
            all_events.append({
                "id": f"pull_kvcache_rank{rank}_{i}",
                "cat": f"pull_kvcache_rank{rank}_{i}",
                "name": f"pull_kvcache_rank{rank}_{i}",
                "pid": "PullKVCache",
                "tid": df['tid'][i],
                "ph": 's',
                "ts": df['start_time'][i],
            })

            all_events.append({
                "id": f"pull_kvcache_rank{rank}_{i}",
                "cat": f"pull_kvcache_rank{rank}_{i}",
                "name": f"pull_kvcache_rank{rank}_{i}",
                "pid": "PullKVCache",
                "tid": df_decode['tid'][i],
                "ph": 't',
                "ts": df['end_time'][i],
            })

    return all_events
