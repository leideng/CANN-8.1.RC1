# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

import logging


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL
}


def set_log_level(level="info"):
    if level.lower() in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS.get(level.lower()))
    else:
        logger.warning("Set %s log level failed.", level)


def set_logger(profiler_logger):
    profiler_logger.propagate = False
    profiler_logger.setLevel(logging.INFO)
    if not profiler_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        profiler_logger.addHandler(stream_handler)


logger = logging.getLogger("msServiceProfiler")
set_logger(logger)
