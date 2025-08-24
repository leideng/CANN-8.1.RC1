#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 
import os
import stat
import sys
import csv


SAVE_DATA_FILE_AUTHORITY = stat.S_IWUSR | stat.S_IRUSR
OPEN_FLAGS = os.O_WRONLY | os.O_CREAT


class TableOutputWrapper:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.file = None
        self.stdout = None
        self.csv_writer = None

    def __enter__(self):
        if self.filepath:
            self.file = os.fdopen(os.open(self.filepath, OPEN_FLAGS, SAVE_DATA_FILE_AUTHORITY), 'w')
            self.file.truncate()
            self.csv_writer = csv.writer(self.file, quoting=csv.QUOTE_ALL)
        else:
            self.stdout = sys.stdout
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()

    def write(self, row):
        if self.csv_writer:
            self.csv_writer.writerow(row)
        if self.stdout:
            fmt = "{:<20} " * len(row)
            self.stdout.write(fmt.format(*row))
            self.stdout.write("\n")