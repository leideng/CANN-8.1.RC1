#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.

import configparser
import logging
import ssl
import os
import re
import json

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.ssl_ import create_urllib3_context
from generic_utils import check_input_file


class SecAdapter(HTTPAdapter):
    def __init__(self, ciphers):
        self.ciphers = ciphers
        super().__init__()

    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = create_urllib3_context(ciphers=self.ciphers)
        kwargs['ssl_version'] = ssl.PROTOCOL_TLSv1_2
        return super(SecAdapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs['ssl_context'] = create_urllib3_context(ciphers=self.ciphers)
        return super(SecAdapter, self).proxy_manager_for(*args, **kwargs)


class ModelInfoUtil(object):
    MODEL_TYPE_ENCODE = {"1": "Offline Model", "2": "Train Script"}
    CONFIG_PATH = os.path.join(os.path.realpath(os.path.dirname(__file__)), "ModelZoo.ini")
    URL_SECURE_PATTERN = re.compile(r"https://[-_A-Za-z0-9.]+\.huawei\.com[/a-zA-Z]+")
    LIMIT_SIZE = 100 * 1024 ** 2  # MB
    CONFIG_FILE_SIZE = 1 * 1024 ** 2 # MB
    CONFIG_KEY = "APIGateway"

    def __init__(self):
        check_input_file(ModelInfoUtil.CONFIG_PATH, ModelInfoUtil.CONFIG_FILE_SIZE)
        modelzoo_config = configparser.ConfigParser()
        modelzoo_config.read(self.CONFIG_PATH, encoding="utf-8")
        self.url = modelzoo_config.get(self.CONFIG_KEY, "url")
        cipher_list_str = modelzoo_config.get(self.CONFIG_KEY, "CIPHERS")
        cipher_list = json.loads(cipher_list_str)
        self.ciphers = ":".join(cipher_list)

        self.header = {
            "X-HW-ID": modelzoo_config.get(self.CONFIG_KEY, "X-HW-ID"),
            "Content-Type": "application/json;charset=utf-8",
            "Cookie": "default=default"
        }
        self.page_size = 50
        self.inserted_keys = [
            "id",
            "modelName",
            "modelType",
            "categoriesId",
            "lastModifyTime",
            "updateTime",
            "precision",
            "applicationArea",
            "frame",
            "versionName",
            "version",
            "versionPath",
            "modelFileUrl",
            "processor",
            "processorType",
            "modelFormat",
            "fileSize"
        ]

    @classmethod
    def is_secure_url(cls, url):
        return cls.URL_SECURE_PATTERN.match(url) is not None

    def check_url(self):
        if self.is_secure_url(self.url):
            return
        while True:
            message = input(f"As your script will visit external url {self.url} which may be insecure."
                            "Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
            if message == "continue" or message == "c":
                break
            elif message == "exit":
                exit(0)
            else:
                logging.info("Input is error, please enter 'exit' or 'c' or 'continue'.")

    def get_from_model_zoo(self):
        page_no, count, total = 1, 0, 0
        result = []
        success = True
        self.check_url()

        sess = requests.Session()
        sess.mount(self.url, SecAdapter(self.ciphers))
        size_in_bytes = 0
        while not total or count < total:
            payload = {
                "type": "1",
                "pageNo": page_no,
                "pageSize": self.page_size,
                "lang": "zh"
            }
            try:
                response, size_in_bytes = self.get_response(sess, payload, size_in_bytes)
                status_code = response.get("code")
                if status_code != 200:
                    raise ValueError(f'Response code invalid, got response code {status_code}.')
                response_data = response["data"]
                if not total:
                    total = response_data["totalCount"] or -1  # if count is 0, set -1 to break loop
                if not isinstance(total, int):
                    raise ValueError(f'Expect value of [totalCount] is int, but got {type(total)}.')

                res = response_data["list"]
                if not isinstance(res, list):
                    raise ValueError(f'Expect value of [list] is list, but got {type(res)}.')
                count += len(res)
            except Exception as exp:
                logging.error(exp)
                success = False
                break
            page_no += 1
            result.extend(res)

        sess.close()
        output = {"result": "success" if success else "fail"}
        if not success:
            return output
        target = self._filter_result(result)
        output["count"] = len(target)
        output["models"] = target

        return output

    def get_response(self, session, payload, size_in_bytes):
        buffer = bytearray()
        with session.get(self.url, headers=self.header, params=payload, verify=True,
                         stream=True, timeout=5) as response:
            for chunk in response.iter_content(4096):
                size_in_bytes += len(chunk)
                buffer.extend(chunk)
                if size_in_bytes > ModelInfoUtil.LIMIT_SIZE:
                    raise ValueError(f'Response is too large, '
                                     f'total received exceeds {ModelInfoUtil.LIMIT_SIZE / 1024 ** 2} MB.')
        response_json = json.loads(buffer)
        return response_json, size_in_bytes

    def _filter_result(self, result):
        target = []
        for model in result:
            filtered = dict((column_name, '') for column_name in self.inserted_keys)
            filtered.update({
                k: v
                for k, v in model.items()
                if k in self.inserted_keys
            })
            target.append(filtered)

        return target
