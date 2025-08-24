#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright 2024-2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from jinja2 import Template

CONTENT = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024-2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataflow.flow_func as ff

class {{clz_name}}():
    def __init__(self):
        pass

    @ff.init_wrapper()
    def init_flow_func(self, meta_params):
        # init flow func here
        return ff.FLOW_FUNC_SUCCESS

    {% for f_name in f_names %}
    @ff.proc_wrapper()
    def {{f_name}}(self, run_context, input_flow_msgs):
        # process input message and set output here
        return ff.FLOW_FUNC_FAILED
    {% endfor %}

"""


TPL = Template(CONTENT)


def gen_py_func_code(clz_name, f_names):
    global TPL

    return TPL.render(clz_name=clz_name, f_names=f_names)
