Function: Computes the maximum along segments of a tensor.

## UnsortedSegmentMax算子介绍
### 1. 算子功能
沿着张量的片段计算最大值.

### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Tensorflow  |
| 实现方式 | Tik      |
| 动态/静态shape  | 动态&静态 |

### 3. 主要工程结构
```
project
│
└───framework #插件目录
│   └───common #插件公共依赖
│   └───omg #插件公共依赖
│   └───tf_plugin #tensorflow插件目录
│       │   tensorflow_unsorted_segment_max_plugin.cc
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   unsorted_segment_max.cc
│   │   unsorted_segment_max.h
│
└───tbe
│   └───impl #算子实现目录
│       └───dynamic #算子实现目录
│           │   unsorted_segment_max.py #算子动态实现
│       │   unsorted_segment_max.py  #算子静态实现
│   └───op_info_cfg #算子信息库
│       └───aicore
│           └───ascend310
│               │   unsorted_segment_max.ini
│           └───ascend310p
│               │   unsorted_segment_max.ini
│           └───ascend910
│               │   unsorted_segment_max.ini
└───testcase #测试用例
│   └───st
│       └───unsorted_segment_max
│           └───ascend310
│               │   UnsortedSegmentMax_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───space_to_depth
│               │   test_unsorted_segment_max_impl.py #算子实现ut
```
