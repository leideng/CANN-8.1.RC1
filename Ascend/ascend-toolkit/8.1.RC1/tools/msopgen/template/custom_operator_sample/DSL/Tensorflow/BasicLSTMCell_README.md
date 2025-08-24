Function: Basic LSTM Cell forward calculation

## BasicLSTMCell算子介绍
### 1. 算子功能
基础的LSTM循环网络计算单元


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | TensorFlow  |
| 实现方式 | DSL      |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│
└───framework #插件目录
│   └───common #插件公共依赖
│   └───omg #插件公共依赖
│   └───tf_plugin #tensorflow插件目录
│       │   tensorflow_basic_lstm_cell_plugin.cc
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   basic_lstm_cell.cc
│   │   basic_lstm_cell.h
│
└───tbe
│   └───impl #算子实现目录
│       │   basic_lstm_cell.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   basic_lstm_cell.ini #算子信息库
│           └───ascend310p
│               │   basic_lstm_cell.ini #算子信息库
│           └───ascend910
│               │   basic_lstm_cell.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───basic_lstm_cell
│           └───ascend310
│               │   BasicLSTMCell_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───basic_lstm_cell
│               │   test_lp_norm_impl.py #算子实现ut
```
