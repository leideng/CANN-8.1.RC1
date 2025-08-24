Function: Updates "var" according to the Adam algorithm.

## ApplyAdamD算子介绍
### 1. 算子功能
使用adam正则化来更新对应值。


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
│       │   tensorflow_apply_adam_d_plugin.cc 
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   apply_adam_d.cc
│   │   apply_adam_d.h
│   
└───tbe
│   └───impl #算子实现目录
│       │   apply_adam_d.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend910
│               │   apply_adam_d.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───apply_adam_d
│           └───ascend910
│               │   ApplyAdamD_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───apply_adam_d
│               │   test_apply_adam_d_impl.py #算子实现ut
```
