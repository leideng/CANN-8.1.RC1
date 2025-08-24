Function: Returns locations of nonzero/true values in a tensor.

## Where算子介绍
### 1. 算子功能
获取输入tensor中非零元素的位置

### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | TensorFlow  |
| 算子类型 |  AICPU     |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│  
└───framework #插件目录
│   └───common #插件公共依赖
│   └───omg #插件公共依赖
│   └───tf_plugin #TensorFlow插件目录
│       │   tensorflow_where_plugin.cc
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   where.cc
│   │   where.h
│   
└───cpukernel
│   └───impl #算子实现目录
│       │   where.cc
│       │   where.h
│   └───op_info_cfg
│       └───aicpu_kernel
│           │   where.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───where
│           └───aicpu_kernel
│               │   Where_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───where
│               │   test_where_impl.cc #算子实现ut
```
