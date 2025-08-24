Function: Cast a tensor from the src data type to the dst data type.

## Cast算子介绍
### 1. 算子功能
将输入tensor进行数据类型转换

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
│       │   tensorflow_cast_plugin.cc
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   cast.cc
│   │   cast.h
│   
└───cpukernel
│   └───impl #算子实现目录
│       │   cast.cc
│       │   cast.h
│   └───op_info_cfg
│       └───aicpu_kernel
│           │   cast.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───cast
│           └───aicpu_kernel
│               │   Cast_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───cast
│               │   test_cast_impl.cc #算子实现ut
```
