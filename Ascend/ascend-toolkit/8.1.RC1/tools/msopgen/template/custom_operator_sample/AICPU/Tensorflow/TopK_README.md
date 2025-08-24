Function: Find values and indices of "k" largest elements for the last * dimension.


## TopK算子介绍
### 1. 算子功能
获取输入tensor中前k个最大元素的值和索引

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
│       │   tensorflow_top_k_plugin.cc
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   top_k.cc
│   │   top_k.h
│   
└───cpukernel
│   └───impl #算子实现目录
│       │   top_k.cc
│       │   top_k.h
│   └───op_info_cfg
│       └───aicpu_kernel
│           │   top_k.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───top_k
│           └───aicpu_kernel
│               │   TopK_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───top_k
│               │   test_top_k_impl.cc #算子实现ut
```
