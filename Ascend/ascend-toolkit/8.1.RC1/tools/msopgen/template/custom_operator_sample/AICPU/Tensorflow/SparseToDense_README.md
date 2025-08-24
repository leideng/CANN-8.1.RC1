Function: Convert a sparse representation into a dense tensor.

## SparseToDense算子介绍
### 1. 算子功能
稀疏矩阵转密集矩阵

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
│       │   tensorflow_sparse_to_dense_plugin.cc
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   sparse_to_dense.cc
│   │   sparse_to_dense.h
│   
└───cpukernel
│   └───impl #算子实现目录
│       │   sparse_to_dense.cc
│       │   sparse_to_dense.h
│   └───op_info_cfg
│       └───aicpu_kernel
│           │   sparse_to_dense.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───sparse_to_dense
│           └───aicpu_kernel
│               │   SparseToDense_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───sparse_to_dense
│               │   test_sparse_to_dense_impl.cc #算子实现ut
```
