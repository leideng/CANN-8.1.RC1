Function: Return the truth value of (x1 < x2) element-wise.

## Less算子介绍
### 1. 算子功能
逐一比较tensor1中各元素是否小于tensor2

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
│       │   tensorflow_less_plugin.cc
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   less.cc
│   │   less.h
│   
└───cpukernel
│   └───impl #算子实现目录
│       │   less.cc
│       │   less.h
│   └───op_info_cfg
│       └───aicpu_kernel
│           │   less.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───less
│           └───aicpu_kernel
│               │   Less_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───less
│               │   test_less_impl.cc #算子实现ut
```
