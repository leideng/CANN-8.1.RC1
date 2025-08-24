Function: Reslove the complexity of pooling function.

## AdaptiveMaxPool2d算子介绍
### 1. 算子功能
对输入tensor进行2维的自适应最大池化操作

### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Onnx  |
| 算子类型 |  AICPU     |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│  
└───framework #插件目录
│   └───common #插件公共依赖
│   └───omg #插件公共依赖
│   └───onnx_plugin #Onnx插件目录
│       │   adaptive_max_pool2d_plugin.cc
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   adaptive_max_pool2d.cc
│   │   adaptive_max_pool2d.h
│   
└───cpukernel
│   └───impl #算子实现目录
│       │   adaptive_max_pool2d.cc
│       │   adaptive_max_pool2d.h
│   └───op_info_cfg
│       └───aicpu_kernel
│           │   fill_v2d.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───adaptive_max_pool2d
│           └───aicpu_kernel
│               │   AdaptiveMaxPool2d_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───adaptive_max_pool2d
│               │   test_adaptive_max_pool2d_impl.cc #算子实现ut
```
