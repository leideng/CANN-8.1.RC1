Function: Scale the weights with a correction factor to the long term statistics prior to quantization.


## CusCorrectionMul算子介绍
### 1. 算子功能
使用校正因子处理权重.


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Mindspore  |
| 实现方式 | DSL      |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│
└───mindspore
│   └───impl #原型公共依赖
│       │   cus_correction_mul_impl.py
└───proto #算子原型目录
│   │   cus_correction.py
└───testcase #测试用例
│   └───st
│       └───cus_correction
│           │   CusCorrectionMul_case_sample.json #算子st用例
│   └───ut
│       └───ops_test
│           │   test_cus_correction_mul_impl.py #算子实现ut
```
