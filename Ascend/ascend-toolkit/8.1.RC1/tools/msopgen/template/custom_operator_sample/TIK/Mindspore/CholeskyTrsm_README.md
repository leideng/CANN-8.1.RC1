Function: Only compute the res of the diag part of input matrix with dim 128.


## CholeskyTrsm算子介绍
### 1. 算子功能
Cholesky分解的逆实现


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Mindspore  |
| 实现方式 | Tik      |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│
└───mindspore
│   └───impl #原型公共依赖
│       │   cholesky_trsm_impl.py
└───proto #算子原型目录
│   │   cholesky_trsm.py
└───testcase #测试用例
│   └───st
│       └───cholesky_trsm
│           │   CholeskyTrsm_case_sample.json #算子st用例
│   └───ut
│       └───ops_test
│           │   test_cholesky_trsm_impl.py #算子实现ut
```
