Function: Multiply matrix 'a' by matrix 'b' in batches.


## BatchMatmul算子介绍
### 1. 算子功能
批量的进行张量切片相乘


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
│       │   batch_matmul_impl.py
└───proto #算子原型目录
│   │   batch_matmul.py
└───testcase #测试用例
│   └───st
│       └───batch_matmul
│           │   BatchMatmul_case_sample.json #算子st用例
│   └───ut
│       └───ops_test
│           │   test_batch_matmul_impl.py #算子实现ut
```
