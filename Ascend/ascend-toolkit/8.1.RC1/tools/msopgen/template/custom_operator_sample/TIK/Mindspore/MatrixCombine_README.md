Function: Move the batch matrix to the result matrix diagonal part.

## MatrixCombine算子介绍
### 1. 算子功能
将批处理矩阵移动柜到结果矩阵对角线不分，注意：输入tensor的秩必须为3.


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
│       │   matrix_combine_impl.py
└───proto #算子原型目录
│   │   matrix_combine.py
└───testcase #测试用例
│   └───st
│       └───matrix_combine
│           │   MatrixCombine_case_sample.json #算子st用例
│   └───ut
│       └───ops_test
│           │   test_matrix_combine_impl.py #算子实现ut
```
