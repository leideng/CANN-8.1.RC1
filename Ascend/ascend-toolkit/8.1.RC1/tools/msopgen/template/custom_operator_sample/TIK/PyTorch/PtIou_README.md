Function: Image Processing Intersection-over-Union(IoU) Calculation, Iou = Area of Overlap / Area of Union.


## PtIou算子介绍
### 1. 算子功能
交并比（Intersection-over-Union，IoU），目标检测中使用的一个概念，
是产生的候选框（candidate bound）与原标记框（ground truth bound）的交叠率，
即它们的交集与并集的比值。最理想情况是完全重叠，即比值为1。


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Pytorch  |
| 实现方式 | Tik      |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│  
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   pt_iou.cc
│   │   pt_iou.h
│   
└───tbe
│   └───impl #算子实现目录
│       │   pt_iou.py
│   └───op_info_cfg
│       └───aicore
│           └───ascend310
│               │   pt_iou.ini #算子信息库
└───testcase #测试用例
│   └───st
│       └───pt_iou
│           └───ascend310
│               │   PtIou_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───pt_iou
│               │   test_pt_iou_impl.py #算子实现ut
```
