Function: Decode the bounding box according to different encoding or decoding methods.


## DecodeBboxV2算子介绍
### 1. 算子功能
目标检测SSD网络中的常用算子，主要根据不同的编码/解码方式(code_type)对边界框进行解码，具体功能可参考相关论文。


### 2. 基本信息
| **类型**       | **状态**    |
|-------------|---------------|
| 框架类型    | Tensorflow  |
| 实现方式 | Tik      |
| 动态/静态shape  | 静态 |

### 3. 主要工程结构
```
project
│
└───framework #插件目录
│   └───common #插件公共依赖
│   └───omg #插件公共依赖
│   └───tf_plugin #tensorflow插件目录
│       │   tensorflow_decode_bbox_v2_plugin.cc # 算子插件文件
│
└───proto #算子原型目录
│   └───util #原型公共依赖
│   │   decode_bbox_v2.cc
│   │   decode_bbox_v2.h
│
└───tbe
│   └───impl #算子实现目录
│       │   decode_bbox_v2.py
│   └───op_info_cfg #算子信息库
│       └───aicore
│           └───ascend310
│               │   decode_bbox_v2.ini
│           └───ascend310p
│               │   decode_bbox_v2.ini
│           └───ascend910
│               │   decode_bbox_v2.ini
└───testcase #测试用例
│   └───st
│       └───decode_bbox_v2
│           └───ascend310
│               │   DecodeBboxV2_case.json #算子st用例
│   └───ut
│       └───ops_test
│           └───DecodeBboxV2
│               │   test_decode_bbox_v2_impl.py #算子实现ut
```
