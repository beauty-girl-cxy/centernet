# CenterNet代码阅读记录
## 文件结构
```
├─data
├─exp
├─experiments       
├─images
├─models
├─readme
└─src
    ├─lib
    │  ├─datasets   
    │  │  ├─dataset 
    │  │  └─sample  
    │  ├─detectors  
    │  ├─external   
    │  ├─models     
    │  │  └─networks
    │  │      └─DCNv2       
    │  │          └─src     
    │  │              └─cuda
    │  ├─trains
    │  └─utils
    └─tools
        ├─kitti_eval        
        └─voc_eval_lib      
            ├─datasets      
            ├─model
            ├─nms
            └─utils
```
`experiment`文件夹中存放了一些shell文件

`src`文件夹存放了核心代码

## 核心代码`src`文件夹结构
```
C:.
│  demo.py
│  main.py
│  test.py
│  _init_paths.py  将c:\Users\FX63\Desktop\CenterNet-master\src\lib添加到PYTHON临时搜索路径
│
├─lib
│  │  logger.py    设置输出日志格式
│  │  opts.py      用于解析命令行参数
│  │
│  ├─datasets               定义的数据集类
│  │  │  dataset_factory.py 定义了两个字典dataset_factory和_sample_factory，两个字典的值分别对应dataset和sample文件夹下定义的数据集类
│  │  │
│  │  ├─dataset 数据集的读入
│  │  │      coco.py        定义数据集类COCO
│  │  │      coco_hp.py     定义数据集类COCOHP
│  │  │      kitti.py       定义数据集类KITTI
│  │  │      pascal.py      定义数据集类PascalVOC
│  │  │
│  │  └─sample 数据集预处理和增强
│  │          ctdet.py      定义数据集类CTDetDataset
│  │          ddd.py        定义数据集类DddDataset
│  │          exdet.py      定义数据集类EXDetDataset
│  │          multi_pose.py 定义数据集类MultiPoseDataset
│  │
│  ├─detectors                  定义的检测器类
│  │      base_detector.py      定义BaseDetector类
│  │      ctdet.py              基于base_detector.py中的基类BaseDetector定义了CtdetDetector类。
│  │      ddd.py                基于base_detector.py中的基类BaseDetector定义了DddDetector类。
│  │      detector_factory.py   定义了一个字典，值为detectors文件夹下ctdet.py ddd.py exdet.py multi_pose.py中定义的四个类。
│  │      exdet.py              基于base_detector.py中的基类BaseDetector定义了ExdetDetector类。
│  │      multi_pose.py         基于base_detector.py中的基类BaseDetector定义了MultiPoseDetector类。
│  │
│  ├─external 用Cython写的NMS，在detecors/multi_pose.py中被调用，已在utils/nms.py中改为python版本
│  │      .gitignore
│  │      Makefile
│  │      nms.pyx
│  │      setup.py
│  │      __init__.py
│  │
│  ├─models
│  │  │  data_parallel.py    有关数据并行的东西，暂时不知道干啥的
│  │  │  decode.py           输出结果的解码
│  │  │  losses.py           损失函数
│  │  │  model.py            创建模型、加载模型和保存模型
│  │  │  scatter_gather.py   将变量切成大致相等的块并将它们分布在给定的GPU上
│  │  │  utils.py            暂时不知道干啥的
│  │  │
│  │  └─networks             定义了5个主干网络，其中最后两个需要使用CUDA编程搭建的DCNv2网络
│  │      │  dlav0.py
│  │      │  large_hourglass.py
│  │      │  msra_resnet.py
│  │      │  pose_dla_dcn.py
│  │      │  resnet_dcn.py
│  │      │
│  │      └─DCNv2  基于CUDA编程搭建的DCNv2网络，是主干网络之一
│  │          │  .gitignore
│  │          │  build.py
│  │          │  build_double.py
│  │          │  dcn_v2.py
│  │          │  dcn_v2_func.py
│  │          │  LICENSE
│  │          │  make.sh
│  │          │  README.md
│  │          │  test.py
│  │          │  __init__.py
│  │          │
│  │          └─src
│  │              │  dcn_v2.c
│  │              │  dcn_v2.h
│  │              │  dcn_v2_cuda.c
│  │              │  dcn_v2_cuda.h
│  │              │  dcn_v2_cuda_double.c
│  │              │  dcn_v2_cuda_double.h
│  │              │  dcn_v2_double.c
│  │              │  dcn_v2_double.h
│  │              │
│  │              └─cuda
│  │                      dcn_v2_im2col_cuda.cu
│  │                      dcn_v2_im2col_cuda.h
│  │                      dcn_v2_im2col_cuda_double.cu
│  │                      dcn_v2_im2col_cuda_double.h
│  │                      dcn_v2_psroi_pooling_cuda.cu
│  │                      dcn_v2_psroi_pooling_cuda.h
│  │                      dcn_v2_psroi_pooling_cuda_double.cu
│  │                      dcn_v2_psroi_pooling_cuda_double.h
│  │
│  ├─trains                 训练的代码
│  │      base_trainer.py   定义了基础的训练器BaseTrainer
│  │      ctdet.py          定义了基于BaseTrainer的CtdetTrainer
│  │      ddd.py            定义了基于BaseTrainer的DddTrainer
│  │      exdet.py          定义了基于BaseTrainer的ExdetTrainer
│  │      multi_pose.py     定义了基于BaseTrainer的MultiPoseTrainer
│  │      train_factory.py  定义了字典train_factory，四个值分别对应了ctdet.py ddd.py exdet.py multi_pose.py中定义的四个训练器
│  │
│  └─utils
│          ddd_utils.py     定义了ddd需要的功能性函数
│          debugger.py
│          image.py
│          oracle_utils.py
│          post_process.py  后处理
│          utils.py         定义了AverageMeter类，计算并存储平均值和当前值
│          __init__.py
│
└─tools   不知道干啥的
    │  calc_coco_overlap.py            计算coco格式数据集的overlap
    │  convert_hourglass_weight.py     模型文件转换
    │  convert_kitti_to_coco.py        kitti格式数据集转为coco格式
    │  eval_coco.py·                   评估coco格式数据集结果
    │  eval_coco_hp.py                 评估coco_hp格式数据集结果
    │  get_kitti.sh                    获取数据集的shell文件
    │  get_pascal_voc.sh               获取数据集的shell文件
    │  merge_pascal_json.py
    │  reval.py
    │  vis_pred.py
    │  _init_paths.py
    │
    ├─kitti_eval 对kitti格式数据集检测结果进行验证
    │      evaluate_object_3d.cpp
    │      evaluate_object_3d_offline
    │      evaluate_object_3d_offline.cpp
    │      mail.h
    │      README.md
    │
    └─voc_eval_lib  对voc格式数据集检测结果进行验证
        │  LICENSE
        │  Makefile
        │  setup.py
        │  __init__.py
        │
        ├─datasets
        │      bbox.pyx
        │      ds_utils.py
        │      imdb.py
        │      pascal_voc.py
        │      voc_eval.py
        │      __init__.py
        │
        ├─model
        │      bbox_transform.py
        │      config.py
        │      nms_wrapper.py
        │      test.py
        │      __init__.py
        │
        ├─nms
        │      .gitignore
        │      cpu_nms.c
        │      cpu_nms.pyx
        │      gpu_nms.cpp
        │      gpu_nms.hpp
        │      gpu_nms.pyx
        │      nms_kernel.cu
        │      py_cpu_nms.py
        │      __init__.py
        │
        └─utils
                .gitignore
                bbox.pyx
                blob.py
                timer.py
                visualization.py
                __init__.py
```

