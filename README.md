# DataLoader 测试套件

针对 `ovi/data/loader.py`（Ovi 训练数据加载模块，946 行）的自动化测试。
全部用例离线运行：不依赖真实数据集、不联网、纯 CPU，测试数据由
`conftest.py` 在临时目录合成。

## 快速开始

```
python -m pytest tests_dataloader -v --basetemp=.pytest_tmp
```

## 文件结构

```
tests_dataloader/
├── conftest.py                 # 数据工厂
├── test_safe_collate.py        # UT001–004  批整理：补零 / 透传 / 混合 key 告警
├── test_aspect_buckets.py      # UT005–007  宽高比分桶：空输入 / 一致性 / 64 对齐
├── test_dataset_filtering.py   # UT008–015  初始化过滤：缺字段 / 帧数时长边界 / 全无效报错
├── test_process_data.py        # UT016–024、032  取数：解码 / 补齐 / 裁剪补零 / 重试回退 / 缩放
├── test_bucket_sampler.py      # UT025–028  批采样器：同桶 / drop_last / 分布式切分
├── test_aspect_ratio_loader.py # UT029–031  端到端：出批 / 批内对齐 / 模态隔离
└── test_defect_deep.py         # UT033–035  异常场景补充（跨桶回退 / fps 静默回退）
```

## 测试方法分布

| 方法 | 数量 | 说明 |
|---|---:|---|
| 等价类（EQV） | 9 | 输入分组各测一个代表（字段有无、类型对错、路径真假） |
| 边界值（BVA） | 11 | 专测区间端点及其两侧（帧数容差四点、时长上限两侧等） |
| 场景法（ST） | 10 | 完整流程串联（端到端出批、故障回退、全损坏报错） |
| 白盒（WB） | 6 | 针对源码分支与公式（64 对齐、or 链 fps、混合 key 告警） |

## 环境依赖

Python 3.14、torch 2.14.0+cpu、torchaudio 2.11.0+cpu、imageio、
imageio-ffmpeg、pytest、pytest-cov。无 GPU 要求。
