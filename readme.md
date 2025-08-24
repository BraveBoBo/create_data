# 示例可视化

本目录下提供了示例 episode 的可视化 GIF：

![assembly-v3 episode 0 camera0 rgb](assembly-v3_ep000_camera0_rgb.gif)

该 GIF 展示了 metaworld 任务 assembly-v3 的第 0 个 episode 在 camera0_rgb 视角下的完整过程，可用于数据集可视化、任务理解与调试。
# create_data 目录说明

本目录用于数据集的处理、生成与可视化，主要面向机器人模仿学习和强化学习任务。包含脚本、元数据、任务描述和教程等内容。

## 目录结构

- `collect_metadata.py`  
  数据集元数据收集与统计脚本。

- `metaworld_tasks_50_v2.json`  
  Meta-World MT50 任务集的环境名称及多样化自然语言描述。

- `turotial.ipynb`  
  数据集读取、结构分析、可视化与 GIF 导出等常用操作的 Jupyter 教程。

- `Metaworld/`  
  Meta-World 官方代码仓库（如为子模块或源码拷贝）。

## 主要功能

- 统计和分析数据集结构（如 zarr 格式的 metaworld/rlbench 数据集）。
- 导出 episode 级别的相机帧为 GIF，便于可视化。
- 批量统计 RLBench 多任务数据集的 episode 数、step 数、分辨率等。
- 提供任务描述的多样化文本，支持语言条件任务和数据增强。

## 快速开始

1. 运行 `turotial.ipynb`，了解如何读取和可视化数据集。
2. 使用 `collect_metadata.py` 统计数据集元信息。
3. 使用 `metaworld_tasks_50_v2.json` 进行任务描述多样化实验。

## 依赖

- Python 3.8+
- numpy, zarr, imageio, matplotlib, tqdm 等
- 相关 RL/IL 框架（如 diffusion_policy, metaworld, rlbench）

## 参考

- [Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning](https://arxiv.org/abs/1910.10897)
- [RLBench: Robot Learning Benchmark](https://github.com/stepjam/RLBench)
