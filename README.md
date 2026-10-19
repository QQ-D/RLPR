# RLPR 复现与改进实验

更新至 2026 年 10 月 19 日。保存原始代码、按 Word 思路修改的工程、实验配置与日志、完整结果和后续方案。数据集、模型、预处理输入、NumPy 缓存及完整环境留在本机。

上游：[QiZS-BIT/RLPR](https://github.com/QiZS-BIT/RLPR)。原始源码快照：`ad6478a5499bab227d7f063cb77d4843dc6f9b4a`；修改版快照：`8c8bb597c4a88a2937e311bd045ecf89c3da6d1c`。各实验协议另有当时的源码和检查点哈希。

## 先看这些文件

- [后续研究与实验方案](docs/NEXT_STEPS.md)：跨城市验证、4D 雷达基线和高度 BEV 对照。
- [全部实验索引](experiments/INDEX.md)：保留成功、失败和中止的历史记录。
- [阶段报告](reports/)：所有汇报、PDF 及生成脚本。
- [温度对照报告](experiments/mulran_temperature_2026-10-19_192214/report.pdf)及[结果 JSON](experiments/mulran_temperature_2026-10-19_192214/results_summary.json)。
- [多正样本报告](experiments/mulran_spatial_positives_2026-10-19_181228/report.pdf)及[结果 JSON](experiments/mulran_spatial_positives_2026-10-19_181228/results_summary.json)。

## 当前结果

实验值为三个跨模态种子的平均 F1，固定 seed0 的 L5/R5 单模态权重。

| 方法 | Sejong 验证 F1 | DCC 测试 F1 | Riverside 测试 F1 |
|---|---:|---:|---:|
| 论文 RLPR | — | 0.7419 | 0.8057 |
| 官方权重本机测试 | — | 0.7500 | 0.81077 |
| 原单正样本 InfoNCE 温度 0.07 | 0.86635 | **0.75802** | **0.79678** |
| 空间多正样本 温度 0.07 | 0.86653 | 0.75476 | 0.79201 |
| 原单正样本 温度 0.03 | 0.84417 | 0.73940 | 0.79010 |
| 原单正样本 温度 0.10 | **0.86974** | 0.74642 | 0.79288 |

多正样本和温度 0.10 在各自实验中由验证规则选中，测试 F1 却未改善。选优锁保持原样，不按测试分数改写历史选择。DCC/Riverside 已被多次查看，属于探索性研究基准。当前结果不支持宣称整体优于论文，尚未完成其他数据集和 4D 雷达上的方法验证。

原作者源码默认 MulRan 预算为 LiDAR 5 轮、radar 5 轮、跨模态 1 轮；这不是论文明确写出的训练轮数。历史额外训练与短训另有记录。

## 目录

| 路径 | 内容 |
|---|---|
| `src/` | 原始可运行代码快照 |
| `src_spectral/` | 修改版模型、数据处理、训练、测试和回归检查 |
| `experiments/` | 冻结脚本、配置、JSON/CSV 结果、运行日志、核验与 TensorBoard 标量日志 |
| `reports/` | 阶段报告和生成脚本 |
| `design/` | 原 Word 思路文件 |
| `backups/` | 已有代码备份，不包含模型或数据 |
| `source_history/` | 源码版本、提交记录及相对原始版本的补丁 |
| `environment.lock.txt` | 本机依赖快照，包含本机路径，不能视为通用安装文件 |
| `UPLOAD_MANIFEST.json`、`SHA256SUMS.txt` | 上传范围、文件大小及校验值 |

## 运行与复核

先按[源码 README](src_spectral/README.md)准备环境、数据和官方权重，再看[修改版用法](src_spectral/docs/word-complete-usage.md)与对应实验的 `protocol.json`。实验记录中的权重和缓存路径用于本机溯源，实际文件没有上传。

```bash
cd src_spectral
python -m unittest discover -s tests -v
```

本机环境为 `/mnt/data300/envs/rlpr`，工程根目录为 `/media/rwj/ZX1 2TB/RLPR`。部分自动实验脚本固定本机绝对路径，迁移机器需统一路径配置；旧冻结脚本保留原内容。发布前完整回归检查在实际运行工程执行，上传源码逐文件核对。

没有上传 RAW/processed 数据、BEV 输入图、点云、模型、NumPy 缓存、wheel 或完整 Conda 环境。

## 历史记录

[2026 年 10 月 13 日 Sejong 短训记录](docs/SEJONG_2026-10-13.md)保留最初的范围与原始输出。早期 `nan` F1 属于当时版本；后续评估修正和正式对照另有日志，不能将短训与完整训练混用。
