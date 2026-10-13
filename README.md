# RLPR Sejong 复现记录（2026-10-13）

本目录归档本机已运行的 RLPR 源码、短训脚本、环境依赖快照、阶段汇报及原始运行日志。

上游：[QiZS-BIT/RLPR](https://github.com/QiZS-BIT/RLPR)，源码提交 `ad6478a5499bab227d7f063cb77d4843dc6f9b4a`。
`src/` 保存本次本地运行的 71 个 Git 跟踪文件，不包含上游 `.git` 历史或本地 Python 缓存。

## 文件

- `src/`：模型、训练、评估、各数据集预处理代码及上游 README。
- `sejong_short_train.sh`：Sejong 小样本三阶段短训入口。
- `environment.lock.txt`：本机依赖快照，包含本地 wheel 路径及 ROS 依赖，不能直接视为通用安装文件。
- `reports/2026-10-13/`：Word、PDF、HTML 汇报及原始运行日志。
- `SHA256SUMS.txt`：上传文件校验值。

数据集、预处理输出、模型权重、wheel、描述子缓存及完整 Conda 环境留在本机。

## 本次范围与结果

数据：MulRan Sejong01、Sejong02；同步有效帧 17,471；测试数据库 1,505，查询 1,380。
短训：训练区域中抽取数据库 1,024、查询 256，batch size 1，LiDAR、radar、跨模态三个阶段各 2 轮。

| 阶段 | epoch 1 loss | epoch 2 loss |
|---|---:|---:|
| LiDAR | 0.14391835 | 0.05139263 |
| Radar | 0.15860988 | 0.05707145 |
| 跨模态 | 0.62873495 | 0.34575570 |

| Sejong 跨模态测试 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | F1 原始输出 |
|---|---:|---:|---:|---:|---:|
| 作者 MulRan 预训练权重 | 77.32% | 88.77% | 92.54% | 95.94% | 0.8831734961 |
| 本次跨模态短训第 2 轮 | 46.52% | 64.13% | 70.58% | 77.39% | nan |

`trainer/eval.py` 的 F1 零分母处理和 Recall@1percent 累计逻辑存在待修正问题；短训 F1 无效，Recall@1percent 原始输出待审计。上述源码保留本次实际运行版本，没有在归档时改动评估算法。
本次是 Sejong 小样本运行验证，不代表完成全 MulRan 或全部数据集复现。其他数据集进度以本阶段报告的本机范围为准。

## 本机运行

短训脚本包含本机 Conda 路径 `/home/rwj/anaconda3` 和环境 `/mnt/data300/envs/rlpr`。原始可运行项目位于 `/media/rwj/ZX1 2TB/RLPR`；GitHub 副本不包含运行所需的数据和模型。

```bash
bash "/media/rwj/ZX1 2TB/RLPR/sejong_short_train.sh"
```

使用作者预训练权重评估：

```bash
source /home/rwj/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/data300/envs/rlpr
cd "/media/rwj/ZX1 2TB/RLPR/src"
export RLPR_MULRAN_DATA="/media/rwj/ZX1 2TB/RLPR/processed/mulran_sejong"
export RLPR_MULRAN_TEST_SPLIT=sejong
export RLPR_CHECKPOINT_PATH="/media/rwj/ZX1 2TB/RLPR/pretrained/mulran.pth.tar"
export RLPR_TRAINING_ROOT="/media/rwj/ZX1 2TB/RLPR/runs/sejong_eval"
export TORCH_HOME="/media/rwj/ZX1 2TB/RLPR/cache/torch"
python test_cross_modal.py --dataset mulran
```

评估短训模型时，将 `RLPR_CHECKPOINT_PATH` 设为 `/media/rwj/ZX1 2TB/RLPR/runs/sejong_short_train/r2l_mulran/weights/2.pth.tar`。

其他机器需要先配置环境并下载所需数据和权重，再根据实际路径修改脚本及环境变量。
