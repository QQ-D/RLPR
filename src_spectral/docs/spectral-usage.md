# Spectral-RLPR 使用说明

代码目录：`/media/rwj/ZX1 2TB/RLPR/src_spectral`，分支 `spectral-rlpr`。
原版 `../src` 和 `../backups/rlpr_code_2026-10-14_201621` 保留。
使用已有 `rlpr` Conda 环境，不需要安装新依赖。

## 方案与 Word 的对应关系

1. 移除 spectral 变体的 LiDAR/radar Mamba PCE。浅层 CNN 后加入频域处理，
   后面的 ResNet、Transformer、CAP/NetVLAD 及 512 维描述子接口保留。
2. `rfft2` 获取实输入的频谱；以幅度统计预测每个输入的低通截止频率。
   掩码为正实数，抑制高频，保留部分幅度信息，实际效果需要实验验证。
3. Q/K 由相位的圆周表示生成，V 由复数频谱的实部和虚部生成；
   注意力在每种模态内部计算，数据库描述子可独立预计算。
4. 约束实频谱边界的共轭对称性，再用 `irfft2` 重建特征。
5. 第二阶段完整冻结 radar 分支及 BatchNorm 统计；
   `loss = InfoNCE_local + InfoNCE_global + 0.1 * phase_loss`。
   相位损失对同步同帧的 radar/LiDAR 特征计算，并降低低能量频率权重、排除 DC。

实现修正：FFT 尺寸随输入确定；不用 Word 中倒置的 225x50 常量。
FFT 使用 FP32。正实幅度掩码本身不改变相位，相位损失训练浅层 CNN 和
相位注意力；掩码通过检索损失以及后续注意力传来的间接相位梯度更新。
它是相位余弦一致性，不是统计意义的幅值平方相干度。
随后新增 radar→LiDAR 的相位交叉注意力（第二阶段成对训练）；实现与运行指令
见 [交叉融合说明](phase-cross-fusion.md)。旧模型默认关闭，便于保留已测结果。
没有声称相位天然稀疏不变。
当前相位约束不补偿传感器外参、车辆位姿差异或数据增强造成的几何偏移。

`50x225`（MulRan/nuScenes/Boreas）和 `100x100`（Snail/Hercules）路径都已接入。
后者沿用仓库原有 Cart 类名，实际数据表示以各数据集预处理脚本为准。

## Sejong 三阶段短训（每阶段 2 轮）

```bash
bash "/media/rwj/ZX1 2TB/RLPR/src_spectral/scripts/sejong_spectral_train.sh"
```

默认数据库 1024、查询 256、batch size 1；仅抽取 Sejong 训练区域。
训练查询从训练数据库内抽取，与原短训采用相同采样。此设置用于执行验证，
完整训练的泛化性能需使用正式划分和足够训练轮数评估。
每次写入新的 `../runs/sejong_spectral_<时间>/`，包含 `run.log`、`run.json`、
索引、BEV、TensorBoard 和三个阶段权重。终端最后打印跨模态权重路径。

可调整：`RLPR_SMOKE_EPOCHS`、`RLPR_SMOKE_DB`、`RLPR_SMOKE_QUERY`。
`RLPR_SMOKE_ROOT` 可指定输出目录，重跑时请使用新目录。

## 用新权重评估 Sejong

训练结束后，将下面的 checkpoint 路径替换成终端打印的实际路径：

```bash
source /home/rwj/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/data300/envs/rlpr
cd "/media/rwj/ZX1 2TB/RLPR/src_spectral"
export TORCH_HOME="/media/rwj/ZX1 2TB/RLPR/cache/torch"
export RLPR_MULRAN_DATA="/media/rwj/ZX1 2TB/RLPR/processed/mulran_sejong"
export RLPR_MULRAN_TEST_SPLIT=sejong
export RLPR_CHECKPOINT_PATH="/media/rwj/ZX1 2TB/RLPR/runs/sejong_spectral_<时间>/r2l_mulran/weights/2.pth.tar"
export RLPR_TRAINING_ROOT="/media/rwj/ZX1 2TB/RLPR/runs/sejong_spectral_eval_$(date +%Y-%m-%d_%H%M%S)"
python test_cross_modal.py --dataset mulran --model spectral
```

原版作者权重仍通过 `--model rlpr` 评估；它不能作为 spectral 模型直接评估。
spectral 第二阶段必须加载新架构的两支单模态预训练权重，避免冻结随机初始化的 radar 模块。
其余数据集入口增加同样的 `--model spectral` 参数，先按原脚本准备数据和训练索引。
Hercules 使用 spectral Snail 跨模态权重初始化。

## 消融选项

- `RLPR_SPECTRAL_DYNAMIC=0`：输入无关的可学习低通掩码。
- `RLPR_SPECTRAL_PHASE_ATTENTION=0`：关闭相位注意力。
- `RLPR_PHASE_LOSS_WEIGHT=0`：关闭相位损失。

阶段一、阶段二、评估使用相同 attention/dynamic 配置；checkpoint 会检查它们。
改变结构配置时重新训练对应单模态权重。

## 验证范围与限制

2026-10-14 验证：16 项 CPU 单元测试通过；RTX 4060 Laptop GPU 上使用合成数据
完成 LiDAR、radar、跨模态三个阶段各两轮（每轮一个 batch），保存并重新加载权重。
一对真实 Sejong BEV 前向通过；100x100 路径 GPU 前向和反向通过。
验证摘要见 [GPU 记录](verification/gpu-2026-10-14.json)。这些合成损失不是检索性能。

检查层/模型尺寸、梯度、独立描述子、冻结教师、训练损失和权重兼容性；
单元检查不代表性能提升。完整真实数据训练由用户运行。
2026-10-14 评估修正：F1 的 precision=recall=0 点按 0 处理，防止整个最大 F1
变成 nan；各 Recall 截断独立计数，修正 Top-13 被当成 Top-20 的累计错误。
Recall@1percent 保留原作者按查询数取 1% 的口径，并在终端显示具体 Top-K；
Sejong 为 Top-13，这与通常按数据库规模取 1% 的定义不同。
已测出的 Recall@1/5/10/20 不受累计逻辑修正影响；先前 nan F1 和 1percent
输出无效；Spectral 已由用户用相同权重复测，原版还需使用修正后的代码复测
这两项指标。原版 `../src` 保持不变。

## 用户真实 Sejong 短训结果（2026-10-14）

三个阶段各两轮，数据库 1024、查询 256、batch size 1。
权重：`../runs/sejong_spectral_2026-10-14_202056/r2l_mulran/weights/2.pth.tar`。

| 指标 | 原版两轮短训 | Spectral 两轮短训 | 变化（百分点） |
|---|---:|---:|---:|
| Recall@1 | 46.52% | 46.88% | +0.36 |
| Recall@5 | 64.13% | 64.86% | +0.73 |
| Recall@10 | 70.58% | 71.81% | +1.23 |
| Recall@20 | 77.39% | 77.97% | +0.58 |
| Recall@1percent（Top-13） | 待重测 | 73.91% | — |
| 最大 F1 | 待重测 | 0.6493 | — |

这些数据来自用户实际运行日志。Spectral 旧评估输出 F1=nan、1percent=77.97%，
两项旧值无效；修正后复测最大 F1 为 0.6492942453854507，Top-13 Recall 为 73.91%。
原版的 F1 和 Top-13 尚未复测。单次短训仅显示小幅改善，不能证明稳定收益
或单独归因于相位模块。两个模型的跨模态损失定义不同，不直接比较总 loss 大小。

## Word 完整文字方案

`spectral_word` 补齐纯相位表征、原 Transformer 架构的相位 Q/K、统计相干度
替代第二阶段 InfoNCE。41 项 CPU 测试、GPU 合成优化与真实一批 BEV 执行通过。
真实完整版本训练待用户运行。入口及 Word 对照见 [完整版本说明](word-complete-usage.md)。

## MulRan 官方预训练模型复现（2026-10-14）

使用原版 RLPR 网络及 `../pretrained/mulran.pth.tar`，在作者脚本生成的 DCC
和 Riverside 测试划分上实际评估。仅沿用此前 F1 除零/Recall 独立计数修正。
论文表 I 不含 Top-1percent，该项不用于论文比较。

| 测试集 | 来源 | Recall@1 | Recall@5 | Recall@10 | max F1 |
|---|---|---:|---:|---:|---:|
| Riverside | 论文表 I | 64.85% | 78.77% | 84.22% | 0.8057 |
| Riverside | 实际复现 | 65.26% | 79.23% | 84.40% | 0.8108 |
| DCC | 论文表 I | 55.32% | 73.94% | 82.98% | 0.7419 |
| DCC | 实际复现 | 55.32% | 74.47% | 82.45% | 0.7500 |

DCC 数据库 204、查询 188；Riverside 数据库 1558、查询 1724。
两组整体达到接近论文的水平，三项 Recall 最大差距 0.53 个百分点；
不是所有数字逐位一致。差异来源尚未单独定位。

新预处理目录 `../processed/mulran_official`；完整日志与数据准备记录在
`../runs/mulran_official_eval_2026-10-14`。结构化记录见
[官方模型复现结果](verification/mulran-official-pretrained-2026-10-14.json)。

重新评估示例（DCC；Riverside 改 split 即可）：

```bash
cd "/media/rwj/ZX1 2TB/RLPR/src_spectral"
TORCH_HOME="/media/rwj/ZX1 2TB/RLPR/cache/torch" \
RLPR_MULRAN_DATA="/media/rwj/ZX1 2TB/RLPR/processed/mulran_official" \
RLPR_MULRAN_TEST_SPLIT=dcc \
RLPR_CHECKPOINT_PATH="/media/rwj/ZX1 2TB/RLPR/pretrained/mulran.pth.tar" \
python test_cross_modal.py --dataset mulran --model rlpr --no-cross-fusion
```
