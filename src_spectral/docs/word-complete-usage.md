# Word 完整文字方案版本

版本名：`spectral_word`。代码位于现有 `src_spectral`。
原 `src` 和之前的 `spectral` 结果仍保留，便于对照。

## Word 内容对应

| 要求 | 实现 |
|---|---|
| 保留极坐标 BEV | 使用原预处理及 50×225 / 100×100 输入 |
| 浅层卷积后二维 FFT | `word_spectral.py` 的 rFFT，与实输入 FFT 等价 |
| 丢弃测量幅度 | 单位复相位代替输入幅度；保留学习得到的低通频率权重 |
| 动态低通掩码，替换 PCE/Mamba | 相位统计条件化的可学习截止频率；原 PCE 不执行 |
| 原 Transformer 架构的相位 Q/K | `phase_transformer.py` 保留 MHA、FFN、残差、归一化，Q/K 改为圆周相位；encoder 移至 iFFT 前 |
| Radar–LiDAR 交叉注意力 | LiDAR Q、冻结 Radar K/V；训练阶段有成对上下文 |
| iFFT 与 NetVLAD | 实频谱约束后 iFFT，原局部/全局池化及 512 维输出 |
| 相干度替代第二阶段 InfoNCE | 有符号描述子相关度、3×3 复频谱 MSC及地理负样本间隔约束，无 InfoNCE；2026-10-15 修正描述子目标 |
| 非对称对齐 | Radar 参数和 BatchNorm 缓冲区冻结；LiDAR 及交叉模块更新 |
| 相位一致性 | 保留默认 0.1 的圆周相位余弦损失，消除 MSC 对固定相位差的歧义 |

Word 的文字与示例代码对“抛弃幅度”和“替代 InfoNCE”存在冲突。
按用户最后指示以完整文字目标为准。Word 未给出统计估计公式及负样本目标，
当前工程定义：描述子使用保留符号的归一化内积，等价于全频率交叉谱实部之和
除以两侧总能量范数（Parseval）；对单位描述子有 L2² = 2 − 2 × score，
使对齐与检索的排序一致。真实复数空间频谱仍按 3×3 邻域估计 MSC，先平均
交叉/自功率谱，再除法。近邻正样本保留为正，十个地理负样本使用 0.2 间隔。
首轮完整实验使用的旧描述子目标为 32 维窗口、16 维步长的 Welch MSC；
它对特征取反不敏感，现已停用于描述子训练，估计器仍保留供诊断和 SciPy 对照。
参考定义：[SciPy coherence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html)。

总损失为局部描述子相干对齐 + 全局描述子相干对齐 +
0.1 ×（复频谱 MSC 损失 + 相位余弦一致性）。独立/融合监督默认各占 0.5。
这项实现不证明整个网络对任意点云稀疏度不变；相位层的幅度缩放检查仅覆盖
傅里叶分量仍高于 1e-7 有效阈值的情况。识别效果需要真实训练和评估。

## 训练

新结构需要重新训练两支单模态模型。脚本复用原 1024/256 训练索引与 BEV，
按 LiDAR、Radar、跨模态三个阶段运行，默认各 2 轮，输出新目录。

```bash
cd /tmp
bash "/media/rwj/ZX1 2TB/RLPR/src_spectral/scripts/sejong_word_train.sh"
```

轮数由 `RLPR_WORD_EPOCHS` 控制。训练输入由 `RLPR_WORD_INPUT_ROOT` 指定，
默认原频域训练目录 `../runs/sejong_spectral_2026-10-14_202056`。
它只提供索引和 BEV；该目录的旧权重不用于初始化新教师。
已有 output 路径会被拒绝，避免覆盖。启动参数应填合法数值；格式错误会在
终端报错，某些早期参数错误发生在状态文件生成之前。

## 测试

将以下权重路径替换为训练结束时打印的实际路径：

```bash
cd "/media/rwj/ZX1 2TB/RLPR/src_spectral"
TORCH_HOME="/media/rwj/ZX1 2TB/RLPR/cache/torch" \
RLPR_MULRAN_DATA="/media/rwj/ZX1 2TB/RLPR/processed/mulran_sejong" \
RLPR_MULRAN_TEST_SPLIT=sejong \
RLPR_CHECKPOINT_PATH="/media/rwj/ZX1 2TB/RLPR/runs/sejong_word_实际时间/r2l_mulran/weights/2.pth.tar" \
python test_cross_modal.py --dataset mulran --model spectral_word --cross-fusion
```

测试使用独立 Radar 查询与 LiDAR 数据库描述子。没有真实另一模态配对信息时，
不直接使用训练交叉融合路径，避免将正确匹配泄露到检索。
旧 `spectral` 权重继续使用 `--model spectral`；新旧权重不能混载。

## 验证

CPU 新旧回归、SciPy 对照、教师冻结、两种尺寸、描述子独立和 checkpoint
验证通过。GPU 合成优化、真实 Sejong 一批 BEV 前向/反向、100×100 反向通过；
新相位 Transformer 和交叉模块均更新。详见
[GPU 记录](verification/word-complete-gpu-2026-10-14.json)。
恢复冻结教师后 CUDA 描述子与保存前完全一致。未恢复冻结属性时 GPU 后端可
产生约 8.2e-6 的差异；测试入口现已恢复该属性。

以上初始验证只说明代码能执行。之前 45.22% 和 46.88% 对应此前两种结构，
不可当作完整版本成绩。后续完整真实训练结果见下节。

## 完整 MulRan 训练与评估（2026-10-15）

按用户要求，完成 LiDAR 5 轮、Radar 5 轮、跨模态 1 轮。
使用完整 Sejong 训练区域和 KAIST01/02，数据库 16,602、查询 16,568；
batch size 8、seed 0、学习率 5e-5、每轮衰减 0.8。相位损失权重 0.1、
独立/融合监督权重 0.5。每阶段均完成，未使用测试区域或 Riverside/DCC 训练。

LiDAR 每轮平均 loss：0.20305797、0.19150352、0.15094170、0.12155832、0.10698099。
Radar 每轮平均 loss：0.20459646、0.22348140、0.17805886、0.15134342、0.13081431。
跨模态平均 loss：2.2367485。新旧模型的损失不同，不直接比较总 loss。

| 测试集 | 模型 | Recall@1 | Recall@5 | Recall@10 | max F1 |
|---|---|---:|---:|---:|---:|
| Riverside | 官方权重本机复现 | 65.26% | 79.23% | 84.40% | 0.8108 |
| Riverside | Word 完整版 5+5+1 | 54.81% | 72.80% | 78.94% | 0.7125 |
| DCC | 官方权重本机复现 | 55.32% | 74.47% | 82.45% | 0.7500 |
| DCC | Word 完整版 5+5+1 | 52.13% | 71.28% | 80.32% | 0.7059 |

本次完整训练未超过官方权重：Riverside、DCC 的 Recall@1 分别低
10.45、3.19 个百分点。属于一次固定 seed 的实际实验，不证明某个单独模块
造成下降。该次实验使用固定的原损失完成一次训练，没有依据测试成绩选择权重。
随后发现描述子 MSC 的取反歧义，修正实验单独记录于下节；两次结果不能混用。

结果目录：`../runs/mulran_word_full_2026-10-14_232504`。
最终权重：`r2l_mulran/weights/1.pth.tar`；两支初始化权重分别在
`l_mulran/weights/5.pth.tar`、`r_mulran/weights/5.pth.tar`。
完整配置、数据准备、训练日志、测试日志及对照 CSV 均保存在该目录。
结构化摘要见 [完整训练结果](verification/mulran-word-full-2026-10-15.json)。

重测完整训练权重时，使用 `--model spectral_word --cross-fusion`，
`RLPR_MULRAN_DATA` 设为 `../processed/mulran_official`，
`RLPR_MULRAN_TEST_SPLIT` 选 `dcc` 或 `riverside`。

## 描述子对齐修正实验（2026-10-15）

旧 MSC 对 `x` 与 `-x` 给出同样分数，而欧氏检索会区分两者。回归测试先在旧实现
下复现失败，再通过保留符号的相关度修正。CPU 全套 44 项测试通过，覆盖取反、
零描述子有限梯度、教师无梯度及独立检索。新权重使用
`alignment_loss=signed_coherency_v2` 记录训练目标；原损失为
`welch_coherence_v1`，旧权重仍可独立评估。

此次只从上节的 LiDAR、Radar 各第 5 轮权重重新开始跨模态第 1 轮，
不接着旧跨模态权重或优化器训练。数据、随机种子、学习率、batch size、
相位辅助权重及交叉融合权重均不变。这是修正目标的受控实验，不保证超过官方模型。
输出目录：`../runs/mulran_word_signed_2026-10-15_074007`，包含独立日志、
代码差异、训练配置和前后对照。

此次跨模态 1 轮已完成（平均 loss 1.6069736，目标不同，不与旧总 loss 直接比较）。
新权重中 Radar 全部参数及缓冲区与复用的第 5 轮教师完全一致。

| 测试集 | 模型 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | max F1 |
|---|---|---:|---:|---:|---:|---:|
| DCC | 旧描述子 MSC | 52.13% | 71.28% | 80.32% | 84.57% | 0.7059 |
| DCC | 本次有符号相关度 | 55.85% | 70.74% | 81.38% | 86.17% | 0.7603 |
| DCC | 官方权重 | 55.32% | 74.47% | 82.45% | 90.96% | 0.7500 |
| Riverside | 旧描述子 MSC | 54.81% | 72.80% | 78.94% | 85.85% | 0.7125 |
| Riverside | 本次有符号相关度 | 56.32% | 72.16% | 80.05% | 86.37% | 0.7325 |
| Riverside | 官方权重 | 65.26% | 79.23% | 84.40% | 88.75% | 0.8108 |

本次相较旧目标，DCC/Riverside Recall@1 提高 3.72/1.51 个百分点，F1 也提高，
但两组 Recall@5 均略降。DCC 的 Recall@1/F1 略高于官方权重，其他召回指标仍低；
Riverside 的 Recall@1 仍低于官方 8.94 个百分点。不能据此宣称整体超过 RLPR。
这是查看首轮测试后修正目标的探索实验，未额外搜索测试最优轮次或权重。

最终权重：`r2l_mulran/weights/1.pth.tar`。
结构化记录见 [修正实验结果](verification/mulran-word-signed-2026-10-15.json)。

## 单因素消融实验（2026-10-15）

新增 `RLPR_WORD_ALIGNMENT_LOSS`：默认 `signed_coherency_v2`，可选 `infonce`。
InfoNCE 对照保留 Word 网络、频域辅助及融合监督，只替换局部/全局描述子目标，
方向仍为冻结 Radar 查询到可训练 LiDAR，温度 0.07。checkpoint 顶层
`alignment_loss` 记录实际目标；旧 `spectral_config.alignment` 是教师兼容所需的
历史结构字段，不用于选择当前损失。

四组固定设置：

| 名称 | 交叉融合 | 频域辅助权重 | 描述子损失 |
|---|---|---:|---|
| control | 开 | 0.1 | 有符号相关度 |
| no_fusion | 关 | 0.1 | 有符号相关度 |
| no_frequency_aux | 开 | 0 | 有符号相关度 |
| infonce | 开 | 0.1 | 原 InfoNCE |

`no_frequency_aux` 只关闭损失监督，频域网络仍执行。每组复用相同的第 5 轮单模态
权重，重新训练跨模态 1 轮。固定独立的 sampler/worker 随机生成器（seed 0），
避免关闭融合模块改变模型初始化随机数消耗后，连带改变训练顺序。因此 control
也重新训练，历史完整对照保留，不将两种采样设置混用来归因。

先在 Sejong 预留区域验证：数据库 1,505、查询 1,380，与训练无帧重叠，最近
位置距离约 50.98 米。按 Recall@1、F1、预先声明顺序依次打破平局，固定选择后
再统一评估四组 DCC/Riverside，不根据测试成绩重新选择配置。这两组基准在此前
已被查看，本批次仍属于探索实验。

实验目录：`../runs/mulran_word_ablations_2026-10-15_120339`。协议、驱动脚本、
状态、采样校验、各组权重和日志分别保存于该目录。四组已全部完成，均为每组
跨模态 1 轮。48 项 CPU 测试通过，各组 Radar 参数/缓冲区不变，LiDAR 各有
87 个状态张量改变，采样器及 worker 最终随机状态一致。

| 组别 | Sejong 验证 R@1 | DCC R@1 | Riverside R@1 | DCC F1 | Riverside F1 |
|---|---:|---:|---:|---:|---:|
| 本次完整对照 | 72.90% | 53.19% | 56.79% | 0.7395 | 0.7343 |
| 仅关闭交叉融合 | 72.90% | 53.19% | 56.44% | 0.7364 | 0.7321 |
| 仅关闭频域辅助损失 | 72.83% | 55.32% | 57.48% | 0.7563 | 0.7402 |
| 仅换回 InfoNCE | 72.39% | 54.79% | 57.48% | 0.7699 | 0.7586 |
| 官方预训练权重 | — | 55.32% | 65.26% | 0.7500 | 0.8108 |

三组消融均未提高 Sejong 验证 Recall@1。按预设规则，control 与 no_fusion
的 Recall@1 同分，再比较 F1 后，选择仍是 control。选择于 12:39:15 固定，
之后才评估本批次 DCC/Riverside；没有根据测试成绩改选。

正式测试中，关闭频域辅助损失或使用 InfoNCE 的部分指标较本次 control 提高；
Riverside 两者 R@1 均为 57.48%，仍低于官方 7.78 个百分点。InfoNCE 的
Riverside R@5/R@10/R@20 为 74.94/81.38/86.95%，F1 为 0.7586，均高于
本次 control，但还不能认定具有稳定的跨数据集收益。当前不据此替换默认方案。

本次 control 的 DCC/Riverside R@1 为 53.19/56.79%，与历史 signed 实验的
55.85/56.32% 不同：本次固定了独立采样随机流，改变了训练样本顺序及负样本
随机序列。因此以本次 control 做单因素对照；历史成绩继续保留，不混用归因。
差异也提示后续需要多种子复验，单轮、单种子的微小收益不足以支持强结论。

完整 R@1/5/10/20/1% 和 F1 见目录内 `comparison.csv`；包含论文、官方和历史
实验的正式测试总表见 `benchmark_comparison.csv`。
结构化记录：[消融结果](verification/mulran-word-ablations-2026-10-15.json)。

## 延长跨模态训练的轮数对照

轮数优先级已修正为 `--epochs` > `RLPR_EPOCHS` > 数据集阶段默认值。
`--epochs 5` 表示训练到**总计第5轮**；从第1轮续训时再执行4轮。
默认设置仍为单模态各5轮、MulRan跨模态1轮。从已完成轮数续训到相同或更低
目标轮数会报错，避免出现没有更新却显示训练结束的情况。

新检查点保存模型、优化器、学习率调度器和 Python/NumPy/Torch/CUDA、采样器、
DataLoader 启动种子生成器的随机状态（各轮重建 worker）。恢复时检查损失、批大小和运行确定性设置。
旧检查点仍可加载，但缺少随机状态的旧权重不能保证连续训练轨迹一致。

本次增加轮数实验复用原 Word 单模态第5轮权重，固定网络、损失、学习率策略、
数据划分和 batch8，只延长跨模态阶段。每轮在 Sejong 预留区域验证；先锁定
验证 Recall@1 最优轮数（再比较F1、较早轮数），再评估第1轮、选中轮、第5轮
的 DCC/Riverside 成绩，不根据正式测试挑选轮数。

运行完整对照：

```bash
cd "/media/rwj/ZX1 2TB/RLPR/src_spectral"
/mnt/data300/envs/rlpr/bin/python -u scripts/run_mulran_longer_alignment.py
```

驱动自动创建独立实验目录、检查点、协议、日志及最终 CSV。
它设置 `RLPR_SAMPLING_SEED=0`、`RLPR_DETERMINISTIC=1` 和启动前的
`CUBLAS_WORKSPACE_CONFIG=:4096:8`，训练与评估均启用严格确定性检查。
独立实验的第1轮是其自身的对照；不要要求与此前仅固定 cuDNN 的历史训练
逐位一致。严格模式的两次3批次 GPU 短测得到相同权重；54项回归测试通过。

初次历史一致性检查未通过的记录保留于
`../runs/mulran_word_longer_2026-10-15_232554/reproducibility_diagnosis.json`。
其采样随机状态一致，前两个 batch 的 loss 相同，第三个开始存在数值差异。
尚不能重建旧任务的 GPU 内存分配和内核选择，不将该次运行用于轮数选择。
严格对照实验目录：`../runs/mulran_word_longer_2026-10-15_233921`。

五轮及测试已于2026-10-16 00:33完成：

| 跨模态轮数 | Sejong验证R@1 | DCC R@1 | Riverside R@1 | DCC F1 | Riverside F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 72.75% | 53.72% | 55.86% | 0.7336 | 0.7271 |
| 2 | 72.75% | — | — | — | — |
| 3 | 72.46% | — | — | — | — |
| 4（验证集选中） | 73.04% | 53.19% | 52.96% | 0.7222 | 0.7003 |
| 5 | 72.39% | 52.66% | 53.07% | 0.7132 | 0.7005 |

训练loss依次为1.6079664、1.3649797、1.2823361、1.2326968、1.2006087。
验证集最高R@1仅比第1轮增加0.29个百分点，正式测试没有提高；这次不扩展到
10轮。各检查点权重有限，Radar状态不变，实际优化步数及学习率符合协议。
协议、选择和检查点哈希均已核验。结构化记录：
[轮数对照](verification/mulran-word-longer-2026-10-16.json)。

## 跨模态困难负样本对照

这是额外的训练采样实验，默认关闭：`RLPR_CROSS_HARD_NEGATIVES=0`。
设置为5时，每个query使用5个困难负样本、5个随机负样本。按Radar查询描述子
到LiDAR数据库描述子的距离，从原有训练候选集（地理距离>12m）中挖掘，每轮
刷新一次。特征提取只使用训练数据，保持部署时的独立描述子路径。

混合采样使用局部随机流，主NumPy随机流仍消耗原有10个负样本抽样。因此后续
正样本抽样与随机负样本对照一致，避免连带改变正样本。新检查点记录采样设置；
将旧随机采样检查点直接改为困难采样续训会报错。新实验从同一L5/R5权重开始。

```bash
cd "/media/rwj/ZX1 2TB/RLPR/src_spectral"
/mnt/data300/envs/rlpr/bin/python -u scripts/run_mulran_longer_alignment.py \
  --hard-negatives 5 \
  --baseline-run "/media/rwj/ZX1 2TB/RLPR/runs/mulran_word_longer_2026-10-15_233921"
```

同样运行5轮，每轮Sejong验证，先固定选择再做DCC/Riverside测试。
本次实验目录：`../runs/mulran_word_hardneg_2026-10-16_003626`。
66项回归测试通过，独立代码检查未发现阻塞问题。五轮训练和预定测试已完成：

| 轮数 | 随机负样本验证R@1 | 困难负样本验证R@1 | 困难负样本验证F1 |
|---|---:|---:|---:|
| 1 | 72.75% | 72.83% | 0.8524 |
| 2（困难采样选中） | 72.75% | 73.48% | 0.8611 |
| 3 | 72.46% | 72.68% | 0.8562 |
| 4 | 73.04% | 72.83% | 0.8532 |
| 5 | 72.39% | 73.48% | 0.8580 |

第2轮与第5轮验证R@1同分，按预定F1规则选第2轮。为了分离轮数与采样影响，
在第4轮训练期间预先声明匹配轮数分析，最终补测了随机采样第2轮；原基线
报告保持不变，额外结果单独保存。

| 同为跨模态第2轮 | DCC R@1 | Riverside R@1 | DCC F1 | Riverside F1 |
|---|---:|---:|---:|---:|
| 随机负样本 | 54.26% | 55.45% | 0.7402 | 0.7225 |
| 5困难+5随机 | 55.32% | 56.09% | 0.7393 | 0.7310 |
| R@1差值（百分点） | +1.06 | +0.64 | — | — |

困难负样本对相同轮数的R@1有小幅收益，DCC F1略降，不能说所有指标均提高。
这次选中模型与官方权重的DCC R@1同为55.32%，Riverside仍低于官方65.26%。
该结论限于本次单种子探索；各阶段不按正式测试成绩重新挑选模型。

选中权重：
`../runs/mulran_word_hardneg_2026-10-16_003626/r2l_mulran/weights/2.pth.tar`。
训练记录：[困难采样结果](verification/mulran-word-hardneg-2026-10-16.json)；
公平对照：[匹配轮数结果](verification/mulran-word-matched-epoch-2026-10-16.json)。
实验目录内的 `sampling_comparison.csv` 包含两组预定评估，
`matched_epoch_comparison.csv` 包含同轮数直接对照。所有检查点/缓存、协议/选择
哈希已核验，Radar状态未变，每轮采样器/worker状态与基线一致。

### 剩余差距诊断

固定已选中的第2轮模型和官方模型，不再训练或重新选轮数，在相同数据及9m
真值阈值下补测单模态检索。Riverside Recall@1如下：

| 模型 | Radar→LiDAR | Radar→Radar | LiDAR→LiDAR |
|---|---:|---:|---:|
| 官方模型 | 65.26% | 96.06% | 84.22% |
| 当前Word+困难采样 | 56.09% | 90.02% | 77.78% |

DCC上两者Radar→Radar为73.40/72.87%，LiDAR→LiDAR同为75.53%。
Riverside的差距同时出现在单模态表示和跨模态检索。下一步应优先验证保留
幅度或空间信息的独立消融；这些数值本身不能证明具体是哪一个模块导致退步。
新增结构需要重新训练对应单模态分支，不能给冻结的Radar教师拼接未训练模块。

这里的LiDAR→LiDAR使用跨模态训练后的LiDAR分支，不代表L5初始化的成绩；
Radar→Radar使用始终冻结的R5教师。单模态成绩不是跨模态检索的严格上界。
诊断中的Radar→LiDAR与正式测试复核一致。完整结果：
[单模态诊断](verification/mulran-teacher-diagnostic-2026-10-16.json)；
原始日志和脚本位于`../runs/mulran_teacher_diagnostic_2026-10-16_013559`。
# 幅度保留对照（2026-10-16）

## 2026-10-18：按官方代码轮数运行

用户要求将轮数改为官方设置。论文 Implementation Details 未给出 MulRan 的轮数；原始 `src/scripts/registry.py` 使用单模态各5轮、MulRan 跨模态1轮。因此新对照采用**总共5＋5＋1轮**：两组均从相同种子的 torchvision ResNet18 初始化开始，单模态训练5轮，固定使用第5轮权重进行1轮跨模态训练。两组仍使用修改后的 Word 模型，只切换幅度保留开关。原“历史5轮权重＋追加3轮”的微调实验单独保留。

```bash
cd "/media/rwj/ZX1 2TB/RLPR/src_spectral"
python scripts/amplitude_ablation.py --schedule official_code
```

新结果保存在 `runs/mulran_word_amplitude_official_*`。若整轮已保存，使用该实验目录的冻结脚本 `python <实验目录>/driver.py --run-root <实验目录>` 恢复优化器和随机状态；机器重启会丢失尚未保存完一轮的内存进度。状态记录增加文件及目录同步，以降低重启后出现空记录的风险。

`RLPR_WORD_AMPLITUDE=0`（默认）保持现有 Word 纯相位模型与权重配置。`RLPR_WORD_AMPLITUDE=1` 对照 Word 示例中保留幅度的写法：频域 Transformer 输入仍为相位内容，在自注意力输出、训练时交叉融合输出后恢复原始 FFT 幅度。它不包含空间残差，不能直接将旧纯相位权重当作已训练的幅度模型来测试。

受控实验入口（须在 rlpr 环境，使用完整已处理 MulRan 数据）：

```bash
cd "/media/rwj/ZX1 2TB/RLPR/src_spectral"
python scripts/amplitude_ablation.py
```

脚本给纯相位、保留幅度两组使用相同历史 L5/R5 起始权重，各追加训练 LiDAR、Radar 3 轮，再分别训练跨模态 2 轮。新单模态轮次从 1 计，表示追加微调；不恢复旧优化器。每组用 Sejong 验证选模型，全部选定后再测 DCC/Riverside。实验运行时禁止修改源代码，协议和日志保存在新建的 `runs/mulran_word_amplitude_*` 下。中断恢复使用该目录冻结的 `driver.py --run-root <实验目录>`；恢复时核查代码、协议和权重哈希。
# 保留幅度版：交叉融合监督权重对照（2026-10-19）

## 后续：相位辅助损失权重对照

`scripts/phase_weight_ablation.py` 固定此前由Sejong选定的融合监督权重0，比较相位辅助损失权重0.1、0.03、0。所有配置复用同一原始幅度版L5/R5，分别从头进行跨模态第1轮，不从上一轮跨模态模型续训。0.1控制组必须逐字节复现此前的 `fusion_00` 模型，否则停止。该权重同时缩放相位一致性与频谱相干性两项辅助损失；设0不是删除频域表示模块，也不是关闭融合前向计算。

命令：`/mnt/data300/envs/rlpr/bin/python -u scripts/phase_weight_ablation.py`。恢复仍使用新实验目录内冻结的 `driver.py --run-root <实验目录>`，确认无存活的同实验进程后运行。先完成三组Sejong验证，按R@1、F1、配置顺序0.1/0.03/0锁定，再测试DCC/Riverside。输出使用独立 `runs/mulran_amplitude_phase_<时间>`，不会覆盖之前的模型或冻结脚本。

相位辅助对照已完成：`runs/mulran_amplitude_phase_2026-10-19_130536`，13:44结束。权重0.1/0.03/0的Sejong R@1为75.00/74.06/74.13%，DCC R@1为55.85/54.79/54.79%，Riverside R@1为63.34/62.65/62.41%。按验证集仍选择0.1；未找到优于上一轮选定模型的R@1配置，不更新到较弱权重。控制组与上一轮模型逐字节复现；三组权重、实际挖掘负样本、采样流、冻结教师、源码/协议/选择哈希、运行环境均已核验；89项回归通过。报告：`reports/2026-10-19/RLPR相位辅助损失消融结果_2026-10-19.pdf`。

## 融合监督权重实验与结果

在完成的幅度版 L5/R5 单模态权重上，分别用融合监督权重0.5、0.1、0训练跨模态1轮。其他设置保持一致；权重0时仍保留融合前向计算，因此这里只消融融合监督权重，不等同于移除整个融合模块。首先重跑0.5控制组，要求模型张量字节与此前的幅度版跨模态第1轮完全一致，否则停止后续实验。

启动：`/mnt/data300/envs/rlpr/bin/python -u scripts/fusion_weight_ablation.py`。恢复：使用新实验目录内的冻结脚本 `driver.py --run-root <实验目录>`；必须确认没有存活的同一实验进程。每轮保存后可恢复；未完成且未保存的一轮需要重跑。

选模规则预先固定：三组全部完成Sejong验证后，依次按Recall@1、F1、配置顺序0.5/0.1/0选择并锁定。之后才测试DCC和Riverside，不按测试结果改选。所有权重、日志和协议写入新的 `runs/mulran_amplitude_fusion_<时间>`；不会覆盖之前62.82%的模型。本轮仍为单种子探索，不能保证超过论文。

已完成实验：`runs/mulran_amplitude_fusion_2026-10-19_122008`，2026-10-19 12:58结束。Sejong选择权重0（R@1 75.00%），其DCC/Riverside R@1为55.85%/63.34%，F1为0.7437/0.7887；相对权重0.5的R@1提升0.53/0.52个百分点。DCC只多命中1个查询，不能据此宣称稳定优势；Riverside仍未超过论文。已核验控制组张量字节复现、三组实际困难负样本数组一致、冻结教师、检查点与协议/源码/选模哈希、运行环境一致。模型：该实验的 `fusion_00/weights/1.pth.tar`；最终报告：`reports/2026-10-19/RLPR融合权重消融结果_2026-10-19.pdf`。只完成融合监督权重消融，未同时改变其他损失或轮数。

# 对齐损失对照（2026-10-19）

`scripts/alignment_loss_ablation.py` 在当前验证选定的幅度版、融合监督0、相位辅助0.1设置上，对比现有 `signed_coherency_v2` 与 `infonce`，两组各从同一原始L5/R5开始跨模态第1轮。控制组必须逐字节复现上一轮选定的 `phase_01` 模型。不会修改网络、训练器或损失实现，不从跨模态模型额外续训。

命令：`/mnt/data300/envs/rlpr/bin/python -u scripts/alignment_loss_ablation.py`。输出为独立 `runs/mulran_amplitude_alignment_<时间>`；恢复用该目录冻结的 `driver.py --run-root <实验目录>`。先完成两组Sejong验证，再按R@1、F1、完全平分时优先控制组的规则锁定，之后测试DCC/Riverside。

InfoNCE沿用现有Word实现：局部和全局分别计算，温度0.07，每个元组内12个同步跨模态配对作为独立实例；附近正样本也作为另一个实例进入softmax。这轮不新增多正样本掩码，因此比较的是现有完整目标选项，不仅是相同正负规则下的公式形状。

对齐损失对照已完成：`runs/mulran_amplitude_alignment_2026-10-19_135238`，14:19结束。Sejong R@1由控制组75.00%变为InfoNCE75.14%，F1由0.8662变为0.8685，因此按预定规则选定InfoNCE。但DCC/Riverside R@1为51.06/62.35%，低于控制组55.85/63.34%；F1为0.7788/0.8011，高于控制组0.7437/0.7887。没有实现提高R@1的目标，不能称为整体优于RLPR。原选择记录保持InfoNCE，旧控制组和所有历史权重继续保留，不根据测试重写选模。93项回归与最终权重/教师/负样本/采样/哈希/环境核验通过；完整报告：`reports/2026-10-19/RLPR对齐损失消融结果_2026-10-19.pdf`。

