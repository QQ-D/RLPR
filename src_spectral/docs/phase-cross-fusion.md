# Word 模块三：Radar–LiDAR 相位交叉注意力

用户要求先按 `../版本1-1-饶.docx` 补齐跨模态交叉融合。

## 数据流与实施范围

采用 Word 的非对称方向：LiDAR 相位生成 Q，冻结的 radar 相位生成 K，
radar 复数频谱生成 V。交叉注意力更新 LiDAR 频谱，随后进行实频谱对称约束、
iFFT、原骨干与池化。radar 单模态教师参数及 BatchNorm 统计保持冻结。

第二阶段训练有同步成对数据，使用交叉融合监督。测试只用 radar 查询与独立
LiDAR 数据库描述子；不将查询对应的真实 LiDAR 或数据库对应的 radar 作为
测试融合上下文。Word 没有指定测试配对策略，本版不额外增加候选重排序网络。

训练保留 Word 最后的公式：`InfoNCE_local + InfoNCE_global + 0.1 * phase_loss`。
每项监督同时覆盖独立 LiDAR 表征与交叉融合的 LiDAR 表征，默认各占 0.5；
这使独立检索路径也获得监督。相位损失降低低能量频率权重。radar 教师停止梯度。
融合权重通过 `RLPR_CROSS_FUSION_WEIGHT` 调整，范围 [0,1]。

该模块新增于 FFT 幅度滤波与 iFFT 之间。沿用已训练的单模态分支及其频域
自注意力，新增真正的 radar→LiDAR 交叉注意力；不改变 BEV 预处理。
这不是双向传感器融合：方向与 Word 的 radar anchor / LiDAR student 一致。

## 实施计划

1. 先写测试：另一模态改变融合输出、radar 停止梯度、独立评估路径不依赖
   配对 LiDAR、两种输入尺寸、旧单模态权重可初始化而新模块保持可训练。
2. 加入 `PhaseCrossAttention`，输出残差融合的 LiDAR 复数频谱。
3. 拆分频域准备/描述子解码，第二阶段训练增加融合辅助输出；保持原四输出接口。
4. 按上述公式接入独立/融合两种表征的损失，保存融合配置并校验完整 checkpoint。
5. 增加训练和测试 `--cross-fusion` 开关；默认关闭，兼容现有 46.88% 权重。
6. 新建短训入口，复用已完成的两支单模态权重及 1024/256 索引，仅重训跨模态
   第二阶段两轮，使用新的结果目录。
7. 验证 CPU 回归、GPU 反向及权重加载/保存。真实训练与评估由用户运行。

不补建备份，不修改原版 `../src`，不覆盖已测结果。融合收益尚待实验验证。

## 运行

复用上次 Spectral 的 LiDAR/radar 第 2 轮权重和 1024/256 训练索引：

```bash
bash "/media/rwj/ZX1 2TB/RLPR/src_spectral/scripts/sejong_cross_fusion_train.sh"
```

默认只训练跨模态第二阶段两轮。结果写入新的
`../runs/sejong_cross_fusion_<北京时间>/`，终端最后打印权重路径。
`RLPR_CROSS_EPOCHS` 调整轮数，`RLPR_CROSS_INIT_ROOT` 指定已有单模态结果目录；
已有目录包含的两支权重、BEV 和索引必须对应同一次实验。

评估时沿用 Sejong 的数据路径、测试划分与环境设置，将
`RLPR_CHECKPOINT_PATH` 改为新权重，运行：

```bash
python test_cross_modal.py --dataset mulran --model spectral --cross-fusion
```

`--cross-fusion` 选择匹配的新 checkpoint 结构，测试模式不会输入另一模态的
真实配对数据做融合。旧 46.88% 权重仍不加此参数；也可显式 `--no-cross-fusion`。
原来的三阶段短训入口可配合 `RLPR_SPECTRAL_CROSS_FUSION=1` 从头训练三阶段。

## 已完成的验证

2026-10-14：27 项 CPU 单元测试通过。独立审查使用实际 Sejong 单模态权重检查了
全部交叉注意力参数的梯度、radar 参数与缓冲区冻结、独立/融合损失的加权公式、
双向描述子独立性及完整 checkpoint 保存/加载，未发现重大问题。

RTX 4060 Laptop GPU：以已有单模态权重初始化，合成数据第二阶段两轮通过；
新交叉模块参数更新、冻结教师与 checkpoint 保存/加载通过；100x100 输入路径
前向/反向通过。摘要见 [GPU 记录](verification/phase-cross-fusion-gpu-2026-10-14.json)。
以上初始检查属于执行验证。用户随后已完成真实 Sejong 训练和评估，见下文。

审查建议的测试补充：把 reverse independence 和完整 roundtrip 的检查持久化为
额外单元用例。这两项已经由独立审查/GPU 检查验证，当前未追加重复的单元用例。

## 用户真实 Sejong 结果（2026-10-14）

第二阶段两轮，原训练索引 1024/256；权重：
`../runs/sejong_cross_fusion_2026-10-14_212433/r2l_mulran/weights/2.pth.tar`。
训练 loss：1.7887907 -> 1.0533565。
Recall@1/5/10/20：45.22%、64.57%、71.52%、77.39%；
Top-13 Recall：73.62%；max F1：0.6351842241826673。
相比之前频域版本 Recall@1 46.88%、F1 0.6493，本次交叉融合短训没有改善。

该结果对应保留测量幅度与 InfoNCE 的旧交叉融合版本。
后续完成的纯相位/相位 Transformer/统计相干度版本为 `spectral_word`，
参见 [完整版本说明](word-complete-usage.md)，真实成绩待训练后测量。
