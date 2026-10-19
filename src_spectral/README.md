# RLPR: Radar-to-LiDAR Place Recognition via Two-Stage Asymmetric Cross-Modal Alignment for Autonomous Driving

## 本地 Spectral-RLPR 实验版本

本工作区新增频域滤波、相位注意力和相位一致性损失。使用 `--model spectral` 选择新架构，
默认 `--model rlpr` 使用原架构。三阶段短训、评估指令与限制见 [使用说明](docs/spectral-usage.md)。
原版工作区为 `../src`。新架构需要训练自己的权重，不能直接用原作者权重评估。

[Zhangshuo Qi](https://github.com/QiZS-BIT)<sup>1</sup>, [Jingyi Xu](https://github.com/BIT-XJY)<sup>2</sup>, [Luqi Cheng](https://github.com/ChengLuqi)<sup>1</sup>, Shichen Wen<sup>1</sup>, [Guangming Xiong](https://ieeexplore.ieee.org/author/37286205000)<sup>1*</sup>

<sup>1</sup> Beijing Institute of Technology <sup>2</sup> Shanghai Jiao Tong University

[[arxiv]](https://arxiv.org/pdf/2512.18279) [[IEEE Xplore]](https://ieeexplore.ieee.org/abstract/document/11578324)

![image](https://github.com/QiZS-BIT/RLPR/blob/main/assets/intro.png)

**RLPR is a robust radar-to-LiDAR cross-modal place recognition framework** By introducing a dual-stream architecture and a two-stage asymmetric alignment strategy, 
it can effectively bridges the domain gap between radar and LiDAR. The framework is compatible with diverse radar types, including single-chip, scanning, and 
4D radar, enabling reliable cross-modal localization to ensure all-weather autonomy in complex autonomous driving environments.

---

## 1. Installation

Reference environment: Python 3.10, PyTorch 2.1 + CUDA 12.1.

```bash
conda create -n unimprv3 python=3.10 -y
conda activate unimprv3

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


## 2. Datasets

Each dataset uses two roots:

| Variable | Meaning |
|----------|---------|
| `RLPR_<NAME>_RAW` | **Official / raw** sensor data (read by `gen_info.py`) |
| `RLPR_<NAME>_DATA` | **Preprocessed** output (written by `gen_*.py`; used by training & test) |

Default preprocessed root: `~/RLPR_data/<name>/`.

Preprocessed layout (all datasets):

```
<RLPR_*_DATA>/
├── info/          # *.pkl frame metadata
├── index/         # *.npy db / query indices
└── bev/           # LiDAR & radar BEV PNGs
```

Below: **what the code expects on disk** (from `dataset/*/gen_info.py`).

### NuScenes

Full v1.0 **trainval** + **test**. Locations used: `boston-seaport`, `singapore-onenorth`, `singapore-queenstown`, `singapore-hollandvillage`.

**Raw**:

```
<NuScenes>/
├── v1.0-trainval/
├── v1.0-test/
├── maps/
└── samples/          # LIDAR_TOP, RADAR_*, CAM_* under sensor folders
```

### MulRan

Sequences: `Sejong01`, `Sejong02`, `KAIST01`, `KAIST02`, `Riverside01`, `Riverside02`, `DCC01`, `DCC02`.

**Raw** — one folder per sequence:

```
<MulRan>/
└── <Seq>/              # e.g. Sejong01, DCC02
    ├── global_pose.csv
    ├── Ouster/         # LiDAR *.bin
    └── polar/          # radar *.png
```

### Boreas

Sequences: `boreas-2020-12-18-13-44`, `boreas-2021-09-14-20-00`, `boreas-2021-01-26-11-22`.

**Raw** — one folder per drive:

```
<Boreas>/
└── <boreas-...>/       # e.g. boreas-2021-09-14-20-00
    ├── applanix/gps_post_process.csv
    ├── lidar/          # *.bin
    ├── radar/          # *.png
    └── camera_224/     # *.png (required by gen_info, not used in training)
```

### SnailRadar

Sessions: `20231208/data4`, `20240116/data5`, `20240123/data2`, `20240116/data2`, `20240116_eve/data3`.

**Raw** — date / session folders:

```
<Snail-Radar>/
└── <date>/<dataN>/     # e.g. 20231208/data4
    ├── utm50r_T_xt32.txt   # poses
    ├── xt32/               # LiDAR *.pcd
    └── eagleg7/enhanced/   # radar *.pcd
```

### Hercules

Runs: `RiverIsland/01`, `RiverIsland/02`, `SportsComplex/01`, `SportsComplex/02`.

**Raw** — campus / run folders:

```
<Hercules>/
└── <SportsComplex|RiverIsland>/<01|02>/
    ├── Aeva_gt.txt
    ├── Continental_gt.txt
    ├── Aeva/           # LiDAR *.bin
    └── continental/    # radar *.bin
```

---

## 3. Training & evaluation

One shell script per dataset under `scripts/`. Each runs **preprocess → train → eval** unless skipped.

| Script | Stages |
|--------|--------|
| `scripts/nusc.sh` | unimodal L/R → cross-modal → eval |
| `scripts/mulran.sh` | unimodal L/R → cross-modal → eval |
| `scripts/snail.sh` | unimodal L/R (hard mining) → cross-modal → eval |
| `scripts/hercules.sh` | cross-modal (Snail init) → eval |
| `scripts/boreas.sh` | preprocess → eval (MulRan weights) |

**Examples**

```bash
# NuScenes
RLPR_NUSC_RAW=/path/to/nuscenes \
RLPR_NUSC_DATA=/path/to/preprocessed/nusc \
./scripts/nusc.sh

# MulRan
RLPR_MULRAN_RAW=/path/to/MulRan \
RLPR_MULRAN_DATA=/path/to/preprocessed/mulran \
./scripts/mulran.sh

# Snail → Hercules (run Snail first)
RLPR_SNAIL_RAW=/path/to/Snail-Radar \
RLPR_SNAIL_DATA=/path/to/preprocessed/snail \
./scripts/snail.sh

RLPR_HERCULES_RAW=/path/to/Hercules \
RLPR_HERCULES_DATA=/path/to/preprocessed/hercules \
./scripts/hercules.sh

# Boreas (needs MulRan weights from mulran.sh)
RLPR_BOREAS_RAW=/path/to/Boreas \
RLPR_BOREAS_DATA=/path/to/preprocessed/boreas \
./scripts/boreas.sh
```

**Useful options**

| Variable | Effect |
|----------|--------|
| `SKIP_PREP=1` | Skip `gen_info` / `gen_index` / `gen_bev` |
| `SKIP_TRAIN=1` | Evaluation only |
| `RLPR_OUTPUT_ROOT=./outputs` | All checkpoints under `./outputs/` |
| `RLPR_NUSC_TEST_SPLIT=bs\|son\|sq` | NuScenes test split |
| `RLPR_MULRAN_TEST_SPLIT=sejong\|riverside\|dcc` | MulRan test split |
| `RLPR_BOREAS_TEST_SPLIT=bor-clear\|bor-snow` | Boreas test split |

Checkpoints: `$RLPR_OUTPUT_ROOT/<run>/weights/<epoch>.pth.tar`.  
`RLPR_RESUME=0` applies to **training scripts only**; `test_cross_modal.py` always loads weights unless `--zero-shot` is set.

**Pre-trained weights** (unimodal + cross-modal, all datasets) are available [here](https://drive.google.com/drive/folders/1DHZ0n6Kevldvd-bJDfc5vvS9koPWbVad?usp=sharing). Extract to `./outputs/` (same layout as `scripts/*.sh`), then set `SKIP_TRAIN=1` to evaluate without retraining.

**⚠️Note:** The provided pre-trained weights for the Snail-Radar dataset were trained with a minor bug in the original code. While they are fully functional for evaluate, we recommend re-running the training pipeline using the provided scripts to achieve optimal place recognition accuracy.

### Analysis (optional)

Two scripts for paper-style analyses — run **after** training, with paths to unimodal or cross-modal checkpoints:

```bash
# Radar feature separability
python r_feat_separability.py --dataset nusc \
    --pretrained_r ./outputs/r_nusc/weights/5.pth.tar

python r_feat_separability.py --dataset snail \
    --pretrained_r ./outputs/r_snail/weights/5.pth.tar

python r_feat_separability.py --dataset mulran_boreas \
    --pretrained_r_mulran ./outputs/r_mulran/weights/5.pth.tar \
    --pretrained_r_boreas ./outputs/r_mulran/weights/5.pth.tar

# LiDAR–radar BEV conditional entropy (edit dataset import in script if needed)
python l_r_bev_kl.py
```

---

## Citation

```bibtex
@article{qi2026rlpr,
  title={RLPR: Radar-to-LiDAR Place Recognition via Two-Stage Asymmetric Cross-Modal Alignment for Autonomous Driving},
  author={Qi, Zhangshuo and Xu, Jingyi and Cheng, Luqi and Wen, Shichen and Xiong, Guangming},
  journal={IEEE Robotics and Automation Letters},
  year={2026}
}
```
