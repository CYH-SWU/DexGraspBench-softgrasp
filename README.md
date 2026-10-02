# SoftGrasp MuJoCo 复现仓库

基于 DexGraspBench 与 BODex 数据集，在 MuJoCo 中复现 SoftGrasp: 基于多模态模仿学习的灵巧手自适应抓取方法 的**仿真数据采集**。

## 项目结构

```bash
~/my_project/
├── DexGraspBench/                # 本仓库
└── bodex_data/                   # 数据源（需自行下载）
    ├── DGN_2k/                   # 物体资产
    │   ├── processed_data/
    │   ├── scene_cfg/
    │   └── valid_split/
    └── bodex_shadow/
        └── succ_collect/         # 成功抓取数据（9588 个 .npy，2397 个物体）
            └── <物体名>/floating/scale0XX.npy
```

## 首次配置
```bash
# 1. 软链物体资产到仓库
mkdir -p assets/object
ln -sfn ~/my_project/bodex_data/DGN_2k assets/object/DGN_2k

# 2. 转换抓取数据（一次性，约 10-30 分钟）
python src/main.py task=format exp_name=debug \
    task.data_name=Learning \
    task.data_path=/home/$USER/my_project/bodex_data/bodex_shadow/succ_collect \
    n_worker=16

# 3. 建采集目录 + 软链数据源
cd output
mkdir -p collect_data_shadow
ln -sfn ../debug_shadow/graspdata collect_data_shadow/graspdata
cd ..
```

## 具体使用

渲染画面但不采集数据
```bash
python src/main.py task=eval exp_name=collect_data \
    setting=collect \
    task.debug_viewer=True \
    task.max_num=1
```

不开启渲染、纯采集数据
```bash
python src/main.py task=eval exp_name=collect_data \
    setting=collect \
    task.debug_render=True \
    task.collect_data=True \
    task.max_num=100
```

## 脚本使用简介

本仓库在 DexGraspBench 基础上扩展了多模态数据可视化脚本。
画关节角度 / 扭矩曲线
```bash
python scripts/plot_collect_curves.py --exp collect_data
```

导出相机图像
```bash
python scripts/export_wrist_cam.py --exp collect_data
```

## 数据可视化图像

**关节角度 / 扭矩曲线**（10 秒完整抓取过程）：

![joint angle and torque curves](docs/collect_curve.png)

**手腕相机视角**（抓取全过程）：

![wrist camera sequence](docs/wrist_cam_seq.gif)
        