#!/usr/bin/env python
"""从采集数据 + 评测结果生成 il_dataset 的 index.json。

用法:
    # 单个实验
    python scripts/build_index.py --exp collect_v1

    # 多个实验合并
    python scripts/build_index.py --exp collect_v1 --exp collect_v2

    # 只要成功的
    python scripts/build_index.py --exp collect_v1 --success-only

    # 自定义输出
    python scripts/build_index.py --exp collect_v1 --out ~/my_project/data/index.json

    # 自定义 train/val 比例
    python scripts/build_index.py --exp collect_v1 --train-ratio 0.8

目录结构假设（由 task=eval setting=collect 生成）:
    output/<exp>_shadow/
    ├── dataset/<obj>/<pose>/<scale>/<idx>.npz     ← 采集数据
    └── succgrasp/<obj>/<pose>/<scale>/<idx>.npy   ← 评测成功的标记
"""
import os
import sys
import json
import glob
import argparse
from datetime import datetime

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Build index.json for BC training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--exp', action='append', required=True,
                   help='实验名（可多次指定）')
    p.add_argument('--out', default='~/my_project/data/index.json',
                   help='输出路径（默认 ~/my_project/data/index.json）')
    p.add_argument('--success-only', action='store_true',
                   help='只保留成功的 episode')
    p.add_argument('--train-ratio', type=float, default=0.7,
                   help='按物体划分的 train 比例（默认 0.7）')
    p.add_argument('--camera', default='wrist_cam',
                   help='图像字段名（默认 wrist_cam）')
    p.add_argument('--quiet', action='store_true',
                   help='不打印每条进度')
    return p.parse_args()


def scan_one_npz(npz_path, succ_dir, camera):
    """读一条 npz，返回 episode dict。"""
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)

    if 't' not in keys or 'qpos_hand' not in keys:
        return None
    img_key = f'image_{camera}'
    if img_key not in keys:
        return None

    parts = npz_path.replace('\\', '/').split('/')
    try:
        idx = int(parts[-1].replace('.npz', ''))
        scale = parts[-2]
        pose = parts[-3]
        obj = parts[-4]
    except (ValueError, IndexError):
        return None

    succ_path = npz_path.replace('dataset', 'succgrasp').replace('.npz', '.npy')
    succ_path = os.path.join(
        succ_dir, obj, pose, scale, f"{idx}.npy"
    )
    success = os.path.exists(succ_path)

    T = len(d['t'])
    img_shape = d[img_key].shape
    qpos_shape = d['qpos_hand'].shape

    return {
        'npz': os.path.abspath(npz_path),
        'obj': obj,
        'pose': pose,
        'scale': scale,
        'idx': idx,
        'T': int(T),
        'success': bool(success),
        'image_key': img_key,
        'image_shape': list(img_shape),
        'qpos_shape': list(qpos_shape),
        't_start': float(d['t'][0]),
        't_end': float(d['t'][-1]),
    }


def main():
    args = parse_args()

    all_episodes = []
    stats_by_exp = {}

    for exp in args.exp:
        root = f'output/{exp}_shadow'
        data_dir = os.path.join(root, 'dataset')
        succ_dir = os.path.join(root, 'succgrasp')

        if not os.path.isdir(data_dir):
            print(f"[WARN] 跳过 {exp}: 找不到 {data_dir}")
            continue

        npz_files = sorted(glob.glob(
            os.path.join(data_dir, '**', '*.npz'), recursive=True))
        print(f"[{exp}] 扫描 {len(npz_files)} 个 npz")

        n_ok = 0
        n_succ = 0
        n_skip = 0
        for i, npz in enumerate(npz_files):
            ep = scan_one_npz(npz, succ_dir, args.camera)
            if ep is None:
                n_skip += 1
                continue
            ep['exp'] = exp
            all_episodes.append(ep)
            n_ok += 1
            if ep['success']:
                n_succ += 1
            if not args.quiet and (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(npz_files)} ...")

        stats_by_exp[exp] = {
            'total': n_ok,
            'success': n_succ,
            'skipped': n_skip,
        }
        print(f"[{exp}] OK={n_ok}  success={n_succ}  skipped={n_skip}")

    if len(all_episodes) == 0:
        print("[FAIL] 没有任何有效 episode")
        return 1

    if args.success_only:
        before = len(all_episodes)
        all_episodes = [e for e in all_episodes if e['success']]
        print(f"success-only: {before} → {len(all_episodes)}")

    objs = sorted(set(e['obj'] for e in all_episodes))
    n_objs = len(objs)
    n_train = max(1, int(round(args.train_ratio * n_objs)))
    train_objs = set(objs[:n_train])
    val_objs = set(objs[n_train:])

    for e in all_episodes:
        e['split'] = 'train' if e['obj'] in train_objs else 'val'

    n_train_eps = sum(1 for e in all_episodes if e['split'] == 'train')
    n_val_eps = sum(1 for e in all_episodes if e['split'] == 'val')
    n_train_frames = sum(e['T'] - 1 for e in all_episodes if e['split'] == 'train')
    n_val_frames = sum(e['T'] - 1 for e in all_episodes if e['split'] == 'val')
    total_frames = sum(e['T'] - 1 for e in all_episodes)

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    meta = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'camera': args.camera,
        'experiments': args.exp,
        'success_only': args.success_only,
        'train_ratio': args.train_ratio,
        'stats_by_exp': stats_by_exp,
        'n_episodes': len(all_episodes),
        'n_train_episodes': n_train_eps,
        'n_val_episodes': n_val_eps,
        'n_train_frames': n_train_frames,
        'n_val_frames': n_val_frames,
        'n_total_frames': total_frames,
        'n_objects': n_objs,
        'n_train_objects': len(train_objs),
        'n_val_objects': len(val_objs),
    }

    payload = {
        'meta': meta,
        'episodes': all_episodes,
    }

    with open(out_path, 'w') as fp:
        json.dump(payload, fp, indent=2)

    print()
    print("=" * 60)
    print(f"wrote {out_path}")
    print("=" * 60)
    print(f"  episodes    : {len(all_episodes)}")
    print(f"  objects     : {n_objs}  "
          f"(train {len(train_objs)}, val {len(val_objs)})")
    print(f"  train       : {n_train_eps} episodes, "
          f"{n_train_frames} samples")
    print(f"  val         : {n_val_eps} episodes, "
          f"{n_val_frames} samples")
    print(f"  total       : {total_frames} samples")
    print(f"  camera      : {args.camera}")
    print(f"  success_only: {args.success_only}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
