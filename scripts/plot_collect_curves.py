"""绘制采集数据的关节角度 / 扭矩曲线，保存到 figs/collect/。

用法:
    # 1) 指定具体文件
    python scripts/plot_collect_curves.py output/collect_test_shadow/dataset/.../86.npz

    # 2) 指定实验，只画该实验下最后一条
    python scripts/plot_collect_curves.py --exp collect_test

    # 3) 不指定，画所有实验里最后一条
    python scripts/plot_collect_curves.py

"""
import os
import sys
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_latest(exp_name):
    """在 output/<exp>_shadow/dataset/ 下找最后一条 npz。"""
    files = sorted(glob.glob(
        f'output/{exp_name}_shadow/dataset/**/*.npz',
        recursive=True))
    if not files:
        raise SystemExit(f"no npz found for exp={exp_name}")
    return files[-1]


def find_latest_all():
    """在所有 output 实验里找最后一条 npz。"""
    files = sorted(glob.glob(
        'output/**/dataset/**/*.npz', recursive=True))
    if not files:
        raise SystemExit("no npz found under output/")
    return files[-1]


def plot_one(f, outdir, dpi=120, show_legend=True):
    print(f"file: {f}")
    d = np.load(f, allow_pickle=True)

    t = d['t']
    q = d['qpos_hand']
    tau = d['qfrc_hand']
    ph = d['phase']

    bounds = [t[i] for i in range(1, len(ph)) if ph[i] != ph[i-1]]

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)


    for j in range(q.shape[1]):
        ax[0].plot(t, q[:, j], lw=0.8,
                   label=f'J{j}' if (show_legend and j < 8) else None)
    ax[0].set_ylabel('joint angle (rad)')
    ax[0].set_title(
        f'joint angle / torque  |  T={len(t)}  '
        f't=[{t[0]:.2f}, {t[-1]:.2f}]s  '
        f'({os.path.basename(f)})'
    )
    ax[0].grid(alpha=0.3)
    if show_legend:
        ax[0].legend(loc='upper right', ncol=4, fontsize=7)

    for j in range(tau.shape[1]):
        ax[1].plot(t, tau[:, j], lw=0.8,
                   label=f'J{j}' if (show_legend and j < 8) else None)
    ax[1].set_ylabel('joint torque (N·m)')
    ax[1].set_xlabel('time (s)')
    ax[1].grid(alpha=0.3)
    if show_legend:
        ax[1].legend(loc='upper right', ncol=4, fontsize=7)

    for a in ax:
        for b in bounds:
            a.axvline(b, color='k', ls='--', alpha=0.3)

    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    tag = os.path.basename(f).replace('.npz', '')
    parts = f.split(os.sep)
    exp = next((p for p in parts if p.endswith('_shadow')), 'unknown')
    out = os.path.join(outdir, f'{exp}__{tag}.png')
    plt.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz', nargs='?', default=None,
                   help='具体 npz 路径（可选）')
    p.add_argument('--exp', default=None,
                   help='实验名，画该实验下最后一条')
    p.add_argument('--outdir', default='figs/collect',
                   help='输出目录（默认 figs/collect）')
    p.add_argument('--dpi', type=int, default=120,
                   help='图片 DPI（默认 120）')
    p.add_argument('--no-legend', action='store_true',
                   help='不画图例')
    args = p.parse_args()

    # 决定输入
    if args.npz:
        f = args.npz
    elif args.exp:
        f = find_latest(args.exp)
    else:
        f = find_latest_all()

    plot_one(f, args.outdir, dpi=args.dpi,
             show_legend=not args.no_legend)


if __name__ == '__main__':
    main()
