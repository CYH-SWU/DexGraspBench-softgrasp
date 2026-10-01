"""画一条采集数据的关节角度 / 扭矩曲线。"""
import sys, os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_latest_npz(exp_name=None):
    pattern = (f'output/{exp_name}_shadow/dataset/**/*.npz'
               if exp_name else 'output/**/dataset/**/*.npz')
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"no npz found: {pattern}")
    return files[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz', nargs='?', default=None)
    p.add_argument('--exp', default=None)
    p.add_argument('--outdir', default='figs/collect')
    args = p.parse_args()

    f = args.npz if args.npz else find_latest_npz(args.exp)
    print(f"file: {f}")

    d = np.load(f, allow_pickle=True)
    t, q, tau, ph = d['t'], d['qpos_hand'], d['qfrc_hand'], d['phase']
    bounds = [t[i] for i in range(1, len(ph)) if ph[i] != ph[i-1]]

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j in range(q.shape[1]):
        ax[0].plot(t, q[:, j], lw=0.8, label=f'J{j}' if j < 8 else None)
    ax[0].set_ylabel('joint angle (rad)')
    ax[0].set_title(f'joint angle / torque  |  T={len(t)}  t=[{t[0]:.2f},{t[-1]:.2f}]s')
    ax[0].grid(alpha=0.3); ax[0].legend(loc='upper right', ncol=4, fontsize=7)

    for j in range(tau.shape[1]):
        ax[1].plot(t, tau[:, j], lw=0.8, label=f'J{j}' if j < 8 else None)
    ax[1].set_ylabel('joint torque (N·m)')
    ax[1].set_xlabel('time (s)')
    ax[1].grid(alpha=0.3); ax[1].legend(loc='upper right', ncol=4, fontsize=7)

    for a in ax:
        for b in bounds:
            a.axvline(b, color='k', ls='--', alpha=0.3)

    plt.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    tag = os.path.basename(f).replace('.npz', '')
    parts = f.split(os.sep)
    exp = next((p for p in parts if p.endswith('_shadow')), 'unknown')
    out = os.path.join(args.outdir, f'{exp}__{tag}.png')
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}")


if __name__ == '__main__':
    main()
