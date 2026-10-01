"""从采集的 npz 中导出 4 路相机图像到 figs 目录。

用法:
    python scripts/export_cameras.py --exp collect_test
    python scripts/export_cameras.py <path/to/xxx.npz>
"""
import os, sys, glob, argparse
import numpy as np
from PIL import Image
import imageio

CAMS = ['front', 'right', 'back', 'left']

def find_latest(exp_name):
    files = sorted(glob.glob(
        f'output/{exp_name}_shadow/dataset/**/*.npz',
        recursive=True))
    if not files:
        raise SystemExit(f"no npz found for exp={exp_name}")
    return files[-1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz', nargs='?', default=None)
    p.add_argument('--exp', default='collect_test')
    p.add_argument('--outdir', default=None)
    p.add_argument('--num-key-frames', type=int, default=5,
                   help='保存几个关键时刻的4相机拼图')
    p.add_argument('--gif', action='store_true',
                   help='是否生成 GIF（默认生成）')
    args = p.parse_args()

    f = args.npz if args.npz else find_latest(args.exp)
    print(f"file: {f}")

    d = np.load(f, allow_pickle=True)
    T = len(d['t'])

    # 输出目录
    if args.outdir is None:
        tag = os.path.basename(f).replace('.npz', '')
        parts = f.split(os.sep)
        exp = next((p for p in parts if p.endswith('_shadow')), 'unknown')
        args.outdir = f'figs/camera/{exp}__{tag}'
    os.makedirs(args.outdir, exist_ok=True)
    print(f"outdir: {args.outdir}")

    # ---------- 1. 保存关键时刻的 4 路拼图 ----------
    idxs = np.linspace(0, T-1, args.num_key_frames, dtype=int)
    for rank, idx in enumerate(idxs):
        imgs = [d[f'image_cam_{n}'][idx] for n in CAMS]
        combined = np.concatenate(imgs, axis=1)     # (H, 4*W, 3)
        out = os.path.join(args.outdir, f'frame_{rank:02d}_idx{idx:03d}.png')
        Image.fromarray(combined).save(out)
    print(f"saved {len(idxs)} key frames")

    # ---------- 2. 每路相机单独保存一张中间帧 ----------
    idx_mid = T // 2
    for n in CAMS:
        Image.fromarray(d[f'image_cam_{n}'][idx_mid]).save(
            os.path.join(args.outdir, f'single_{n}_idx{idx_mid:03d}.png'))
    print(f"saved 4 single-cam frames")

    # ---------- 3. 生成 4 路拼图 GIF ----------
    if args.gif:
        step = max(1, T // 30)      # 约 30 帧
        frames = []
        for idx in range(0, T, step):
            imgs = [d[f'image_cam_{n}'][idx] for n in CAMS]
            frames.append(np.concatenate(imgs, axis=1))
        gif_path = os.path.join(args.outdir, '4cam_seq.gif')
        imageio.mimsave(gif_path, frames, fps=5, loop=0)
        print(f"saved gif: {gif_path} ({len(frames)} frames)")

    # ---------- 4. 写一个说明 ----------
    with open(os.path.join(args.outdir, 'README.txt'), 'w') as fp:
        fp.write(f"source: {f}\n")
        fp.write(f"T = {T}\n")
        fp.write(f"t range = [{d['t'][0]:.4f}, {d['t'][-1]:.4f}]\n")
        fp.write(f"cams = {CAMS}\n")
        fp.write(f"key frame idxs = {idxs.tolist()}\n")
    print("done")

if __name__ == '__main__':
    main()
