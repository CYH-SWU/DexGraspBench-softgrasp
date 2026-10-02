"""从采集的 npz 中导出 wrist_cam 图像到 figs 目录。

用法:
    python scripts/export_wrist_cam.py --exp collect_test
    python scripts/export_wrist_cam.py <path/to/xxx.npz>
"""
import os, sys, glob, argparse
import numpy as np
from PIL import Image
import imageio

CAM = 'wrist_cam'

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
    p.add_argument('--num-key-frames', type=int, default=6)
    p.add_argument('--gif-fps', type=int, default=6)
    args = p.parse_args()

    f = args.npz if args.npz else find_latest(args.exp)
    print(f"file: {f}")

    d = np.load(f, allow_pickle=True)
    T = len(d['t'])
    img = d[f'image_{CAM}']
    H, W = img.shape[1], img.shape[2]
    print(f"T = {T}, image shape = {img.shape}")

    if args.outdir is None:
        tag = os.path.basename(f).replace('.npz', '')
        parts = f.split(os.sep)
        exp = next((p for p in parts if p.endswith('_shadow')), 'unknown')
        args.outdir = f'figs/camera/{exp}__{tag}__{CAM}'
    os.makedirs(args.outdir, exist_ok=True)
    print(f"outdir: {args.outdir}")

    idxs = np.linspace(0, T-1, args.num_key_frames, dtype=int)
    for rank, idx in enumerate(idxs):
        out = os.path.join(args.outdir, f'frame_{rank:02d}_idx{idx:03d}.png')
        Image.fromarray(img[idx]).save(out)
    print(f"saved {len(idxs)} key frames")

    idx_mid = T // 2
    Image.fromarray(img[idx_mid]).save(
        os.path.join(args.outdir, f'single_idx{idx_mid:03d}.png'))
    print(f"saved single mid frame")

    step = max(1, T // 60)
    frames = [img[i] for i in range(0, T, step)]
    gif_path = os.path.join(args.outdir, 'seq.gif')
    imageio.mimsave(gif_path, frames, fps=args.gif_fps, loop=0)
    print(f"saved gif: {gif_path} ({len(frames)} frames)")

    with open(os.path.join(args.outdir, 'README.txt'), 'w') as fp:
        fp.write(f"source: {f}\n")
        fp.write(f"camera: {CAM}\n")
        fp.write(f"resolution: {H}x{W}\n")
        fp.write(f"T = {T}\n")
        fp.write(f"t range = [{d['t'][0]:.4f}, {d['t'][-1]:.4f}]\n")
        fp.write(f"key frame idxs = {idxs.tolist()}\n")
    print("done")

if __name__ == '__main__':
    main()

