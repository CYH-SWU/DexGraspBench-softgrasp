import os
import numpy as np


class GraspRecorder:
    def __init__(self, hand_qposadr, hand_dofadr, sample_hz=30, cam_names=None):
        self.hand_qposadr = np.asarray(hand_qposadr)
        self.hand_dofadr = np.asarray(hand_dofadr)
        self.sample_dt = 1.0 / float(sample_hz)
        self.cam_names = cam_names or []

        # ★ 录制状态
        self.recording = False
        self.last_t = -1e9

        self.buf = {
            "t": [], "s": [], "phase": [],
            "qpos_hand": [], "qfrc_hand": [],
        }
        self.images = {name: [] for name in self.cam_names}

    def start(self):
        self.recording = True
        self.last_t = -1e9
        print(f"[Recorder] START  recording")

    def stop(self):
        self.recording = False
        print(f"[Recorder] STOP   recording  (buffered {len(self.buf['t'])} frames)")

    def should_sample(self, mj_ho):
        if not self.recording:
            return False
        t_global = float(mj_ho.data.time)
        return t_global - self.last_t >= self.sample_dt - 1e-9

    def __call__(self, t, s, mj_ho, phase_name="", frames=None):
        if not self.recording:
            return
        t_global = float(mj_ho.data.time)
        if t_global - self.last_t < self.sample_dt - 1e-9:
            return
        self.last_t = t_global

        self.buf["t"].append(t_global)
        self.buf["s"].append(float(s))
        self.buf["phase"].append(phase_name)
        self.buf["qpos_hand"].append(mj_ho.data.qpos[self.hand_qposadr].copy())
        self.buf["qfrc_hand"].append(mj_ho.data.qfrc_actuator[self.hand_dofadr].copy())

        if frames is not None:
            for cam_name in self.cam_names:
                if cam_name in frames:
                    self.images[cam_name].append(frames[cam_name])

    def to_numpy(self):
        out = {}
        for k, v in self.buf.items():
            if k == "phase":
                out[k] = np.array(v, dtype=object)
            else:
                out[k] = np.asarray(v, dtype=np.float32)
        for cam_name, frames in self.images.items():
            if len(frames) > 0:
                out[f"image_{cam_name}"] = np.asarray(frames, dtype=np.uint8)
        return out

    def save(self, path, extra_meta=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = self.to_numpy()
        if extra_meta is not None:
            data["meta"] = np.array([extra_meta], dtype=object)
        np.savez_compressed(path, **data)
        return path