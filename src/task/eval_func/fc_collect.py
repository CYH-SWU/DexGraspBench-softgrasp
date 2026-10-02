import os
import sys
import logging

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from task.eval_func.base import BaseEval
from util.recorder import GraspRecorder

DUR_OPEN_TO_PRE      = 2.0
DUR_PRE_TO_GRASP     = 4.0
DUR_GRASP_TO_SQUEEZE = 2.0
DUR_HOLD             = 2.0

CAM_NAMES = ["wrist_cam"]


class collectMocapEval(BaseEval):
    def __init__(self, input_npy_path, configs):
        super().__init__(input_npy_path, configs)

        self.hand_qposadr, self.hand_dofadr = self._find_hand_indices()

        # ★ 只有在 collect_data=True 时才创建 recorder
        collect_data = getattr(configs.task, "collect_data", False)
        if collect_data and not configs.task.debug_render:
            raise ValueError(
                "collect_data=True 需要同时开启 debug_render=True"
            )

        if collect_data:
            sample_hz = getattr(configs.task, "sample_hz", 30)
            self.recorder = GraspRecorder(
                self.hand_qposadr, self.hand_dofadr,
                sample_hz=sample_hz,
                cam_names=CAM_NAMES,
            )
            logging.info(
                f"  [collect] recorder enabled "
                f"(sample_hz={sample_hz}, cams={CAM_NAMES})"
            )
        else:
            self.recorder = None
            logging.info("  [collect] recorder disabled (collect_data=False)")

    def _find_hand_indices(self):
        model = self.mj_ho.model
        qpos_adr, dof_adr = [], []
        for jid in range(model.njnt):
            jname = model.joint(jid).name or ""
            jtype = model.jnt_type[jid]
            if "obj" in jname.lower():
                continue
            if jtype == 0:
                continue
            qpos_adr.append(model.jnt_qposadr[jid])
            dof_adr.append(model.jnt_dofadr[jid])
        return np.array(qpos_adr), np.array(dof_adr)

    def _simulate_under_extforce_details(self, pre_obj_qpos):
        d = self.grasp_data
        pregrasp = d["pregrasp_qpos"]
        grasp = d["grasp_qpos"]
        squeeze = d["squeeze_qpos"]

        open_qpos = pregrasp.copy()
        open_qpos[7:] = 0.0

        self.mj_ho.reset_pose_qpos(open_qpos, d["obj_pose"])

        hook = self.recorder
        if hook is not None:
            hook.start()

        logging.info("  [collect] open → pregrasp")
        self.mj_ho.control_hand_with_s_curve(
            open_qpos, pregrasp,
            duration=DUR_OPEN_TO_PRE,
            kind="quintic",
            record_hook=hook,
            phase_name="open_to_pre",
        )

        logging.info("  [collect] pregrasp → grasp")
        self.mj_ho.control_hand_with_s_curve(
            pregrasp, grasp,
            duration=DUR_PRE_TO_GRASP,
            kind="quintic",
            record_hook=hook,
            phase_name="pre_to_grasp",
        )

        logging.info("  [collect] grasp → squeeze")
        self.mj_ho.control_hand_with_s_curve(
            grasp, squeeze,
            duration=DUR_GRASP_TO_SQUEEZE,
            kind="quintic",
            record_hook=hook,
            phase_name="grasp_to_squeeze",
        )

        logging.info("  [collect] hold")
        self.mj_ho.control_hand_with_s_curve(
            squeeze, squeeze,
            duration=DUR_HOLD,
            kind="quintic",
            record_hook=hook,
            phase_name="hold",
        )

        if hook is not None:
            hook.stop()

        return True

    def run(self):
        pre_obj_qpos = self.mj_ho.get_obj_pose().copy()

        self._simulate_under_extforce_details(pre_obj_qpos)

        if self.recorder is not None:
            save_path = (
                self.input_npy_path
                .replace(
                    self.configs.grasp_dir,
                    os.path.join(self.configs.save_dir, "dataset"),
                )
                .replace(".npy", ".npz")
            )
            meta = {
                "obj_path": self.grasp_data["obj_path"],
                "obj_scale": float(self.grasp_data["obj_scale"]),
                "obj_pose": self.grasp_data["obj_pose"].tolist(),
                "source_npy": self.input_npy_path,
            }
            self.recorder.save(save_path, extra_meta=meta)
            logging.info(f"  [collect] saved → {save_path}")
        else:
            logging.info("  [collect] no data saved (collect_data=False)")

        return