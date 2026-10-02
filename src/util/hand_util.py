import os
import time

import trimesh
import numpy as np
import mujoco
import mujoco.viewer
import transforms3d.quaternions as tq

from .rot_util import interplote_pose, interplote_qpos


def camera_look_at(pos, target=(0.0, 0.0, 0.0), world_up=(0.0, 0.0, 1.0)):
    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    z_axis = pos - target
    z_axis /= np.linalg.norm(z_axis) + 1e-12

    x_axis = np.cross(world_up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.array([0, 1, 0]), z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12

    y_axis = np.cross(z_axis, x_axis)

    return pos.tolist(), np.concatenate([x_axis, y_axis]).tolist()


def quat_slerp(q1, q2, t):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)

    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        res = q1 + t * (q2 - q1)
        return res / (np.linalg.norm(res) + 1e-12)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q1 + s1 * q2


def s_curve(t, T, kind="quintic"):
    if t <= 0:
        return 0.0
    if t >= T:
        return 1.0
    x = t / T
    if kind == "quintic":
        return 10 * x**3 - 15 * x**4 + 6 * x**5
    elif kind == "cosine":
        return 0.5 * (1.0 - np.cos(np.pi * x))
    else:
        raise ValueError(f"Unknown S-curve kind: {kind}")


class MjHO:

    hand_prefix: str = "child-"

    def __init__(
        self,
        obj_path,
        obj_scale,
        obj_density,
        hand_xml_path,
        hand_mocap,
        exclude_table_contact,
        friction_coef,
        has_floor_z0,
        debug_render=False,
        debug_viewer=False,
    ):
        self.hand_mocap = hand_mocap
        self.spec = mujoco.MjSpec()
        self.spec.meshdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.spec.option.timestep = 0.004
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY

        if debug_render or debug_viewer:
            self.spec.add_texture(
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                rgb1=[0.3, 0.5, 0.7],
                rgb2=[0.3, 0.5, 0.7],
                width=512,
                height=512,
            )
            self.spec.worldbody.add_light(
                name="spotlight",
                pos=[0, -1, 2],
                castshadow=False,
            )

            self.cam_names = ["wrist_cam"]
        else:
            self.cam_names = []

        self._add_hand(hand_xml_path, hand_mocap)
        self._add_object(obj_path, obj_scale, obj_density, has_floor_z0)
        self._set_friction(friction_coef)
        self.spec.add_key()
        if exclude_table_contact is not None:
            for body_name in exclude_table_contact:
                self.spec.add_exclude(
                    bodyname1="world",
                    bodyname2=f"{self.hand_prefix}{body_name}",
                )

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        qpos2ctrl_matrix = np.zeros((self.model.nu, self.model.nv))
        mujoco.mju_sparse2dense(
            qpos2ctrl_matrix,
            self.data.actuator_moment,
            self.data.moment_rownnz,
            self.data.moment_rowadr,
            self.data.moment_colind,
        )
        self._qpos2ctrl_matrix = qpos2ctrl_matrix[..., :-6]

        self.debug_viewer = None
        self.debug_render = None

        if debug_viewer:
            self.debug_viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self.debug_viewer.lock():
                opt = self.debug_viewer.opt
                opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
                opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
                opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
                opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
            self.debug_viewer.sync()

        if debug_render:
            self.debug_render = mujoco.Renderer(self.model, 320, 320)
            self.debug_options = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.debug_options)
            self.debug_options.geomgroup[:] = [1, 1, 1, 0, 0, 0]
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
            self.debug_images = {name: [] for name in self.cam_names}
        return

    def __del__(self):
        try:
            v = getattr(self, "debug_viewer", None)
            if v is not None:
                try:
                    v.close()
                except Exception:
                    pass
                self.debug_viewer = None
        except Exception:
            pass
        try:
            r = getattr(self, "debug_render", None)
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
                self.debug_render = None
        except Exception:
            pass

    def _add_hand(self, xml_path, mocap_base):
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir

        for g in child_spec.geoms:
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]

        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(
            child_spec.worldbody, self.hand_prefix, ""
        )
        if mocap_base:
            child_world.add_freejoint(name="hand_freejoint")
            self.spec.worldbody.add_body(name="mocap_body", mocap=True)
            self.spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="mocap_body",
                name2=f"{self.hand_prefix}world",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                solimp=[0.9, 0.95, 0.001, 0.5, 2],
                data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            )

        def _find_body(parent, suffix):
            for b in parent.bodies:
                if (b.name or "").endswith(suffix):
                    return b
                r = _find_body(b, suffix)
                if r is not None:
                    return r
            return None

        palm_body = _find_body(child_world, "rh_palm")
        if palm_body is not None:
            palm_body.add_camera(
                name="wrist_cam",
                pos=[0.0, -0.25, 0.10],
                xyaxes=[1,0,0, 0,0,1],
                fovy=60,
            )
            print(f"[INFO] wrist_cam 已挂在 {palm_body.name}")
        else:
            print("[WARN] 找不到 rh_palm，wrist_cam 未添加")
        return

    def _add_object(self, obj_path, obj_scale, obj_density, has_floor_z0):
        if has_floor_z0:
            self.spec.worldbody.add_geom(
                name="object_collision_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0, 0, 0],
                size=[0, 0, 1.0],
                group=3,
            )

        obj_body = self.spec.worldbody.add_body(name="object")
        obj_body.add_freejoint(name="obj_freejoint")
        parts_folder = os.path.join(obj_path, "urdf/meshes")
        for file in os.listdir(parts_folder):
            file_path = os.path.join(parts_folder, file)
            mesh_name = file.replace(".obj", "")
            mesh_id = mesh_name.replace("convex_piece_", "")

            self.spec.add_mesh(
                name=mesh_name,
                file=file_path,
                scale=[obj_scale, obj_scale, obj_scale],
            )
            obj_body.add_geom(
                name=f"object_visual_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=0,
                contype=0,
                conaffinity=0,
                group=2,
            )
            obj_body.add_geom(
                name=f"object_collision_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=obj_density,
                group=3,
            )
        return

    def _set_friction(self, test_friction):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        self.spec.option.impratio = 10
        for g in self.spec.geoms:
            g.friction[:2] = test_friction
            g.condim = 4
        return

    def _qpos2ctrl(self, hand_qpos):
        if self.hand_mocap:
            return self._qpos2ctrl_matrix[:, 6:] @ hand_qpos[7:]
        else:
            return self._qpos2ctrl_matrix @ hand_qpos

    def get_obj_pose(self):
        return self.data.qpos[-7:]

    def get_contact_info(self, hand_qpos, obj_pose, obj_margin=0):
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = obj_margin

        self.reset_pose_qpos(hand_qpos, obj_pose)

        object_id = self.model.nbody - 1
        hand_id = self.model.nbody - 2
        world_id = -1 if self.hand_mocap else 0

        ho_contact = []
        hh_contact = []
        for contact in self.data.contact:
            body1_id = self.model.geom(contact.geom1).bodyid
            body2_id = self.model.geom(contact.geom2).bodyid
            body1_name = self.model.body(self.model.geom(contact.geom1).bodyid).name
            body2_name = self.model.body(self.model.geom(contact.geom2).bodyid).name

            if (
                body1_id > world_id and body1_id < hand_id and body2_id == object_id
            ) or (body2_id > world_id and body2_id < hand_id and body1_id == object_id):
                if body2_id == object_id:
                    contact_normal = contact.frame[0:3]
                    hand_body_name = body1_name.removeprefix(self.hand_prefix)
                    obj_body_name = body2_name
                else:
                    contact_normal = -contact.frame[0:3]
                    hand_body_name = body2_name.removeprefix(self.hand_prefix)
                    obj_body_name = body1_name
                ho_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact_normal,
                        "body1_name": hand_body_name,
                        "body2_name": obj_body_name,
                    }
                )
            elif (
                body1_id > world_id
                and body1_id < hand_id
                and body2_id > world_id
                and body2_id < hand_id
            ):
                hh_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact.frame[0:3],
                        "body1_name": body1_name,
                        "body2_name": body2_name,
                    }
                )

        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = 0
        return ho_contact, hh_contact

    def set_ext_force_on_obj(self, ext_force):
        self.data.xfrc_applied[-1] = ext_force
        return

    def reset_pose_qpos(self, hand_qpos, obj_pose):
        self.model.key_qpos[0] = np.concatenate([hand_qpos, obj_pose], axis=0)
        self.model.key_ctrl[0] = self._qpos2ctrl(hand_qpos)
        self.model.key_qvel[0] = 0
        self.model.key_act[0] = 0
        if self.hand_mocap:
            self.model.key_mpos[0] = hand_qpos[:3]
            self.model.key_mquat[0] = hand_qpos[3:7]

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return

    def _render_all_cameras(self):
        if self.debug_render is None:
            return None
        frames = {}
        for cam_name in self.cam_names:
            self.debug_render.update_scene(self.data, cam_name, self.debug_options)
            frames[cam_name] = self.debug_render.render().copy()
        return frames

    def control_hand_with_interp(
        self, hand_qpos1, hand_qpos2, step_outer=10, step_inner=10
    ):
        if self.hand_mocap:
            pose_interp = interplote_pose(hand_qpos1[:7], hand_qpos2[:7], step_outer)
        qpos_interp = interplote_qpos(
            self._qpos2ctrl(hand_qpos1), self._qpos2ctrl(hand_qpos2), step_outer
        )
        for j in range(step_outer):
            if self.hand_mocap:
                self.data.mocap_pos[0] = pose_interp[j, :3]
                self.data.mocap_quat[0] = pose_interp[j, 3:7]
            self.data.ctrl[:] = qpos_interp[j]
            mujoco.mj_forward(self.model, self.data)
            self.control_hand_step(step_inner)
        return

    def control_hand_step(self, step_inner):
        target_sync_dt = self.model.opt.timestep

        for _ in range(step_inner):
            t0 = time.time()
            mujoco.mj_step(self.model, self.data)

            v = self.debug_viewer
            if v is not None:
                try:
                    if not v.is_running():
                        self.debug_viewer = None
                    else:
                        v.cam.lookat[:] = self.data.qpos[-7:-4]
                        v.sync()
                        elapsed = time.time() - t0
                        if elapsed < target_sync_dt:
                            time.sleep(target_sync_dt - elapsed)
                except Exception:
                    self.debug_viewer = None

        frames = self._render_all_cameras()
        if frames is not None:
            for cam_name, frame in frames.items():
                self.debug_images[cam_name].append(frame)
        return

    def control_hand_with_s_curve(
        self,
        hand_qpos1,
        hand_qpos2,
        duration=1.0,
        kind="quintic",
        record_hook=None,
        phase_name="",
        realtime_factor=1.0,
    ):
        dt = self.model.opt.timestep
        n_steps = max(2, int(round(duration / dt)))

        q1 = np.asarray(hand_qpos1, dtype=np.float64).copy()
        q2 = np.asarray(hand_qpos2, dtype=np.float64).copy()

        for i in range(n_steps + 1):
            step_t0 = time.time()
            t = i * dt
            s = s_curve(t, duration, kind=kind)

            if self.hand_mocap:
                pos_i = (1.0 - s) * q1[:3] + s * q2[:3]
                quat_i = quat_slerp(q1[3:7], q2[3:7], s)
                joints_i = (1.0 - s) * q1[7:] + s * q2[7:]

                self.data.mocap_pos[0] = pos_i
                self.data.mocap_quat[0] = quat_i
                full_qpos = np.concatenate([pos_i, quat_i, joints_i])
                self.data.ctrl[:] = self._qpos2ctrl(full_qpos)
            else:
                full_qpos = (1.0 - s) * q1 + s * q2
                self.data.ctrl[:] = self._qpos2ctrl(full_qpos)

            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

            need_render = False
            if record_hook is not None and self.debug_render is not None:
                if hasattr(record_hook, "should_sample"):
                    need_render = record_hook.should_sample(self)
                else:
                    need_render = getattr(record_hook, "recording", True)

            frames = None
            if need_render:
                frames = self._render_all_cameras()

            if record_hook is not None:
                record_hook(
                    t=t, s=s, mj_ho=self,
                    phase_name=phase_name,
                    frames=frames,
                )

            v = self.debug_viewer
            if v is not None:
                try:
                    if not v.is_running():
                        self.debug_viewer = None
                    else:
                        v.cam.lookat[:] = self.data.qpos[-7:-4]
                        v.sync()

                        if realtime_factor > 1e-6:
                            target = dt / realtime_factor
                            elapsed = time.time() - step_t0
                            if elapsed < target:
                                time.sleep(target - elapsed)
                except Exception:
                    self.debug_viewer = None
        return


class RobotKinematics:
    def __init__(self, xml_path):
        spec = mujoco.MjSpec.from_file(xml_path)
        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mesh_geom_info = {}
        for i in range(self.mj_model.ngeom):
            geom = self.mj_model.geom(i)
            mesh_id = geom.dataid
            if mesh_id != -1:
                mjm = self.mj_model.mesh(mesh_id)
                vert = self.mj_model.mesh_vert[
                    mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]
                ]
                face = self.mj_model.mesh_face[
                    mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]
                ]
                body_name = self.mj_model.body(geom.bodyid).name
                mesh_name = mjm.name
                self.mesh_geom_info[f"{body_name}_{mesh_name}"] = {
                    "vert": vert,
                    "face": face,
                    "geom_id": i,
                }
        return

    def forward_kinematics(self, q):
        self.mj_data.qpos = q
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        return

    def get_init_meshes(self):
        init_mesh_lst = []
        mesh_name_lst = []
        for k, v in self.mesh_geom_info.items():
            mesh_name_lst.append(k)
            init_mesh_lst.append(trimesh.Trimesh(vertices=v["vert"], faces=v["face"]))
        return mesh_name_lst, init_mesh_lst

    def get_poses(self, root_pose):
        geom_poses = np.zeros((len(self.mesh_geom_info), 7))
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        for i, v in enumerate(self.mesh_geom_info.values()):
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_poses[i, :3] = root_rot @ geom_trans + root_trans
            geom_poses[i, 3:] = tq.mat2quat(root_rot @ geom_rot)
        return geom_poses

    def get_posed_meshes(self, root_pose):
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        full_tm = []
        for k, v in self.mesh_geom_info.items():
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            posed_vert = (v["vert"] @ geom_rot.T + geom_trans) @ root_rot.T + root_trans
            posed_tm = trimesh.Trimesh(vertices=posed_vert, faces=v["face"])
            full_tm.append(posed_tm)
        full_tm = trimesh.util.concatenate(full_tm)
        return full_tm


if __name__ == "__main__":
    xml_path = os.path.join(
        os.path.dirname(__file__), "../../assets/hand/shadow/customized.xml"
    )
    kinematic = RobotKinematics(xml_path)
    hand_qpos = np.zeros((22))
    kinematic.forward_kinematics(hand_qpos)
    visual_mesh = kinematic.get_posed_meshes()
    visual_mesh.export(f"debug_hand.obj")