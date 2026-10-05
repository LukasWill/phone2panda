"""Thin wrapper around LIBERO for the single task we use: LIBERO-Goal "put the bowl on the plate".

Why our own wrapper instead of LeRobot's `LiberoEnv`?
  We need things a policy-evaluation env does not expose: exact object poses (to script and
  retarget), the 50 fixed init states *and* freshly sampled scenes, and a physics-only mode
  (no rendering) that is ~15x faster on CPU. Everything a policy sees is produced to match
  LeRobot exactly, so policies trained on our data and on `lerobot/libero` are interchangeable:

  - images: raw LIBERO camera images rotated by 180 deg (flip H and W), as in LeRobot's
    `LiberoProcessorStep`; keys `observation.images.image` (agentview) and `...image2` (wrist)
  - state (8-D): [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)], float32
  - after every reset: 10 no-op steps with the gripper open so objects settle
  - episodes end on success or after 300 steps (LeRobot's limit for libero_goal)

Simulator facts worth knowing (measured, see notebooks/00_sim_sanity.ipynb):
  - world frame: x forward from the robot, y left, z up; table top at z ~= 0.898
  - robot base at (-0.66, 0, 0.912); bowl ~0.56 m in front of it, plate ~0.16 m beyond the bowl
  - bowl: 11.1 cm wide, 5.3 cm tall, body origin at its bottom centre
  - plate: 13.7 cm wide, 1.7 cm tall, body origin at its bottom centre
  - the 50 eval init states move each object by only ~+-1.5 cm
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUITE = "libero_goal"
TASK_LANGUAGE = "put the bowl on the plate"
MAX_STEPS = 300
NUM_SETTLE_STEPS = 10
NOOP = np.array([0, 0, 0, 0, 0, 0, -1.0])
BOWL, PLATE = "akita_black_bowl_1", "plate_1"

# Measured object geometry (metres), used by the scripted policy and by retargeting.
BOWL_RADIUS, BOWL_HEIGHT = 0.0557, 0.0526
PLATE_RADIUS, PLATE_HEIGHT = 0.0685, 0.0166


def ensure_libero_config() -> None:
    """`import libero` asks interactive questions (input()) if ~/.libero/config.yaml is missing,
    which hangs a notebook cell forever. Write the default config before the first import."""
    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    cfg = cfg_dir / "config.yaml"
    if cfg.exists():
        return
    import importlib.util

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("LIBERO is not installed: pip install 'lerobot[libero]' (Linux only)")
    root = Path(list(spec.submodule_search_locations)[0]) / "libero"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        f"benchmark_root: {root}\nbddl_files: {root / 'bddl_files'}\ninit_states: {root / 'init_files'}\n"
        f"datasets: {root.parent / 'datasets'}\nassets: {root / 'assets'}\n"
    )


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion -> axis * angle. Same formula as LeRobot's LiberoProcessorStep,
    so our state vectors are bit-for-bit comparable with lerobot/libero."""
    q = np.asarray(q, dtype=np.float64)
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den <= 1e-10:
        return np.zeros(3)
    return q[:3] / den * 2.0 * np.arccos(w)


def quat2mat(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion -> 3x3 rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class Scene:
    """Everything a scripted / retargeted controller needs, in the LIBERO world frame."""
    eef_pos: np.ndarray        # (3,) grip site, between the fingertips
    eef_R: np.ndarray          # (3, 3) gripper orientation; its y-axis is the finger closing axis
    gripper_qpos: np.ndarray   # (2,) finger joint positions (~0.039 open, ~0.001 closed)
    bowl_pos: np.ndarray       # (3,) bowl bottom centre
    bowl_quat: np.ndarray
    plate_pos: np.ndarray      # (3,) plate bottom centre
    plate_quat: np.ndarray


def lerobot_state(raw_obs: dict) -> np.ndarray:
    return np.concatenate([
        raw_obs["robot0_eef_pos"], quat2axisangle(raw_obs["robot0_eef_quat"]), raw_obs["robot0_gripper_qpos"],
    ]).astype(np.float32)


def lerobot_images(raw_obs: dict, size: int | None = None) -> dict[str, np.ndarray]:
    """LeRobot camera keys, rotated 180 deg, optionally resized (INTER_AREA = proper downsampling)."""
    import cv2

    out = {}
    for raw_key, key in (("agentview_image", "observation.images.image"),
                         ("robot0_eye_in_hand_image", "observation.images.image2")):
        if raw_key not in raw_obs:
            continue
        img = np.ascontiguousarray(raw_obs[raw_key][::-1, ::-1])
        if size is not None and img.shape[0] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        out[key] = img
    return out


class TaskEnv:
    """One LIBERO env for our task.

    render=False builds the simulator without cameras (physics only): use it whenever only the
    outcome matters (scripted checks, retargeting replays). render=True gives camera images and
    should run on a GPU runtime (MUJOCO_GL=egl), because software rendering is very slow.
    """

    def __init__(self, render: bool = True, camera_size: int = 256):
        ensure_libero_config()
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[SUITE]()
        names = [suite.get_task(i).language for i in range(suite.n_tasks)]
        self.task_id = names.index(TASK_LANGUAGE)
        task = suite.get_task(self.task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.init_states = suite.get_task_init_states(self.task_id)  # (50, n_state)
        self.render = render
        kwargs = (dict(camera_heights=camera_size, camera_widths=camera_size) if render
                  else dict(use_camera_obs=False, has_offscreen_renderer=False))
        self._env = OffScreenRenderEnv(bddl_file_name=bddl, **kwargs)
        self.t = 0
        self.raw_obs: dict = {}

    # ---- episode control --------------------------------------------------------------------
    def reset(self, init_state_id: int | None = None, seed: int | None = None) -> dict:
        """init_state_id in [0, 50): one of LIBERO's fixed evaluation scenes.
        init_state_id=None: a fresh scene sampled from the task's placement regions (use a seed
        for reproducibility). Training data must only ever come from fresh scenes."""
        if seed is not None:
            self._env.seed(seed)
        obs = self._env.reset()
        if init_state_id is not None:
            obs = self._env.set_init_state(self.init_states[init_state_id])
        for _ in range(NUM_SETTLE_STEPS):
            obs, _, _, _ = self._env.step(NOOP)
        self.t = 0
        self.raw_obs = obs
        return obs

    def step(self, action: np.ndarray) -> tuple[dict, bool, bool]:
        """Returns (raw_obs, success, done)."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        obs, _, _, _ = self._env.step(action)
        self.t += 1
        success = bool(self._env.check_success())
        self.raw_obs = obs
        return obs, success, success or self.t >= MAX_STEPS

    def close(self):
        self._env.close()

    # ---- state access -------------------------------------------------------------------------
    def scene(self) -> Scene:
        o = self.raw_obs
        return Scene(
            eef_pos=np.array(o["robot0_eef_pos"]), eef_R=quat2mat(np.array(o["robot0_eef_quat"])),
            gripper_qpos=np.array(o["robot0_gripper_qpos"]),
            bowl_pos=np.array(o[f"{BOWL}_pos"]), bowl_quat=np.array(o[f"{BOWL}_quat"]),
            plate_pos=np.array(o[f"{PLATE}_pos"]), plate_quat=np.array(o[f"{PLATE}_quat"]),
        )

    def low_dim_state(self) -> np.ndarray:
        """Compact simulator state for offline RL / world-model heads: robot state (8) + bowl pose (7)
        + plate position (3) = 18-D."""
        s = self.scene()
        return np.concatenate([lerobot_state(self.raw_obs), s.bowl_pos, s.bowl_quat, s.plate_pos]).astype(np.float32)

    def frame(self, size: int | None = 128) -> dict[str, np.ndarray]:
        """Policy-ready observation in LeRobot's format (requires render=True)."""
        f = lerobot_images(self.raw_obs, size)
        f["observation.state"] = lerobot_state(self.raw_obs)
        return f

    def sim_state(self) -> np.ndarray:
        """Full MuJoCo state (to re-render or restart an episode exactly)."""
        return np.array(self._env.get_sim_state())
