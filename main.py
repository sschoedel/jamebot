from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import glfw
import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import least_squares


COLLISION_GEOM_GROUP = 3
DOUBLE_CLICK_SECONDS = 0.25
DOUBLE_CLICK_PIXELS = 5.0
PLOT_MARGIN = 12
PLOT_MAX_POINTS = 600
FORCE_PLOT_LIMIT_PADDING = 2.0
LINEAR_POSITION_PLOT_RANGE = (0.0, 0.1)
BALLSCREW_EQUILIBRIUM_POSITION = 0.05
KEYBOARD_PERTURBATION_DELTA_V = 0.03
KEYBOARD_PERTURBATION_ARROW_SECONDS = 0.45
KEYBOARD_PERTURBATION_ARROW_LENGTH = 0.16
KEYBOARD_PERTURBATION_ARROW_HEIGHT = 0.12
KEYBOARD_PERTURBATION_ARROW_WIDTH = 0.008
KEYBOARD_PERTURBATION_ARROW_RGBA = np.array([1.0, 0.35, 0.0, 0.9], dtype=np.float32)
KEYBOARD_PERTURBATION_DIRECTIONS = (
    np.array([1.0, 0.0], dtype=np.float64),
    np.array([0.0, 1.0], dtype=np.float64),
    np.array([-1.0, 0.0], dtype=np.float64),
    np.array([0.0, -1.0], dtype=np.float64),
)
STANDARD_VIS_FLAG_KEYS = {
    glfw.KEY_C: ("contact points", mujoco.mjtVisFlag.mjVIS_CONTACTPOINT),
    glfw.KEY_F: ("contact forces", mujoco.mjtVisFlag.mjVIS_CONTACTFORCE),
    glfw.KEY_P: ("contact force split", mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT),
}
CONSTRAINT_FREE = "free"
CONSTRAINT_VERTICAL = "vertical only"
CONSTRAINT_FOOT_PIN = "foot pinned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a resettable MuJoCo simulation.")
    parser.add_argument("model", type=Path, nargs="?", default=Path("robot_model/scene.xml"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--reset-clearance", type=float, default=0.0, help="Lowest robot mesh height after reset.")
    parser.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the upright LQR balance controller.",
    )
    parser.add_argument("--control-limit", type=float, default=10.0, help="Symmetric actuator command limit.")
    parser.add_argument("--flywheel-control-limit", type=float, default=0.5, help="Symmetric flywheel torque limit.")
    parser.add_argument("--flywheel-gain-scale", type=float, default=1.0, help="Scale applied to LQR flywheel rows.")
    parser.add_argument(
        "--flywheel-velocity-damping",
        type=float,
        default=0.01,
        help="Extra damping torque per rad/s of flywheel speed.",
    )
    parser.add_argument(
        "--flywheel-slew-rate",
        type=float,
        default=20.0,
        help="Maximum flywheel command change in torque units per second.",
    )
    parser.add_argument("--lqr-clearance", type=float, default=-0.001, help="Lowest robot mesh height for linearization.")
    parser.add_argument("--lqr-fd-eps", type=float, default=1e-6, help="Finite-difference epsilon for linearization.")
    parser.add_argument("--trim-tol", type=float, default=1e-5, help="Maximum accepted equilibrium qacc norm.")
    parser.add_argument("--trim-max-nfev", type=int, default=2000, help="Maximum function evaluations per trim seed.")
    parser.add_argument(
        "--force-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show actuator force history plots in the viewer.",
    )
    parser.add_argument("--force-plot-window", type=float, default=5.0, help="Seconds of actuator force history to show.")
    parser.add_argument("--dry-run", action="store_true", help="Load the model, design the controller, and exit.")
    return parser.parse_args()


def first_freejoint_qadr(model: mujoco.MjModel) -> int | None:
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[joint_id])
    return None


def first_freejoint_dofadr(model: mujoco.MjModel) -> int | None:
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_dofadr[joint_id])
    return None


def robot_min_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    mins = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
            continue
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            vert_start = int(model.mesh_vertadr[mesh_id])
            vert_count = int(model.mesh_vertnum[mesh_id])
            vertices = model.mesh_vert[vert_start : vert_start + vert_count]
            geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
            world_vertices = data.geom_xpos[geom_id] + vertices @ geom_xmat.T
            mins.append(float(np.min(world_vertices[:, 2])))
        else:
            mins.append(float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id]))
    return min(mins, default=0.0)


def lowest_body_point_local(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    world_points = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            vert_start = int(model.mesh_vertadr[mesh_id])
            vert_count = int(model.mesh_vertnum[mesh_id])
            vertices = model.mesh_vert[vert_start : vert_start + vert_count]
            geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
            geom_vertices = data.geom_xpos[geom_id] + vertices @ geom_xmat.T
            world_points.append(geom_vertices[int(np.argmin(geom_vertices[:, 2]))])
        else:
            world_points.append(data.geom_xpos[geom_id] - np.array([0.0, 0.0, model.geom_rbound[geom_id]]))

    if not world_points:
        return np.zeros(3), data.xpos[body_id].copy()

    world_point = min(world_points, key=lambda point: float(point[2]))
    body_xmat = data.xmat[body_id].reshape(3, 3)
    local_point = body_xmat.T @ (world_point - data.xpos[body_id])
    return local_point, world_point.copy()


def reset_upright(model: mujoco.MjModel, data: mujoco.MjData, clearance: float) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0
    data.ctrl[:] = 0
    data.qacc_warmstart[:] = 0

    free_qadr = first_freejoint_qadr(model)
    if free_qadr is not None:
        data.qpos[free_qadr : free_qadr + 3] = (0, 0, 0)
        data.qpos[free_qadr + 3 : free_qadr + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(model, data)
        data.qpos[free_qadr + 2] += clearance - robot_min_z(model, data)
    else:
        print("warning: no free joint found, reset may not work as expected")

    mujoco.mj_forward(model, data)


def tangent_position_names(model: mujoco.MjModel) -> list[str]:
    entries: list[tuple[int, str]] = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        dof_adr = int(model.jnt_dofadr[joint_id])
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            freejoint_names = (
                "base_over_foot:x",
                "base_over_foot:y",
                f"{name}:z",
                f"{name}:rot_x",
                f"{name}:rot_y",
                f"{name}:rot_z",
            )
            entries.extend((dof_adr + offset, entry_name) for offset, entry_name in enumerate(freejoint_names))
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            entries.extend((dof_adr + offset, f"{name}:rot_{axis}") for offset, axis in enumerate(("x", "y", "z")))
        else:
            entries.append((dof_adr, name))

    names = [name for _, name in sorted(entries)]
    if len(names) != model.nv:
        return [f"qpos_error[{i}]" for i in range(model.nv)]
    return names


def tangent_state_names(model: mujoco.MjModel) -> list[str]:
    qpos_names = tangent_position_names(model)
    qvel_names = [f"vel:{name}" for name in qpos_names]
    act_names = [
        f"act:{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or i}" for i in range(model.na)
    ]
    return qpos_names + qvel_names + act_names


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_yaw(quat: np.ndarray) -> float:
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def rotate_xy_world_to_yaw_frame(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    yaw = quat_yaw(quat)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array(
        [
            cos_yaw * vector[0] + sin_yaw * vector[1],
            -sin_yaw * vector[0] + cos_yaw * vector[1],
        ],
        dtype=np.float64,
    )


def rotate_xy_yaw_frame_to_world(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    yaw = quat_yaw(quat)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array(
        [
            cos_yaw * vector[0] - sin_yaw * vector[1],
            sin_yaw * vector[0] + cos_yaw * vector[1],
        ],
        dtype=np.float64,
    )


def remove_yaw(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat)
    yaw = quat_yaw(quat)
    yaw_quat = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
    tilt_quat = quat_multiply(quat_conjugate(yaw_quat), quat)
    if tilt_quat[0] < 0.0:
        tilt_quat = -tilt_quat
    return tilt_quat / np.linalg.norm(tilt_quat)


def quat_log_vector(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    vector = quat[1:4]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1e-12:
        return 2.0 * vector
    angle = 2.0 * np.arctan2(vector_norm, quat[0])
    return vector * (angle / vector_norm)


def yaw_invariant_freejoint_error(reference_quat: np.ndarray, data_quat: np.ndarray) -> np.ndarray:
    reference_tilt = remove_yaw(reference_quat)
    data_tilt = remove_yaw(data_quat)
    error_quat = quat_multiply(quat_conjugate(reference_tilt), data_tilt)
    return quat_log_vector(error_quat)


def body_origin_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, None, data.xpos[body_id], body_id)
    return jacp @ data.qvel


def base_and_foot_body_ids(model: mujoco.MjModel) -> tuple[int | None, int | None]:
    base_body_id = None
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            base_body_id = int(model.jnt_bodyid[joint_id])
            break

    foot_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot"))
    if foot_body_id < 0:
        foot_body_id = None
    return base_body_id, foot_body_id


def yaw_frame_base_foot_xy(model: mujoco.MjModel, data: mujoco.MjData, free_qadr: int) -> np.ndarray | None:
    base_body_id, foot_body_id = base_and_foot_body_ids(model)
    if base_body_id is None or foot_body_id is None:
        return None

    relative_position = data.xpos[base_body_id] - data.xpos[foot_body_id]
    return rotate_xy_world_to_yaw_frame(data.qpos[free_qadr + 3 : free_qadr + 7], relative_position)


def yaw_frame_base_foot_xy_velocity(model: mujoco.MjModel, data: mujoco.MjData, free_qadr: int) -> np.ndarray | None:
    base_body_id, foot_body_id = base_and_foot_body_ids(model)
    if base_body_id is None or foot_body_id is None:
        return None

    relative_velocity = body_origin_velocity(model, data, base_body_id) - body_origin_velocity(model, data, foot_body_id)
    return rotate_xy_world_to_yaw_frame(data.qpos[free_qadr + 3 : free_qadr + 7], relative_velocity)


def state_error(model: mujoco.MjModel, reference: mujoco.MjData, data: mujoco.MjData) -> np.ndarray:
    qpos_error = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, qpos_error, 1.0, reference.qpos, data.qpos)
    qvel_error = data.qvel - reference.qvel
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        dofadr = int(model.jnt_dofadr[joint_id])
        reference_base_foot_xy = yaw_frame_base_foot_xy(model, reference, qadr)
        data_base_foot_xy = yaw_frame_base_foot_xy(model, data, qadr)
        if reference_base_foot_xy is not None and data_base_foot_xy is not None:
            qpos_error[dofadr : dofadr + 2] = data_base_foot_xy - reference_base_foot_xy

        qpos_error[dofadr + 3 : dofadr + 6] = yaw_invariant_freejoint_error(
            reference.qpos[qadr + 3 : qadr + 7],
            data.qpos[qadr + 3 : qadr + 7],
        )

        reference_base_foot_xy_velocity = yaw_frame_base_foot_xy_velocity(model, reference, qadr)
        data_base_foot_xy_velocity = yaw_frame_base_foot_xy_velocity(model, data, qadr)
        if reference_base_foot_xy_velocity is not None and data_base_foot_xy_velocity is not None:
            qvel_error[dofadr : dofadr + 2] = data_base_foot_xy_velocity - reference_base_foot_xy_velocity

    pieces = [qpos_error, qvel_error]
    if model.na:
        pieces.append(data.act - reference.act)
    return np.concatenate(pieces)


@dataclass
class TrimResult:
    residual_norm: float
    cost: float
    success: bool
    message: str
    nfev: int
    initial_clearance: float
    min_z: float
    contact_count: int
    constraint_count: int


def quat_from_roll_pitch(roll: float, pitch: float) -> np.ndarray:
    roll_quat = np.array([np.cos(0.5 * roll), np.sin(0.5 * roll), 0.0, 0.0])
    pitch_quat = np.array([np.cos(0.5 * pitch), 0.0, np.sin(0.5 * pitch), 0.0])
    quat = quat_multiply(pitch_quat, roll_quat)
    return quat / np.linalg.norm(quat)


def trim_joint_ids(model: mujoco.MjModel) -> list[int]:
    joint_ids: list[int] = []
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            joint_ids.append(joint_id)
        elif joint_type == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[joint_id]:
            joint_ids.append(joint_id)
    return joint_ids


def scalar_joint_bounds(model: mujoco.MjModel, joint_id: int) -> tuple[float, float]:
    qadr = int(model.jnt_qposadr[joint_id])
    if model.jnt_limited[joint_id]:
        lower, upper = model.jnt_range[joint_id]
        return float(lower), float(upper)

    center = float(model.qpos0[qadr])
    return center - 0.5, center + 0.5


def actuator_control_bounds(model: mujoco.MjModel, actuator_id: int, fallback_limit: float | None) -> tuple[float, float]:
    if model.actuator_ctrllimited[actuator_id]:
        lower, upper = model.actuator_ctrlrange[actuator_id]
        return float(lower), float(upper)

    limit = 100.0 if fallback_limit is None else float(fallback_limit)
    return -limit, limit


def equilibrium_clearance_seeds(requested_clearance: float) -> list[float]:
    seeds = [
        requested_clearance,
        min(requested_clearance, -0.015),
        -0.02,
        -0.025,
        -0.01,
        -0.005,
        -0.03,
        -0.04,
    ]
    unique: list[float] = []
    for seed in seeds:
        if not any(abs(seed - existing) < 1e-12 for existing in unique):
            unique.append(seed)
    return unique


def set_trim_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    free_qadr: int,
    joint_ids: list[int],
    variables: np.ndarray,
) -> mujoco.MjData:
    joint_count = len(joint_ids)
    joint_values = variables[3 : 3 + joint_count]
    controls = variables[3 + joint_count :]

    data.time = 0.0
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = controls
    data.qacc_warmstart[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    if model.na:
        data.act[:] = 0.0

    data.qpos[free_qadr : free_qadr + 3] = (0.0, 0.0, variables[0])
    data.qpos[free_qadr + 3 : free_qadr + 7] = quat_from_roll_pitch(variables[1], variables[2])
    for joint_id, joint_value in zip(joint_ids, joint_values):
        data.qpos[int(model.jnt_qposadr[joint_id])] = joint_value

    mujoco.mj_forward(model, data)
    return data


def find_balance_equilibrium(model: mujoco.MjModel, args: argparse.Namespace) -> tuple[mujoco.MjData, TrimResult]:
    free_qadr = first_freejoint_qadr(model)
    if free_qadr is None:
        raise RuntimeError("balance trim requires a freejoint")

    base = mujoco.MjData(model)
    base.qpos[:] = model.qpos0
    base.qvel[:] = 0.0
    base.ctrl[:] = 0.0
    base.qpos[free_qadr : free_qadr + 3] = (0.0, 0.0, 0.0)
    base.qpos[free_qadr + 3 : free_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, base)
    base_min_z = robot_min_z(model, base)

    joint_ids = trim_joint_ids(model)
    joint_qadrs = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    joint_lowers = []
    joint_uppers = []
    joint_initial = []
    for joint_id, qadr in zip(joint_ids, joint_qadrs):
        lower, upper = scalar_joint_bounds(model, joint_id)
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name == "ballscrew":
            target = float(np.clip(BALLSCREW_EQUILIBRIUM_POSITION, lower, upper))
            half_width = min(1e-8, max(1e-12, 0.5 * (upper - lower)))
            lower = max(lower, target - half_width)
            upper = min(upper, target + half_width)
        joint_lowers.append(lower)
        joint_uppers.append(upper)
        initial_value = BALLSCREW_EQUILIBRIUM_POSITION if joint_name == "ballscrew" else model.qpos0[qadr]
        joint_initial.append(float(np.clip(initial_value, lower, upper)))

    control_lowers = []
    control_uppers = []
    for actuator_id in range(model.nu):
        lower, upper = actuator_control_bounds(model, actuator_id, args.control_limit)
        control_lowers.append(lower)
        control_uppers.append(upper)

    best_reference: mujoco.MjData | None = None
    best_trim: TrimResult | None = None
    scratch = mujoco.MjData(model)

    for initial_clearance in equilibrium_clearance_seeds(args.lqr_clearance):
        root_z = initial_clearance - base_min_z
        x0 = np.array([root_z, 0.0, 0.0, *joint_initial, *([0.0] * model.nu)], dtype=np.float64)
        lower = np.array([root_z - 0.08, -0.35, -0.35, *joint_lowers, *control_lowers], dtype=np.float64)
        upper = np.array([root_z + 0.08, 0.35, 0.35, *joint_uppers, *control_uppers], dtype=np.float64)
        x0 = np.clip(x0, lower, upper)

        def residual(variables: np.ndarray) -> np.ndarray:
            set_trim_state(model, scratch, free_qadr, joint_ids, variables)
            controls = variables[3 + len(joint_ids) :]
            regularization = np.array([0.05 * variables[1], 0.05 * variables[2], *(0.001 * controls)])
            return np.concatenate((scratch.qacc.copy(), regularization))

        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=args.trim_max_nfev,
            x_scale="jac",
        )

        reference = mujoco.MjData(model)
        set_trim_state(model, reference, free_qadr, joint_ids, result.x)
        residual_norm = float(np.linalg.norm(reference.qacc))
        trim = TrimResult(
            residual_norm=residual_norm,
            cost=float(result.cost),
            success=bool(result.success),
            message=str(result.message),
            nfev=int(result.nfev),
            initial_clearance=float(initial_clearance),
            min_z=robot_min_z(model, reference),
            contact_count=int(reference.ncon),
            constraint_count=int(reference.nefc),
        )

        if best_trim is None or trim.residual_norm < best_trim.residual_norm:
            best_reference = reference
            best_trim = trim

        if trim.residual_norm <= args.trim_tol:
            break

    if best_reference is None or best_trim is None:
        raise RuntimeError("balance trim failed before evaluating any seed")
    if best_trim.residual_norm > args.trim_tol:
        raise RuntimeError(
            "could not find a static balance equilibrium: "
            f"best qacc norm={best_trim.residual_norm:.3e}, seed clearance={best_trim.initial_clearance:.4g}, "
            f"contacts={best_trim.contact_count}, constraints={best_trim.constraint_count}"
        )

    return best_reference, best_trim


def balance_cost_matrices(model: mujoco.MjModel, state_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    state_count = 2 * model.nv + model.na
    q_weights = np.zeros(state_count)

    for i, name in enumerate(state_names[: model.nv]):
        suffix = name.rsplit(":", 1)[-1]
        if suffix in {"x", "y"}:
            q_weights[i] = 100.0
        elif suffix in {"z", "rot_z"}:
            q_weights[i] = 0.0
        elif suffix in {"rot_x", "rot_y"}:
            q_weights[i] = 100.0
        elif name in {"flywheelX", "flywheelY"}:
            q_weights[i] = 0.0
        elif name == "ballscrew":
            q_weights[i] = 10000.0
        else:
            q_weights[i] = 1.0

    for i, name in enumerate(state_names[model.nv : 2 * model.nv], start=model.nv):
        base_name = name.removeprefix("vel:")
        suffix = base_name.rsplit(":", 1)[-1]
        if suffix in {"x", "y"}:
            q_weights[i] = 100.0
        elif suffix in {"z", "rot_z"}:
            q_weights[i] = 0.0
        elif suffix in {"rot_x", "rot_y"}:
            q_weights[i] = 100.0
        elif base_name in {"flywheelX", "flywheelY"}:
            q_weights[i] = 0.1
        elif base_name == "ballscrew":
            q_weights[i] = 100.0
        else:
            q_weights[i] = 1.0

    if model.nu == 3:
        r_weights = np.array([100.0, 100.0, 10000.0])
    else:
        r_weights = np.ones(model.nu)
    return np.diag(q_weights), np.diag(r_weights)


def positive_cost_state_indices(q_matrix: np.ndarray) -> list[int]:
    return [int(index) for index, weight in enumerate(np.diag(q_matrix)) if weight > 0.0]


def zero_cost_state_indices(q_matrix: np.ndarray) -> list[int]:
    return [int(index) for index, weight in enumerate(np.diag(q_matrix)) if weight == 0.0]


@dataclass
class LqrSolveResult:
    k_matrix: np.ndarray
    method: str


def solve_lqr_gain(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    q_matrix: np.ndarray,
    r_matrix: np.ndarray,
) -> LqrSolveResult:
    try:
        p_matrix = solve_discrete_are(a_matrix, b_matrix, q_matrix, r_matrix)
        k_matrix = np.linalg.solve(
            r_matrix + b_matrix.T @ p_matrix @ b_matrix,
            b_matrix.T @ p_matrix @ a_matrix,
        )
        return LqrSolveResult(k_matrix=k_matrix, method="undiscounted DARE")
    except (ValueError, np.linalg.LinAlgError) as error:
        first_error = error

    for discount in (0.999999, 0.9999, 0.999, 0.99):
        try:
            scaled = np.sqrt(discount)
            p_matrix = solve_discrete_are(scaled * a_matrix, scaled * b_matrix, q_matrix, r_matrix)
            k_matrix = np.linalg.solve(
                r_matrix + discount * b_matrix.T @ p_matrix @ b_matrix,
                discount * b_matrix.T @ p_matrix @ a_matrix,
            )
            return LqrSolveResult(
                k_matrix=k_matrix,
                method=f"discounted DARE beta={discount:g} after undiscounted DARE failed: {first_error}",
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

    raise RuntimeError(f"could not solve full-state LQR Riccati equation: {first_error}") from first_error


@dataclass
class LqrDesign:
    reference: mujoco.MjData
    reference_ctrl: np.ndarray
    trim: TrimResult
    state_names: list[str]
    q_weights: np.ndarray
    r_weights: np.ndarray
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    k_matrix: np.ndarray
    solve_method: str
    positive_cost_state_indices: list[int]
    zero_cost_state_indices: list[int]
    spectral_radius: float


class BalanceController:
    def __init__(
        self,
        model: mujoco.MjModel,
        design: LqrDesign,
        control_limit: float | None,
        flywheel_control_limit: float | None,
        flywheel_velocity_damping: float,
        flywheel_slew_rate: float,
    ):
        self.model = model
        self.design = design
        self.control_limit = control_limit
        self.flywheel_control_limit = flywheel_control_limit
        self.flywheel_velocity_damping = flywheel_velocity_damping
        self.flywheel_slew_rate = flywheel_slew_rate
        self.flywheel_actuators = self.find_flywheel_actuators()
        self.actuator_dofadrs = self.actuator_dof_addresses()
        self.last_ctrl = np.zeros(model.nu)
        self.last_error_norm = 0.0

    def find_flywheel_actuators(self) -> list[int]:
        flywheel_actuators = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else ""
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
            if "flywheel" in joint_name.lower() or "flywheel" in actuator_name.lower():
                flywheel_actuators.append(actuator_id)
        return flywheel_actuators

    def actuator_dof_addresses(self) -> list[int | None]:
        dofadrs: list[int | None] = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                dofadrs.append(None)
            else:
                dofadrs.append(int(self.model.jnt_dofadr[joint_id]))
        return dofadrs

    def reset(self, data: mujoco.MjData) -> None:
        self.last_ctrl[:] = data.ctrl

    def actuator_limit(self, actuator_id: int) -> float | None:
        limits = []
        if self.control_limit is not None:
            limits.append(abs(float(self.control_limit)))
        if self.model.actuator_ctrllimited[actuator_id]:
            ctrl_lower, ctrl_upper = self.model.actuator_ctrlrange[actuator_id]
            limits.append(min(abs(float(ctrl_lower)), abs(float(ctrl_upper))))
        if self.model.actuator_forcelimited[actuator_id]:
            force_lower, force_upper = self.model.actuator_forcerange[actuator_id]
            limits.append(min(abs(float(force_lower)), abs(float(force_upper))))
        if actuator_id in self.flywheel_actuators and self.flywheel_control_limit is not None:
            limits.append(abs(float(self.flywheel_control_limit)))
        if not limits:
            return None
        return min(limits)

    def clip_control(self, control: np.ndarray) -> np.ndarray:
        clipped = control.copy()
        for actuator_id in range(self.model.nu):
            limit = self.actuator_limit(actuator_id)
            if limit is not None:
                clipped[actuator_id] = np.clip(clipped[actuator_id], -limit, limit)
        return clipped

    def slew_limit_flywheels(self, control: np.ndarray) -> np.ndarray:
        if self.flywheel_slew_rate <= 0.0:
            return control

        limited = control.copy()
        max_delta = self.flywheel_slew_rate * self.model.opt.timestep
        for actuator_id in self.flywheel_actuators:
            delta = np.clip(limited[actuator_id] - self.last_ctrl[actuator_id], -max_delta, max_delta)
            limited[actuator_id] = self.last_ctrl[actuator_id] + delta
        return limited

    def apply(self, data: mujoco.MjData) -> None:
        error = state_error(self.model, self.design.reference, data)
        control = self.design.reference_ctrl - self.design.k_matrix @ error

        for actuator_id in self.flywheel_actuators:
            dofadr = self.actuator_dofadrs[actuator_id]
            if dofadr is not None:
                control[actuator_id] -= self.flywheel_velocity_damping * data.qvel[dofadr]

        control = self.clip_control(control)
        control = self.slew_limit_flywheels(control)
        control = self.clip_control(control)
        data.ctrl[:] = control
        self.last_ctrl[:] = control
        self.last_error_norm = float(np.linalg.norm(error))


def reset_to_lqr_reference(model: mujoco.MjModel, data: mujoco.MjData, design: LqrDesign) -> None:
    data.time = 0.0
    data.qpos[:] = design.reference.qpos
    data.qvel[:] = design.reference.qvel
    data.ctrl[:] = design.reference_ctrl
    data.qacc_warmstart[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    if model.na:
        data.act[:] = design.reference.act
    mujoco.mj_forward(model, data)


def state_from_tangent_perturbation(
    model: mujoco.MjModel,
    reference: mujoco.MjData,
    perturbation: np.ndarray,
) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.time = reference.time
    data.qpos[:] = reference.qpos
    data.qvel[:] = reference.qvel
    data.ctrl[:] = reference.ctrl
    data.qacc_warmstart[:] = reference.qacc_warmstart
    if model.na:
        data.act[:] = reference.act

    mujoco.mj_integratePos(model, data.qpos, perturbation[: model.nv].copy(), 1.0)
    data.qvel[:] += perturbation[model.nv : 2 * model.nv]
    if model.na:
        data.act[:] += perturbation[2 * model.nv :]

    mujoco.mj_forward(model, data)
    return data


def linearized_state_error_transform(
    model: mujoco.MjModel,
    reference: mujoco.MjData,
    eps: float,
) -> np.ndarray:
    state_count = 2 * model.nv + model.na
    transform = np.zeros((state_count, state_count))
    for state_index in range(state_count):
        perturbation = np.zeros(state_count)
        perturbation[state_index] = eps
        positive = state_error(model, reference, state_from_tangent_perturbation(model, reference, perturbation))
        perturbation[state_index] = -eps
        negative = state_error(model, reference, state_from_tangent_perturbation(model, reference, perturbation))
        transform[:, state_index] = (positive - negative) / (2.0 * eps)
    return transform


def design_balance_lqr(model: mujoco.MjModel, args: argparse.Namespace) -> LqrDesign:
    state_count = 2 * model.nv + model.na
    reference, trim = find_balance_equilibrium(model, args)
    reference_ctrl = reference.ctrl.copy()

    standard_a_matrix = np.zeros((state_count, state_count))
    standard_b_matrix = np.zeros((state_count, model.nu))
    mujoco.mjd_transitionFD(model, reference, args.lqr_fd_eps, 1, standard_a_matrix, standard_b_matrix, None, None)

    state_transform = linearized_state_error_transform(model, reference, args.lqr_fd_eps)
    transform_pinv = np.linalg.pinv(state_transform, rcond=1e-9)
    a_matrix = state_transform @ standard_a_matrix @ transform_pinv
    b_matrix = state_transform @ standard_b_matrix

    state_names = tangent_state_names(model)
    q_matrix, r_matrix = balance_cost_matrices(model, state_names)
    solve_result = solve_lqr_gain(a_matrix, b_matrix, q_matrix, r_matrix)
    k_matrix = solve_result.k_matrix
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else ""
        if "flywheel" in actuator_name.lower() or "flywheel" in joint_name.lower():
            k_matrix[actuator_id, :] *= args.flywheel_gain_scale
    spectral_radius = float(max(abs(np.linalg.eigvals(a_matrix - b_matrix @ k_matrix))))

    return LqrDesign(
        reference=reference,
        reference_ctrl=reference_ctrl,
        trim=trim,
        state_names=state_names,
        q_weights=np.diag(q_matrix),
        r_weights=np.diag(r_matrix),
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        k_matrix=k_matrix,
        solve_method=solve_result.method,
        positive_cost_state_indices=positive_cost_state_indices(q_matrix),
        zero_cost_state_indices=zero_cost_state_indices(q_matrix),
        spectral_radius=spectral_radius,
    )


def print_lqr_summary(design: LqrDesign) -> None:
    print(f"LQR solved with {design.solve_method}; closed-loop spectral radius={design.spectral_radius:.6f}")
    print(
        "Trim equilibrium: "
        f"qacc_norm={design.trim.residual_norm:.3e}, min_z={design.trim.min_z:.6f}, "
        f"contacts={design.trim.contact_count}, constraints={design.trim.constraint_count}, "
        f"seed_clearance={design.trim.initial_clearance:.4g}, nfev={design.trim.nfev}"
    )
    print("LQR reference qpos:")
    print(np.array2string(design.reference.qpos, precision=6, suppress_small=True, max_line_width=160))
    print("LQR reference qvel:")
    print(np.array2string(design.reference.qvel, precision=6, suppress_small=True, max_line_width=160))
    print("LQR reference ctrl:")
    print(np.array2string(design.reference_ctrl, precision=6, suppress_small=True, max_line_width=160))
    print("LQR state error at reference:")
    print(", ".join(f"{i}:{name}=0" for i, name in enumerate(design.state_names)))
    print("LQR state order:")
    print(", ".join(f"{i}:{name}" for i, name in enumerate(design.state_names)))
    print("LQR Q diagonal:")
    print(np.array2string(design.q_weights, precision=3, suppress_small=True, max_line_width=140))
    print("LQR R diagonal:")
    print(np.array2string(design.r_weights, precision=3, suppress_small=True, max_line_width=140))
    print("LQR positive-cost state indices:")
    print(", ".join(f"{i}:{design.state_names[i]}" for i in design.positive_cost_state_indices))
    print("LQR zero-cost state indices:")
    print(", ".join(f"{i}:{design.state_names[i]}" for i in design.zero_cost_state_indices))
    print("LQR K matrix:")
    print(np.array2string(design.k_matrix, precision=4, suppress_small=True, max_line_width=160))


class ActuatorForcePlot:
    def __init__(self, model: mujoco.MjModel, window_seconds: float, reference: mujoco.MjData | None = None):
        self.model = model
        self.window_seconds = max(window_seconds, model.opt.timestep)
        history_size = max(2, int(np.ceil(self.window_seconds / model.opt.timestep)) + 2)
        self.times: deque[float] = deque(maxlen=history_size)
        self.forces: deque[np.ndarray] = deque(maxlen=history_size)
        self.positions: deque[np.ndarray] = deque(maxlen=history_size)
        self.rotational_actuators, self.linear_actuators = self.split_actuators()
        self.desired_positions = self.linear_actuator_positions(reference) if reference is not None else None
        self.rotational_figure = self.make_figure("Rotational actuator torque (N m)", self.rotational_actuators)
        self.linear_figure = self.make_figure("Linear actuator force (N)", self.linear_actuators)
        self.linear_position_figure = self.make_figure(
            "Linear actuator position (m)",
            self.linear_actuators,
            symmetric=False,
        )
        self.linear_position_figure.yformat = "%.3f"

    def split_actuators(self) -> tuple[list[int], list[int]]:
        rotational: list[int] = []
        linear: list[int] = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0 or joint_id >= self.model.njnt:
                rotational.append(actuator_id)
                continue

            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
                linear.append(actuator_id)
            else:
                rotational.append(actuator_id)
        return rotational, linear

    def actuator_name(self, actuator_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or f"actuator {actuator_id}"

    def make_figure(self, title: str, actuator_ids: list[int], symmetric: bool = True) -> mujoco.MjvFigure:
        figure = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(figure)
        figure.title = title
        figure.xlabel = "time (s)"
        figure.xformat = "%.1f"
        figure.yformat = "%.2g"
        figure.flg_legend = 1
        figure.flg_symmetric = int(symmetric)
        figure.flg_extend = 0
        figure.gridsize[:] = (3, 3)
        figure.linewidth = 2.0
        figure.figurergba[:] = (0.05, 0.05, 0.05, 0.78)
        figure.panergba[:] = (0.0, 0.0, 0.0, 0.45)
        figure.legendrgba[:] = (0.0, 0.0, 0.0, 0.35)
        figure.range[0, :] = (-self.window_seconds, 0.0)
        figure.range[1, :] = (-1.0, 1.0)

        colors = np.array(
            [
                (1.0, 0.35, 0.25),
                (0.25, 0.75, 1.0),
                (0.3, 1.0, 0.45),
                (1.0, 0.8, 0.2),
            ],
            dtype=np.float32,
        )
        for line_id, actuator_id in enumerate(actuator_ids[: len(figure.linepnt)]):
            figure.linename[line_id] = self.actuator_name(actuator_id)[:99]
            figure.linergb[line_id, :] = colors[line_id % len(colors)]
        return figure

    def clear(self) -> None:
        self.times.clear()
        self.forces.clear()
        self.positions.clear()
        self.rotational_figure.linepnt[:] = 0
        self.linear_figure.linepnt[:] = 0
        self.linear_position_figure.linepnt[:] = 0

    def record(self, data: mujoco.MjData) -> None:
        self.times.append(float(data.time))
        self.forces.append(np.asarray(data.actuator_force, dtype=np.float64).copy())
        self.positions.append(self.linear_actuator_positions(data))

    def linear_actuator_positions(self, data: mujoco.MjData) -> np.ndarray:
        positions = np.full(self.model.nu, np.nan, dtype=np.float64)
        for actuator_id in self.linear_actuators:
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            qadr = int(self.model.jnt_qposadr[joint_id])
            positions[actuator_id] = float(data.qpos[qadr])
        return positions

    def actuator_position_limits(self, actuator_id: int) -> tuple[float, float] | None:
        joint_id = int(self.model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or not self.model.jnt_limited[joint_id]:
            return None
        lower, upper = self.model.jnt_range[joint_id]
        return float(lower), float(upper)

    def actuator_force_plot_range(self, actuator_ids: list[int]) -> tuple[float, float] | None:
        lower_values = []
        upper_values = []
        for actuator_id in actuator_ids:
            if not self.model.actuator_forcelimited[actuator_id]:
                continue
            lower, upper = self.model.actuator_forcerange[actuator_id]
            lower_values.append(float(lower))
            upper_values.append(float(upper))

        if not lower_values:
            return None

        return min(lower_values) - FORCE_PLOT_LIMIT_PADDING, max(upper_values) + FORCE_PLOT_LIMIT_PADDING

    def sampled_history(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not self.times:
            return None

        times = np.asarray(self.times, dtype=np.float64)
        forces = np.asarray(self.forces, dtype=np.float64)
        positions = np.asarray(self.positions, dtype=np.float64)
        relative_times = times - times[-1]
        mask = relative_times >= -self.window_seconds
        relative_times = relative_times[mask]
        forces = forces[mask]
        positions = positions[mask]

        if len(relative_times) > PLOT_MAX_POINTS:
            sample_indices = np.linspace(0, len(relative_times) - 1, PLOT_MAX_POINTS, dtype=np.int64)
            relative_times = relative_times[sample_indices]
            forces = forces[sample_indices]
            positions = positions[sample_indices]
        return relative_times, forces, positions

    def update_force_figure(
        self,
        figure: mujoco.MjvFigure,
        actuator_ids: list[int],
        history: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        relative_times, forces, _positions = history
        figure.linepnt[:] = 0
        if len(relative_times) == 0 or not actuator_ids:
            figure.range[0, :] = (-self.window_seconds, 0.0)
            figure.range[1, :] = self.actuator_force_plot_range(actuator_ids) or (-1.0, 1.0)
            return

        max_points = min(len(relative_times), figure.linedata.shape[1] // 2)
        max_abs = 0.0
        for line_id, actuator_id in enumerate(actuator_ids[: len(figure.linepnt)]):
            values = forces[-max_points:, actuator_id]
            times = relative_times[-max_points:]
            figure.linepnt[line_id] = max_points
            figure.linedata[line_id, 0 : 2 * max_points : 2] = times
            figure.linedata[line_id, 1 : 2 * max_points : 2] = values
            max_abs = max(max_abs, float(np.max(np.abs(values))))

        figure.range[0, :] = (-self.window_seconds, 0.0)
        force_range = self.actuator_force_plot_range(actuator_ids)
        if force_range is not None:
            figure.range[1, :] = force_range
        else:
            y_limit = max(1.0, 1.1 * max_abs)
            figure.range[1, :] = (-y_limit, y_limit)

    def update_position_figure(
        self,
        figure: mujoco.MjvFigure,
        history: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        relative_times, _forces, positions = history
        figure.linepnt[:] = 0
        if len(relative_times) == 0 or not self.linear_actuators:
            figure.range[0, :] = (-self.window_seconds, 0.0)
            figure.range[1, :] = LINEAR_POSITION_PLOT_RANGE
            return

        max_points = min(len(relative_times), figure.linedata.shape[1] // 2)
        times = relative_times[-max_points:]
        line_id = 0
        for actuator_id in self.linear_actuators:
            if line_id >= len(figure.linepnt):
                break

            values = positions[-max_points:, actuator_id]
            if np.all(np.isnan(values)):
                continue

            name = self.actuator_name(actuator_id)
            figure.linename[line_id] = name[:99]
            figure.linergb[line_id, :] = (0.25, 0.75, 1.0)
            figure.linepnt[line_id] = max_points
            figure.linedata[line_id, 0 : 2 * max_points : 2] = times
            figure.linedata[line_id, 1 : 2 * max_points : 2] = values
            line_id += 1

            limits = self.actuator_position_limits(actuator_id)
            if limits is not None:
                for limit_value, label, color in (
                    (limits[0], "min", (1.0, 0.35, 0.25)),
                    (limits[1], "max", (0.3, 1.0, 0.45)),
                ):
                    if line_id >= len(figure.linepnt):
                        break
                    figure.linename[line_id] = f"{name} {label}"[:99]
                    figure.linergb[line_id, :] = color
                    figure.linepnt[line_id] = max_points
                    figure.linedata[line_id, 0 : 2 * max_points : 2] = times
                    figure.linedata[line_id, 1 : 2 * max_points : 2] = limit_value
                    line_id += 1

            if self.desired_positions is None:
                continue

            desired_value = self.desired_positions[actuator_id]
            if not np.isfinite(desired_value) or line_id >= len(figure.linepnt):
                continue

            figure.linename[line_id] = f"{name} desired"[:99]
            figure.linergb[line_id, :] = (1.0, 0.8, 0.2)
            figure.linepnt[line_id] = max_points
            figure.linedata[line_id, 0 : 2 * max_points : 2] = times
            figure.linedata[line_id, 1 : 2 * max_points : 2] = desired_value
            line_id += 1

        figure.range[0, :] = (-self.window_seconds, 0.0)
        figure.range[1, :] = LINEAR_POSITION_PLOT_RANGE

    def render(self, width: int, height: int, context: mujoco.MjrContext) -> None:
        history = self.sampled_history()
        if history is None:
            return

        self.update_force_figure(self.rotational_figure, self.rotational_actuators, history)
        self.update_force_figure(self.linear_figure, self.linear_actuators, history)
        self.update_position_figure(self.linear_position_figure, history)

        figure_width = min(430, max(300, int(width * 0.32)))
        max_base_height = max(45, (height - 4 * PLOT_MARGIN) // 6)
        figure_height = min(180, max(70, int(height * 0.12)), max_base_height)
        force_height = 2 * figure_height
        linear_position_height = 2 * figure_height
        x = max(PLOT_MARGIN, width - figure_width - PLOT_MARGIN)
        rotational_y = max(PLOT_MARGIN, height - force_height - PLOT_MARGIN)
        linear_y = max(PLOT_MARGIN, rotational_y - force_height - PLOT_MARGIN)
        linear_position_y = max(PLOT_MARGIN, linear_y - linear_position_height - PLOT_MARGIN)

        mujoco.mjr_figure(
            mujoco.MjrRect(x, rotational_y, figure_width, force_height),
            self.rotational_figure,
            context,
        )
        mujoco.mjr_figure(
            mujoco.MjrRect(x, linear_y, figure_width, force_height),
            self.linear_figure,
            context,
        )
        mujoco.mjr_figure(
            mujoco.MjrRect(x, linear_position_y, figure_width, linear_position_height),
            self.linear_position_figure,
            context,
        )


class RuntimeConstraint:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.mode = CONSTRAINT_FREE
        self.free_qadr = first_freejoint_qadr(model)
        self.free_dofadr = first_freejoint_dofadr(model)
        self.vertical_xy = np.zeros(2)
        self.vertical_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.foot_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot"))
        self.foot_pin_local = np.zeros(3)
        self.foot_pin_world = np.zeros(3)
        self.ballscrew_qadr: int | None = None
        self.ballscrew_dofadr: int | None = None
        self.foot_pin_ballscrew_qpos = 0.0
        ballscrew_joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ballscrew"))
        if ballscrew_joint_id >= 0:
            self.ballscrew_qadr = int(model.jnt_qposadr[ballscrew_joint_id])
            self.ballscrew_dofadr = int(model.jnt_dofadr[ballscrew_joint_id])

    def set_mode(self, mode: str, data: mujoco.MjData) -> None:
        if mode == CONSTRAINT_VERTICAL and self.free_qadr is None:
            print("vertical-only mode unavailable: no freejoint")
            return
        if mode == CONSTRAINT_FOOT_PIN and (self.free_qadr is None or self.foot_body_id < 0):
            print("foot-pinned mode unavailable: no freejoint or foot body")
            return

        self.mode = mode
        self.capture(data)
        self.project(data)
        print(f"constraint mode: {self.mode}")

    def toggle_vertical(self, data: mujoco.MjData) -> None:
        self.set_mode(CONSTRAINT_FREE if self.mode == CONSTRAINT_VERTICAL else CONSTRAINT_VERTICAL, data)

    def toggle_foot_pin(self, data: mujoco.MjData) -> None:
        self.set_mode(CONSTRAINT_FREE if self.mode == CONSTRAINT_FOOT_PIN else CONSTRAINT_FOOT_PIN, data)

    def capture(self, data: mujoco.MjData) -> None:
        mujoco.mj_forward(self.model, data)
        if self.mode == CONSTRAINT_VERTICAL and self.free_qadr is not None:
            self.vertical_xy[:] = data.qpos[self.free_qadr : self.free_qadr + 2]
            self.vertical_quat[:] = data.qpos[self.free_qadr + 3 : self.free_qadr + 7]
        elif self.mode == CONSTRAINT_FOOT_PIN and self.foot_body_id >= 0:
            self.foot_pin_local[:], foot_world = lowest_body_point_local(self.model, data, self.foot_body_id)
            self.foot_pin_world[:] = foot_world
            self.foot_pin_world[2] = 0.0
            if self.ballscrew_qadr is not None:
                self.foot_pin_ballscrew_qpos = float(data.qpos[self.ballscrew_qadr])

    def recapture_active_mode(self, data: mujoco.MjData) -> None:
        if self.mode != CONSTRAINT_FREE:
            self.capture(data)
            self.project(data)

    def project(self, data: mujoco.MjData) -> None:
        if self.mode == CONSTRAINT_VERTICAL:
            self.project_vertical(data)
        elif self.mode == CONSTRAINT_FOOT_PIN:
            self.project_foot_pin(data)

    def project_vertical(self, data: mujoco.MjData) -> None:
        if self.free_qadr is None or self.free_dofadr is None:
            return

        data.qpos[self.free_qadr : self.free_qadr + 2] = self.vertical_xy
        data.qpos[self.free_qadr + 3 : self.free_qadr + 7] = self.vertical_quat
        constrained_dofs = [0, 1, 3, 4, 5]
        for offset in constrained_dofs:
            data.qvel[self.free_dofadr + offset] = 0.0
            data.qacc_warmstart[self.free_dofadr + offset] = 0.0
        mujoco.mj_forward(self.model, data)

    def foot_pin_world_point(self, data: mujoco.MjData) -> np.ndarray:
        foot_xmat = data.xmat[self.foot_body_id].reshape(3, 3)
        return data.xpos[self.foot_body_id] + foot_xmat @ self.foot_pin_local

    def foot_pin_world_point_with_captured_ballscrew(self, data: mujoco.MjData) -> np.ndarray:
        if self.ballscrew_qadr is None:
            return self.foot_pin_world_point(data)

        current_ballscrew_qpos = float(data.qpos[self.ballscrew_qadr])
        data.qpos[self.ballscrew_qadr] = self.foot_pin_ballscrew_qpos
        mujoco.mj_forward(self.model, data)
        foot_point = self.foot_pin_world_point(data)
        data.qpos[self.ballscrew_qadr] = current_ballscrew_qpos
        mujoco.mj_forward(self.model, data)
        return foot_point

    def project_foot_pin(self, data: mujoco.MjData) -> None:
        if self.free_qadr is None or self.free_dofadr is None or self.foot_body_id < 0:
            return

        mujoco.mj_forward(self.model, data)
        delta = self.foot_pin_world - self.foot_pin_world_point_with_captured_ballscrew(data)
        data.qpos[self.free_qadr : self.free_qadr + 3] += delta
        mujoco.mj_forward(self.model, data)

        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jacp, None, self.foot_pin_world, self.foot_body_id)
        pin_velocity = data.qvel.copy()
        if self.ballscrew_dofadr is not None:
            pin_velocity[self.ballscrew_dofadr] = 0.0
        data.qvel[self.free_dofadr : self.free_dofadr + 3] -= jacp @ pin_velocity
        mujoco.mj_forward(self.model, data)


class Viewer:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        args: argparse.Namespace,
        controller: BalanceController | None,
    ):
        self.model = model
        self.data = data
        self.args = args
        self.controller = controller
        self.balance_enabled = controller is not None and args.balance
        self.paused = False
        self.last_cursor = (0.0, 0.0)
        self.last_click_button: int | None = None
        self.last_click_time = 0.0
        self.last_click_pos = (0.0, 0.0)
        plot_reference = controller.design.reference if controller is not None else None
        self.force_plot = ActuatorForcePlot(model, args.force_plot_window, plot_reference) if args.force_plot else None
        self.force_plot_enabled = self.force_plot is not None
        self.runtime_constraint = RuntimeConstraint(model)
        self.free_qadr = first_freejoint_qadr(model)
        self.free_dofadr = first_freejoint_dofadr(model)
        self.base_body_id, _ = base_and_foot_body_ids(model)
        self.keyboard_perturbation_count = 0
        self.keyboard_perturbation_arrow_until = 0.0
        self.keyboard_perturbation_arrow_start = np.zeros(3, dtype=np.float64)
        self.keyboard_perturbation_arrow_end = np.zeros(3, dtype=np.float64)

        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW.")

        self.window = glfw.create_window(args.width, args.height, "jumpybot", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window.")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        mujoco.mjv_defaultCamera(self.camera)
        mujoco.mjv_defaultOption(self.option)
        self.camera.azimuth = 160
        self.camera.elevation = -20
        self.camera.distance = 0.8
        self.camera.lookat[:] = (0, 0, 0.1)

        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.option.geomgroup[COLLISION_GEOM_GROUP] = 0

        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor_pos)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_scroll_callback(self.window, self.on_scroll)

    def close(self) -> None:
        glfw.terminate()

    def collision_meshes_visible(self) -> bool:
        return bool(self.option.geomgroup[COLLISION_GEOM_GROUP])

    def toggle_collision_meshes(self) -> None:
        self.option.geomgroup[COLLISION_GEOM_GROUP] = 0 if self.collision_meshes_visible() else 1
        state = "shown" if self.collision_meshes_visible() else "hidden"
        print(f"collision meshes {state}")

    def visual_flag_enabled(self, flag: mujoco.mjtVisFlag) -> bool:
        return bool(self.option.flags[int(flag)])

    def toggle_visual_flag(self, name: str, flag: mujoco.mjtVisFlag) -> None:
        self.option.flags[int(flag)] = 0 if self.visual_flag_enabled(flag) else 1
        state = "shown" if self.visual_flag_enabled(flag) else "hidden"
        print(f"{name} {state}")

    def selected_body_name(self) -> str:
        if self.perturb.select <= 0:
            return "none"
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.perturb.select)
        return name or f"body {self.perturb.select}"

    def perturb_mode_name(self) -> str:
        if self.perturb.active & int(mujoco.mjtPertBit.mjPERT_ROTATE):
            return "torque"
        if self.perturb.active & int(mujoco.mjtPertBit.mjPERT_TRANSLATE):
            return "force"
        return "none"

    def update_scene(self) -> None:
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            self.perturb,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

    def click_is_double_click(self, button: int, xpos: float, ypos: float) -> bool:
        now = time.perf_counter()
        dx = xpos - self.last_click_pos[0]
        dy = ypos - self.last_click_pos[1]
        is_double_click = (
            self.last_click_button == button
            and now - self.last_click_time <= DOUBLE_CLICK_SECONDS
            and dx * dx + dy * dy <= DOUBLE_CLICK_PIXELS * DOUBLE_CLICK_PIXELS
        )
        self.last_click_button = None if is_double_click else button
        self.last_click_time = 0.0 if is_double_click else now
        self.last_click_pos = (xpos, ypos)
        return is_double_click

    def reset_simulation(self) -> None:
        if self.controller is not None:
            reset_to_lqr_reference(self.model, self.data, self.controller.design)
            if not self.balance_enabled:
                self.data.ctrl[:] = 0.0
            self.controller.reset(self.data)
            message = "reset balance equilibrium"
        else:
            reset_upright(self.model, self.data, self.args.reset_clearance)
            message = "reset upright"

        self.runtime_constraint.recapture_active_mode(self.data)
        if self.force_plot is not None:
            self.force_plot.clear()
            self.force_plot.record(self.data)
        print(message)

    def select_body_at(self, xpos: float, ypos: float) -> None:
        width, height = glfw.get_window_size(self.window)
        if width <= 0 or height <= 0:
            return

        self.update_scene()

        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        flexid = np.array([-1], dtype=np.int32)
        skinid = np.array([-1], dtype=np.int32)
        relx = min(max(xpos / width, 0.0), 1.0)
        rely = min(max(1.0 - ypos / height, 0.0), 1.0)
        body_id = int(
            mujoco.mjv_select(
                self.model,
                self.data,
                self.option,
                width / height,
                relx,
                rely,
                self.scene,
                selpnt,
                geomid,
                flexid,
                skinid,
            )
        )

        self.perturb.active = 0
        if body_id >= 0:
            self.perturb.select = body_id
            self.perturb.flexselect = int(flexid[0])
            self.perturb.skinselect = int(skinid[0])
            if body_id > 0:
                body_xmat = self.data.xmat[body_id].reshape(3, 3)
                self.perturb.localpos[:] = body_xmat.T @ (selpnt - self.data.xpos[body_id])
            print(f"selected {self.selected_body_name()}")
        else:
            self.perturb.select = 0
            self.perturb.flexselect = -1
            self.perturb.skinselect = -1
            print("selection cleared")

    def control_down(self) -> bool:
        return (
            glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
            or glfw.get_key(self.window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
        )

    def start_perturb(self, button: int) -> bool:
        if not self.control_down() or self.perturb.select <= 0:
            return False

        active = 0
        if button == glfw.MOUSE_BUTTON_LEFT:
            active = int(mujoco.mjtPertBit.mjPERT_ROTATE)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            active = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)

        if not active:
            return False

        self.update_scene()
        mujoco.mjv_initPerturb(self.model, self.data, self.scene, self.perturb)
        self.perturb.active = active
        print(f"applying {self.perturb_mode_name()} to {self.selected_body_name()}")
        return True

    def apply_perturb_force(self) -> None:
        self.data.xfrc_applied[:] = 0
        if self.perturb.active:
            mujoco.mjv_applyPerturbForce(self.model, self.data, self.perturb)

    def keyboard_perturbation_origin(self) -> np.ndarray:
        if self.base_body_id is not None:
            origin = self.data.xpos[self.base_body_id].copy()
        elif self.free_qadr is not None:
            origin = self.data.qpos[self.free_qadr : self.free_qadr + 3].copy()
        else:
            origin = np.zeros(3, dtype=np.float64)
        origin[2] += KEYBOARD_PERTURBATION_ARROW_HEIGHT
        return origin

    def draw_keyboard_perturbation_arrow(self) -> None:
        if time.perf_counter() > self.keyboard_perturbation_arrow_until:
            return
        if self.scene.ngeom >= self.scene.maxgeom:
            return

        geom = self.scene.geoms[self.scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(9),
            KEYBOARD_PERTURBATION_ARROW_RGBA,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            KEYBOARD_PERTURBATION_ARROW_WIDTH,
            self.keyboard_perturbation_arrow_start,
            self.keyboard_perturbation_arrow_end,
        )
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        self.scene.ngeom += 1

    def apply_keyboard_perturbation(self) -> None:
        if self.free_qadr is None or self.free_dofadr is None:
            print("keyboard perturbation unavailable: no freejoint")
            return

        direction = KEYBOARD_PERTURBATION_DIRECTIONS[
            self.keyboard_perturbation_count % len(KEYBOARD_PERTURBATION_DIRECTIONS)
        ]
        self.keyboard_perturbation_count += 1
        world_delta = rotate_xy_yaw_frame_to_world(self.data.qpos[self.free_qadr + 3 : self.free_qadr + 7], direction)
        self.data.qvel[self.free_dofadr : self.free_dofadr + 2] += KEYBOARD_PERTURBATION_DELTA_V * world_delta
        mujoco.mj_forward(self.model, self.data)
        arrow_start = self.keyboard_perturbation_origin()
        arrow_direction = np.array([world_delta[0], world_delta[1], 0.0], dtype=np.float64)
        self.keyboard_perturbation_arrow_start[:] = arrow_start
        self.keyboard_perturbation_arrow_end[:] = arrow_start + KEYBOARD_PERTURBATION_ARROW_LENGTH * arrow_direction
        self.keyboard_perturbation_arrow_until = time.perf_counter() + KEYBOARD_PERTURBATION_ARROW_SECONDS
        print(
            "applied perturbation "
            f"delta_v={KEYBOARD_PERTURBATION_DELTA_V:.3f} m/s "
            f"world_xy=({world_delta[0]:.3f}, {world_delta[1]:.3f})"
        )

    def on_key(self, window, key: int, _scancode: int, action: int, _mods: int) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif key == glfw.KEY_B:
            if self.controller is not None:
                self.balance_enabled = not self.balance_enabled
                if not self.balance_enabled:
                    self.data.ctrl[:] = 0
                    self.controller.reset(self.data)
                print(f"balance controller {'enabled' if self.balance_enabled else 'disabled'}")
        elif key == glfw.KEY_G:
            if self.force_plot is not None:
                self.force_plot_enabled = not self.force_plot_enabled
                print(f"actuator force plots {'shown' if self.force_plot_enabled else 'hidden'}")
        elif key == glfw.KEY_V:
            self.runtime_constraint.toggle_vertical(self.data)
        elif key == glfw.KEY_T:
            self.runtime_constraint.toggle_foot_pin(self.data)
        elif key == glfw.KEY_J:
            self.apply_keyboard_perturbation()
        elif key in (glfw.KEY_3, glfw.KEY_KP_3):
            self.toggle_collision_meshes()
        elif key in STANDARD_VIS_FLAG_KEYS:
            self.toggle_visual_flag(*STANDARD_VIS_FLAG_KEYS[key])
        elif key == glfw.KEY_R:
            self.reset_simulation()

    def on_mouse_button(self, _window, button: int, action: int, _mods: int) -> None:
        xpos, ypos = glfw.get_cursor_pos(self.window)
        self.last_cursor = (xpos, ypos)

        if action == glfw.PRESS:
            if self.start_perturb(button):
                return
            if self.click_is_double_click(button, xpos, ypos):
                self.select_body_at(xpos, ypos)
        elif action == glfw.RELEASE and button in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            self.perturb.active = 0
            self.data.xfrc_applied[:] = 0

    def on_cursor_pos(self, _window, xpos: float, ypos: float) -> None:
        left = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        middle = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        right = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not (left or middle or right):
            self.last_cursor = (xpos, ypos)
            return

        width, height = glfw.get_window_size(self.window)
        dx = xpos - self.last_cursor[0]
        dy = ypos - self.last_cursor[1]
        self.last_cursor = (xpos, ypos)

        shift = (
            glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif middle:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V

        if self.perturb.active:
            mujoco.mjv_movePerturb(
                self.model,
                self.data,
                action,
                dx / max(width, 1),
                dy / max(height, 1),
                self.scene,
                self.perturb,
            )
        else:
            mujoco.mjv_moveCamera(self.model, action, dx / max(width, 1), dy / max(height, 1), self.scene, self.camera)

    def on_scroll(self, _window, _xoffset: float, yoffset: float) -> None:
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, self.scene, self.camera)

    def render(self) -> None:
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        self.apply_perturb_force()
        self.update_scene()
        self.draw_keyboard_perturbation_arrow()
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            viewport,
            (
                "R: reset\n"
                "B: balance on/off\n"
                "G: actuator plots\n"
                "V: vertical-only root\n"
                "T: foot-pinned pivot\n"
                "J: perturb robot\n"
                "3: collision meshes\n"
                "C: contact points\n"
                "F: contact forces\n"
                "P: split contact forces\n"
                "Double-click: select\n"
                "Ctrl+Left: torque\n"
                "Ctrl+Right: force\n"
                "Space: pause/resume\n"
                "Esc: quit"
            ),
            (
                f"time: {self.data.time:.3f}\n"
                f"paused: {self.paused}\n"
                f"balance: {self.balance_enabled}\n"
                f"constraint: {self.runtime_constraint.mode}\n"
                f"actuator plots: {self.force_plot_enabled}\n"
                f"collision meshes: {self.collision_meshes_visible()}\n"
                f"contact points: {self.visual_flag_enabled(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT)}\n"
                f"contact forces: {self.visual_flag_enabled(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE)}\n"
                f"split forces: {self.visual_flag_enabled(mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT)}\n"
                f"selected: {self.selected_body_name()}\n"
                f"perturb: {self.perturb_mode_name()}\n"
                f"ctrl: {np.array2string(self.data.ctrl, precision=2, suppress_small=True)}"
            ),
            self.context,
        )
        if self.force_plot_enabled and self.force_plot is not None:
            self.force_plot.render(width, height, self.context)

    def run(self) -> None:
        self.reset_simulation()
        frame_dt = 1.0 / 60.0
        while not glfw.window_should_close(self.window):
            frame_start = time.perf_counter()
            if not self.paused:
                sim_start = self.data.time
                while self.data.time - sim_start < frame_dt:
                    self.runtime_constraint.project(self.data)
                    self.apply_perturb_force()
                    if self.balance_enabled and self.controller is not None:
                        self.controller.apply(self.data)
                    mujoco.mj_step(self.model, self.data)
                    self.runtime_constraint.project(self.data)
                    if self.force_plot is not None:
                        self.force_plot.record(self.data)

            self.render()
            glfw.swap_buffers(self.window)
            glfw.poll_events()

            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    controller = None

    print(f"loaded {args.model}: nbody={model.nbody} ngeom={model.ngeom} njnt={model.njnt} nu={model.nu}")
    if args.balance:
        design = design_balance_lqr(model, args)
        print_lqr_summary(design)
        controller = BalanceController(
            model,
            design,
            args.control_limit,
            args.flywheel_control_limit,
            args.flywheel_velocity_damping,
            args.flywheel_slew_rate,
        )

    if args.dry_run:
        return

    print(
        "press R to reset, B to toggle balance, G to toggle actuator plots, "
        "V for vertical-only root, T for foot-pinned pivot, 3 to toggle collision meshes, "
        "J for robot perturbation, C for contact points, F for contact forces, P to split contact forces, "
        "Space to pause, Esc to quit"
    )

    viewer = Viewer(model, data, args, controller)
    try:
        viewer.run()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
