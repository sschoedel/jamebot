from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import glfw
import mujoco
import numpy as np


PLOT_MARGIN = 12
PLOT_MAX_POINTS = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kinematic MuJoCo joint range debugger.")
    parser.add_argument("model", type=Path, nargs="?", default=Path("robot_model/scene.xml"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--clearance", type=float, default=0.0, help="Lowest robot mesh height at the center pose.")
    parser.add_argument("--ballscrew-joint", default="ballscrew")
    parser.add_argument("--spring-joint", default="spring")
    parser.add_argument("--ballscrew-step", type=float, default=None)
    parser.add_argument("--spring-step", type=float, default=None)
    parser.add_argument("--plot-window", type=float, default=8.0)
    parser.add_argument("--sweep-seconds", type=float, default=4.0, help="Seconds for one min-to-max auto sweep.")
    parser.add_argument("--dry-run", action="store_true", help="Load the model, print joint ranges, and exit.")
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


def joint_id(model: mujoco.MjModel, name: str) -> int:
    found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if found < 0:
        raise ValueError(f"joint {name!r} not found")
    return int(found)


def joint_range(model: mujoco.MjModel, joint_id_: int) -> tuple[float, float] | None:
    if not model.jnt_limited[joint_id_]:
        return None
    lower, upper = model.jnt_range[joint_id_]
    return float(lower), float(upper)


def joint_center(model: mujoco.MjModel, joint_id_: int) -> float:
    limits = joint_range(model, joint_id_)
    if limits is None:
        return 0.0
    lower, upper = limits
    return 0.5 * (lower + upper)


def joint_qadr(model: mujoco.MjModel, joint_id_: int) -> int:
    joint_type = model.jnt_type[joint_id_]
    if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id_) or str(joint_id_)
        raise ValueError(f"joint {name!r} must be a scalar hinge or slide joint")
    return int(model.jnt_qposadr[joint_id_])


def clamp_to_range(model: mujoco.MjModel, joint_id_: int, value: float) -> float:
    limits = joint_range(model, joint_id_)
    if limits is None:
        return value
    lower, upper = limits
    return min(max(value, lower), upper)


def default_step(model: mujoco.MjModel, joint_id_: int, fallback: float) -> float:
    limits = joint_range(model, joint_id_)
    if limits is None:
        return fallback
    lower, upper = limits
    return max(fallback, 0.01 * abs(upper - lower))


def set_internal_joint_centers(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for joint_id_ in range(model.njnt):
        joint_type = model.jnt_type[joint_id_]
        qadr = int(model.jnt_qposadr[joint_id_])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            data.qpos[qadr : qadr + 4] = (1.0, 0.0, 0.0, 0.0)
            continue
        data.qpos[qadr] = joint_center(model, joint_id_)


class BallscrewPlot:
    def __init__(
        self,
        window_seconds: float,
        limits: tuple[float, float] | None,
        start_value: float,
    ):
        self.window_seconds = max(window_seconds, 1e-3)
        self.limits = limits
        self.start_value = start_value
        self.times: deque[float] = deque(maxlen=max(2, int(120 * self.window_seconds)))
        self.values: deque[float] = deque(maxlen=max(2, int(120 * self.window_seconds)))
        self.figure = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(self.figure)
        self.figure.title = "Ballscrew joint coordinate / length (m)"
        self.figure.xlabel = "wall time (s)"
        self.figure.xformat = "%.1f"
        self.figure.yformat = "%.3g"
        self.figure.flg_legend = 1
        self.figure.flg_symmetric = 0
        self.figure.flg_extend = 0
        self.figure.gridsize[:] = (3, 3)
        self.figure.linewidth = 2.0
        self.figure.figurergba[:] = (0.05, 0.05, 0.05, 0.78)
        self.figure.panergba[:] = (0.0, 0.0, 0.0, 0.45)
        self.figure.legendrgba[:] = (0.0, 0.0, 0.0, 0.35)
        self.figure.linename[0] = "ballscrew"
        self.figure.linergb[0, :] = (0.25, 0.75, 1.0)
        self.figure.linename[1] = "min"
        self.figure.linergb[1, :] = (1.0, 0.35, 0.25)
        self.figure.linename[2] = "max"
        self.figure.linergb[2, :] = (0.3, 1.0, 0.45)
        self.figure.linename[3] = "start"
        self.figure.linergb[3, :] = (1.0, 0.8, 0.2)

    def record(self, now: float, value: float) -> None:
        self.times.append(now)
        self.values.append(value)

    def render(self, width: int, height: int, context: mujoco.MjrContext) -> None:
        if not self.times:
            return

        times = np.asarray(self.times, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        relative_times = times - times[-1]
        mask = relative_times >= -self.window_seconds
        relative_times = relative_times[mask]
        values = values[mask]
        if len(relative_times) > PLOT_MAX_POINTS:
            indices = np.linspace(0, len(relative_times) - 1, PLOT_MAX_POINTS, dtype=np.int64)
            relative_times = relative_times[indices]
            values = values[indices]

        max_points = min(len(relative_times), self.figure.linedata.shape[1] // 2)
        times = relative_times[-max_points:]
        values = values[-max_points:]
        y_values = [float(value) for value in values]
        self.figure.linepnt[:] = 0
        self.write_line(0, times, values)

        line_id = 1
        if self.limits is not None:
            for limit_value in self.limits:
                self.write_line(line_id, times, np.full(max_points, limit_value))
                y_values.append(limit_value)
                line_id += 1
        self.write_line(3, times, np.full(max_points, self.start_value))
        y_values.append(self.start_value)

        y_min = min(y_values)
        y_max = max(y_values)
        margin = max(0.001, 0.1 * max(y_max - y_min, 1e-9))
        self.figure.range[0, :] = (-self.window_seconds, 0.0)
        self.figure.range[1, :] = (y_min - margin, y_max + margin)

        plot_width = min(620, max(360, int(width * 0.46)))
        plot_height = min(220, max(150, int(height * 0.24)))
        rect = mujoco.MjrRect(
            width - plot_width - PLOT_MARGIN,
            height - plot_height - PLOT_MARGIN,
            plot_width,
            plot_height,
        )
        mujoco.mjr_figure(rect, self.figure, context)

    def write_line(self, line_id: int, times: np.ndarray, values: np.ndarray) -> None:
        count = min(len(times), len(values), self.figure.linedata.shape[1] // 2)
        if count <= 0:
            return
        self.figure.linepnt[line_id] = count
        self.figure.linedata[line_id, 0 : 2 * count : 2] = times[-count:]
        self.figure.linedata[line_id, 1 : 2 * count : 2] = values[-count:]


class KinematicJointViewer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace):
        self.model = model
        self.data = data
        self.args = args
        self.free_qadr = first_freejoint_qadr(model)
        self.free_dofadr = first_freejoint_dofadr(model)
        self.ballscrew_joint = joint_id(model, args.ballscrew_joint)
        self.spring_joint = joint_id(model, args.spring_joint)
        self.ballscrew_qadr = joint_qadr(model, self.ballscrew_joint)
        self.spring_qadr = joint_qadr(model, self.spring_joint)
        self.ballscrew_limits = joint_range(model, self.ballscrew_joint)
        self.spring_limits = joint_range(model, self.spring_joint)
        self.ballscrew_step = args.ballscrew_step or default_step(model, self.ballscrew_joint, 0.001)
        self.spring_step = args.spring_step or default_step(model, self.spring_joint, 0.0002)
        self.ballscrew_value = joint_center(model, self.ballscrew_joint)
        self.spring_value = joint_center(model, self.spring_joint)
        self.ballscrew_start = self.ballscrew_value
        self.spring_start = self.spring_value
        self.root_qpos = np.zeros(7)
        self.auto_ballscrew = False
        self.auto_spring = False
        self.ballscrew_direction = 1.0
        self.spring_direction = 1.0
        self.last_cursor = (0.0, 0.0)

        self.reset_pose()
        self.plot = BallscrewPlot(args.plot_window, self.ballscrew_limits, self.ballscrew_start)

        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW.")

        self.window = glfw.create_window(args.width, args.height, "jumpybot joint range debug", None, None)
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
        self.camera.lookat[:] = (0.0, 0.0, 0.1)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor_pos)
        glfw.set_scroll_callback(self.window, self.on_scroll)

    def close(self) -> None:
        glfw.terminate()

    def reset_pose(self) -> None:
        self.data.qpos[:] = self.model.qpos0
        set_internal_joint_centers(self.model, self.data)
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        self.data.qacc_warmstart[:] = 0
        self.data.qpos[self.ballscrew_qadr] = self.ballscrew_start
        self.data.qpos[self.spring_qadr] = self.spring_start

        if self.free_qadr is not None:
            self.data.qpos[self.free_qadr : self.free_qadr + 3] = (0.0, 0.0, 0.0)
            self.data.qpos[self.free_qadr + 3 : self.free_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
            mujoco.mj_forward(self.model, self.data)
            self.data.qpos[self.free_qadr + 2] += self.args.clearance - robot_min_z(self.model, self.data)
            self.root_qpos[:] = self.data.qpos[self.free_qadr : self.free_qadr + 7]

        self.apply_joint_values()

    def apply_joint_values(self) -> None:
        self.ballscrew_value = clamp_to_range(self.model, self.ballscrew_joint, self.ballscrew_value)
        self.spring_value = clamp_to_range(self.model, self.spring_joint, self.spring_value)
        if self.free_qadr is not None:
            self.data.qpos[self.free_qadr : self.free_qadr + 7] = self.root_qpos
        if self.free_dofadr is not None:
            self.data.qvel[self.free_dofadr : self.free_dofadr + 6] = 0.0
        self.data.qpos[self.ballscrew_qadr] = self.ballscrew_value
        self.data.qpos[self.spring_qadr] = self.spring_value
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset_joint_values(self) -> None:
        self.ballscrew_value = self.ballscrew_start
        self.spring_value = self.spring_start
        self.auto_ballscrew = False
        self.auto_spring = False
        self.apply_joint_values()

    def set_joint_value(self, which: str, value: float) -> None:
        if which == "ballscrew":
            self.ballscrew_value = value
        elif which == "spring":
            self.spring_value = value
        self.apply_joint_values()

    def nudge(self, which: str, direction: float, mods: int) -> None:
        multiplier = 10.0 if mods & glfw.MOD_SHIFT else 1.0
        if which == "ballscrew":
            self.ballscrew_value += direction * self.ballscrew_step * multiplier
        elif which == "spring":
            self.spring_value += direction * self.spring_step * multiplier
        self.apply_joint_values()

    def sweep(self, which: str, dt: float) -> None:
        if which == "ballscrew":
            limits = self.ballscrew_limits
            if limits is None:
                return
            lower, upper = limits
            span = upper - lower
            self.ballscrew_value += self.ballscrew_direction * span * dt / max(self.args.sweep_seconds, 1e-6)
            if self.ballscrew_value >= upper:
                self.ballscrew_value = upper
                self.ballscrew_direction = -1.0
            elif self.ballscrew_value <= lower:
                self.ballscrew_value = lower
                self.ballscrew_direction = 1.0
        elif which == "spring":
            limits = self.spring_limits
            if limits is None:
                return
            lower, upper = limits
            span = upper - lower
            self.spring_value += self.spring_direction * span * dt / max(self.args.sweep_seconds, 1e-6)
            if self.spring_value >= upper:
                self.spring_value = upper
                self.spring_direction = -1.0
            elif self.spring_value <= lower:
                self.spring_value = lower
                self.spring_direction = 1.0

    def limit_value(self, which: str, index: int) -> float | None:
        limits = self.ballscrew_limits if which == "ballscrew" else self.spring_limits
        if limits is None:
            return None
        return limits[index]

    def on_key(self, window, key: int, _scancode: int, action: int, mods: int) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            self.reset_joint_values()
        elif key == glfw.KEY_A and action == glfw.PRESS:
            self.auto_ballscrew = not self.auto_ballscrew
        elif key == glfw.KEY_S and action == glfw.PRESS:
            self.auto_spring = not self.auto_spring
        elif key == glfw.KEY_UP:
            self.nudge("ballscrew", 1.0, mods)
        elif key == glfw.KEY_DOWN:
            self.nudge("ballscrew", -1.0, mods)
        elif key == glfw.KEY_RIGHT:
            self.nudge("spring", 1.0, mods)
        elif key == glfw.KEY_LEFT:
            self.nudge("spring", -1.0, mods)
        elif key == glfw.KEY_1 and self.limit_value("ballscrew", 0) is not None:
            self.set_joint_value("ballscrew", self.limit_value("ballscrew", 0))
        elif key == glfw.KEY_2:
            self.set_joint_value("ballscrew", self.ballscrew_start)
        elif key == glfw.KEY_3 and self.limit_value("ballscrew", 1) is not None:
            self.set_joint_value("ballscrew", self.limit_value("ballscrew", 1))
        elif key == glfw.KEY_4 and self.limit_value("spring", 0) is not None:
            self.set_joint_value("spring", self.limit_value("spring", 0))
        elif key == glfw.KEY_5:
            self.set_joint_value("spring", self.spring_start)
        elif key == glfw.KEY_6 and self.limit_value("spring", 1) is not None:
            self.set_joint_value("spring", self.limit_value("spring", 1))

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
            mouse_action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif middle:
            mouse_action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            mouse_action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        mujoco.mjv_moveCamera(
            self.model,
            mouse_action,
            dx / max(width, 1),
            dy / max(height, 1),
            self.scene,
            self.camera,
        )

    def on_scroll(self, _window, _xoffset: float, yoffset: float) -> None:
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, self.scene, self.camera)

    def render(self, now: float) -> None:
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            self.perturb,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            viewport,
            (
                "No physics: mj_forward only\n"
                "Root freejoint fixed\n"
                "Up/Down: ballscrew\n"
                "Left/Right: spring\n"
                "Shift: 10x step\n"
                "1/2/3: ballscrew min/center/max\n"
                "4/5/6: spring min/center/max\n"
                "A: auto-sweep ballscrew\n"
                "S: auto-sweep spring\n"
                "R: reset centers\n"
                "Esc: quit"
            ),
            (
                f"ballscrew q: {self.ballscrew_value:.6g}\n"
                f"ballscrew start: {self.ballscrew_start:.6g}\n"
                f"ballscrew range: {self.format_range(self.ballscrew_limits)}\n"
                f"ballscrew step: {self.ballscrew_step:.6g}\n"
                f"ballscrew auto: {self.auto_ballscrew}\n"
                f"spring q: {self.spring_value:.6g}\n"
                f"spring start: {self.spring_start:.6g}\n"
                f"spring range: {self.format_range(self.spring_limits)}\n"
                f"spring step: {self.spring_step:.6g}\n"
                f"spring auto: {self.auto_spring}\n"
                f"time: {now:.3f}"
            ),
            self.context,
        )
        self.plot.render(width, height, self.context)

    @staticmethod
    def format_range(limits: tuple[float, float] | None) -> str:
        if limits is None:
            return "unlimited"
        return f"[{limits[0]:.6g}, {limits[1]:.6g}]"

    def run(self) -> None:
        start = time.perf_counter()
        last = start
        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            dt = now - last
            last = now
            elapsed = now - start
            if self.auto_ballscrew:
                self.sweep("ballscrew", dt)
            if self.auto_spring:
                self.sweep("spring", dt)
            self.apply_joint_values()
            self.plot.record(elapsed, self.ballscrew_value)
            self.render(elapsed)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            sleep_time = max(0.0, (1.0 / 60.0) - (time.perf_counter() - now))
            time.sleep(sleep_time)


def print_joint_summary(model: mujoco.MjModel, joint_name: str) -> None:
    found = joint_id(model, joint_name)
    qadr = joint_qadr(model, found)
    limits = joint_range(model, found)
    axis = model.jnt_axis[found]
    print(
        f"{joint_name}: qposadr={qadr} axis={np.array2string(axis, precision=6)} "
        f"range={KinematicJointViewer.format_range(limits)} start={joint_center(model, found):.6g}"
    )


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    print(f"loaded {args.model}: nbody={model.nbody} ngeom={model.ngeom} njnt={model.njnt} nu={model.nu}")
    print_joint_summary(model, args.ballscrew_joint)
    print_joint_summary(model, args.spring_joint)
    if args.dry_run:
        return
    viewer = KinematicJointViewer(model, data, args)
    try:
        viewer.run()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
