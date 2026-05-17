"""Add perturbation controls and force visualization to mjlab's viser viewer.

Importing this module monkey-patches ``mjlab.viewer.viser.viewer.ViserPlayViewer``
to:

- Append a 'Jamebot Perturb' folder with direction/magnitude sliders, an
  'Apply push' button (uses both sliders), and a 'Random xy push' button
  (random angle, magnitude from slider). Manual pushes are timed in
  **simulation time** (not wall clock), so pausing or slowing playback
  doesn't truncate them.
- Draw a red arrow on the body link whenever any external force is applied
  via ``xfrc_applied`` -- covers both the buttons here and the auto-push
  event (``apply_body_impulse``). The arrow updates each frame inside
  ``sync_env_to_viewer`` and hides itself when the force drops near zero.

The patch is class-level and idempotent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import viser

from mjlab.viewer.viser.viewer import ViserPlayViewer

if TYPE_CHECKING:
  from mjlab.entity import Entity

# Force application via the manual buttons.
FORCE_DURATION_S = 0.1  # in SIM time, not wall clock
DEFAULT_MAGNITUDE_N = 1.0
MAX_MAGNITUDE_N = 5.0

# Arrow visualization.
ARROW_M_PER_N = 0.05  # arrow length per Newton of force
ARROW_Z_OFFSET = 0.13  # body link origin -> chassis CoM height
ARROW_COLOR = (1.0, 0.1, 0.1)
ARROW_FORCE_THRESHOLD_N = 0.05  # hide arrow below this magnitude
ARROW_NAME = "/jamebot_force_arrow"

_PATCH_FLAG = "_jamebot_perturb_patch_applied"


@dataclass
class _PendingPush:
  forces: torch.Tensor
  torques: torch.Tensor
  body_ids: list[int]
  asset: "Entity"
  target_sim_time: float


# Per-viewer pending-clear state. Keyed by id(viewer).
_pending_pushes: dict[int, _PendingPush] = {}


def _sim_time(viewer: ViserPlayViewer) -> float:
  """Current simulation time (in seconds) for the displayed env."""
  env = viewer.env.unwrapped
  env_idx = int(viewer.cfg.env_idx)
  return float(env.episode_length_buf[env_idx].item()) * float(env.step_dt)


def _apply_xy_force(
  viewer: ViserPlayViewer, angle_rad: float, magnitude_n: float
) -> None:
  if magnitude_n <= 0.0:
    return

  env = viewer.env.unwrapped
  asset: Entity = env.scene["robot"]
  device = asset.data.root_link_pos_w.device
  num_envs = env.num_envs

  body_ids, _ = asset.find_bodies("body")
  num_bodies = len(body_ids)

  fx = magnitude_n * math.cos(angle_rad)
  fy = magnitude_n * math.sin(angle_rad)
  forces = torch.zeros((num_envs, num_bodies, 3), device=device)
  torques = torch.zeros_like(forces)
  forces[:, 0, 0] = fx
  forces[:, 0, 1] = fy

  with viewer._sim_lock:
    asset.write_external_wrench_to_sim(forces, torques, body_ids=body_ids)

  _pending_pushes[id(viewer)] = _PendingPush(
    forces=forces,
    torques=torques,
    body_ids=body_ids,
    asset=asset,
    target_sim_time=_sim_time(viewer) + FORCE_DURATION_S,
  )


def _maybe_clear_pending(viewer: ViserPlayViewer) -> None:
  pending = _pending_pushes.get(id(viewer))
  if pending is None:
    return
  if _sim_time(viewer) < pending.target_sim_time:
    return
  zero_f = torch.zeros_like(pending.forces)
  zero_t = torch.zeros_like(pending.torques)
  try:
    with viewer._sim_lock:
      pending.asset.write_external_wrench_to_sim(
        zero_f, zero_t, body_ids=pending.body_ids
      )
  except Exception:
    pass
  _pending_pushes.pop(id(viewer), None)


def _ensure_arrow_handle(viewer: ViserPlayViewer) -> viser.ArrowsHandle:
  """Lazily create the force arrow handle, hidden by default."""
  handle = getattr(viewer, "_jamebot_force_arrow", None)
  if handle is not None:
    return handle
  handle = viewer._server.scene.add_arrows(
    ARROW_NAME,
    points=np.array([[[0, 0, 0], [0, 0, 0]]], dtype=np.float32),
    colors=np.array(ARROW_COLOR, dtype=np.float32),
    shaft_radius=0.005,
    head_radius=0.012,
    head_length=0.025,
    visible=False,
  )
  viewer._jamebot_force_arrow = handle  # type: ignore[attr-defined]
  return handle


def _update_force_arrow(viewer: ViserPlayViewer) -> None:
  """Poll body_external_wrench and update the arrow each frame."""
  env = viewer.env.unwrapped
  if "robot" not in env.scene._entities:
    return
  asset: Entity = env.scene["robot"]

  body_ids, _ = asset.find_bodies("body")
  body_id = body_ids[0]
  env_idx = int(viewer.cfg.env_idx)

  # body_external_wrench shape: (nworld, nbody, 6) -- (fx, fy, fz, tx, ty, tz).
  wrench = asset.data.body_external_wrench[env_idx, body_id, :3]
  force = wrench.detach().cpu().numpy().astype(np.float32)
  magnitude = float(np.linalg.norm(force))

  handle = _ensure_arrow_handle(viewer)
  if magnitude < ARROW_FORCE_THRESHOLD_N:
    if handle.visible:
      handle.visible = False
    return

  base = asset.data.root_link_pos_w[env_idx].detach().cpu().numpy().astype(np.float32)
  start = np.array([base[0], base[1], base[2] + ARROW_Z_OFFSET], dtype=np.float32)
  end = start + force * ARROW_M_PER_N
  handle.points = np.array([[start, end]], dtype=np.float32)
  if not handle.visible:
    handle.visible = True


def _install_patch() -> None:
  if getattr(ViserPlayViewer, _PATCH_FLAG, False):
    return

  original_setup = ViserPlayViewer.setup
  original_sync = ViserPlayViewer.sync_env_to_viewer

  def patched_setup(self: ViserPlayViewer) -> None:
    original_setup(self)
    with self._server.gui.add_folder("Jamebot Perturb"):
      direction_slider = self._server.gui.add_slider(
        "Direction (rad)",
        min=0.0,
        max=2.0 * math.pi,
        step=0.01,
        initial_value=0.0,
      )
      magnitude_slider = self._server.gui.add_slider(
        "Magnitude (N)",
        min=0.0,
        max=MAX_MAGNITUDE_N,
        step=0.1,
        initial_value=DEFAULT_MAGNITUDE_N,
      )

      apply_btn = self._server.gui.add_button(
        "Apply push",
        icon=viser.Icon.WIND,
      )

      @apply_btn.on_click
      def _(_) -> None:
        _apply_xy_force(self, direction_slider.value, magnitude_slider.value)

      random_btn = self._server.gui.add_button(
        "Random xy push",
        icon=viser.Icon.ARROWS_SHUFFLE,
      )

      @random_btn.on_click
      def _(_) -> None:
        angle = float(torch.rand(1).item()) * 2.0 * math.pi
        _apply_xy_force(self, angle, magnitude_slider.value)

  def patched_sync(self: ViserPlayViewer) -> None:
    original_sync(self)
    try:
      _maybe_clear_pending(self)
      _update_force_arrow(self)
    except Exception:
      pass

  ViserPlayViewer.setup = patched_setup  # type: ignore[method-assign]
  ViserPlayViewer.sync_env_to_viewer = patched_sync  # type: ignore[method-assign]
  setattr(ViserPlayViewer, _PATCH_FLAG, True)


_install_patch()
