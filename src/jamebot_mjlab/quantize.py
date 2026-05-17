"""INT8 dynamic-quantization smoke test for the jamebot actor.

Importing this module installs two monkey patches:

1. ``MjlabOnPolicyRunner.get_inference_policy`` wraps the returned actor in
   a ``SwitchablePolicy`` that owns both the original fp32 model (on the
   training device, typically CUDA) and a dynamically-quantized INT8 copy
   (always on CPU -- PyTorch's INT8 backends are CPU-only).

2. ``ViserPlayViewer.setup`` is extended with a 'Quantization' folder
   containing a checkbox bound to ``policy.use_int8``. Toggling it switches
   the play loop between fp32 and INT8 inference at runtime.

The INT8 path runs the actor on CPU and shuttles obs/action tensors across
the device boundary per step. For play (num_envs=1, ~65-dim obs, 3-dim
action), the overhead is microseconds and irrelevant. The point of the
flag is to verify the policy still behaves under INT8 weights -- this is
the smoke test that gates eventual MCU deployment, where 96 KB of RAM
won't fit a 100 KB fp32 actor.

This patch is opt-in: only installed when the play wrapper sees the
``--quantize-int8`` flag.
"""

from __future__ import annotations

import copy
import io
from typing import TYPE_CHECKING, Any

import torch
import torch.ao.quantization as tq
import viser

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.viewer.viser.viewer import ViserPlayViewer

if TYPE_CHECKING:
  pass

_PATCH_FLAG = "_jamebot_quant_patch_applied"


def _model_size_bytes(model: torch.nn.Module) -> int:
  """Return the on-disk size of ``model``'s state in bytes."""
  buf = io.BytesIO()
  torch.save(model.state_dict(), buf)
  return buf.tell()


def _param_count(model: torch.nn.Module) -> int:
  return sum(p.numel() for p in model.parameters())


def quantize_actor_int8(actor: torch.nn.Module) -> torch.nn.Module:
  """Return a CPU-bound INT8 dynamic-quantized copy of the actor.

  Quantizes nn.Linear weights; activations remain fp32 (dynamic quant).
  The returned module errors if moved to CUDA -- INT8 kernels are CPU-only.
  """
  actor_cpu = copy.deepcopy(actor).cpu().eval()
  return tq.quantize_dynamic(actor_cpu, {torch.nn.Linear}, dtype=torch.qint8)


class SwitchablePolicy:
  """Callable policy that dispatches to either fp32 (cuda) or INT8 (cpu).

  The mjlab viewer calls ``policy(obs) -> actions`` per step and may also
  call ``policy.reset(...)`` on episode reset. Both are forwarded to
  whichever inner policy is active.
  """

  def __init__(
    self,
    fp32_policy: torch.nn.Module,
    int8_policy: torch.nn.Module,
    fp32_device: torch.device | str,
  ) -> None:
    self.fp32 = fp32_policy
    self.int8 = int8_policy
    self.fp32_device = torch.device(fp32_device)
    self.use_int8 = False

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    if not self.use_int8:
      return self.fp32(obs)
    # INT8 path: dispatch on CPU, return on the env's device.
    obs_cpu = obs.detach().to("cpu")
    with torch.no_grad():
      action_cpu = self.int8(obs_cpu)
    return action_cpu.to(self.fp32_device)

  def reset(self, *args: Any, **kwargs: Any) -> None:
    for inner in (self.fp32, self.int8):
      fn = getattr(inner, "reset", None)
      if fn is not None:
        fn(*args, **kwargs)

  def eval(self) -> "SwitchablePolicy":
    self.fp32.eval()
    self.int8.eval()
    return self


def _install_patch() -> None:
  if getattr(MjlabOnPolicyRunner, _PATCH_FLAG, False):
    return

  original_get_inference_policy = MjlabOnPolicyRunner.get_inference_policy
  original_setup = ViserPlayViewer.setup

  def patched_get_inference_policy(
    self: MjlabOnPolicyRunner, device: str | None = None
  ) -> SwitchablePolicy:
    fp32_policy = original_get_inference_policy(self, device=device)
    target_device = device if device is not None else self.device
    print(
      f"[jamebot quant] fp32 actor: {_param_count(fp32_policy):,} params, "
      f"{_model_size_bytes(fp32_policy) / 1024:.1f} KB on {target_device}"
    )
    int8_policy = quantize_actor_int8(fp32_policy)
    print(
      f"[jamebot quant] int8 actor: {_param_count(int8_policy):,} params, "
      f"{_model_size_bytes(int8_policy) / 1024:.1f} KB on cpu (loaded)"
    )
    return SwitchablePolicy(fp32_policy, int8_policy, target_device)

  def patched_setup(self: ViserPlayViewer) -> None:
    original_setup(self)
    policy = getattr(self, "policy", None)
    if not isinstance(policy, SwitchablePolicy):
      return
    with self._server.gui.add_folder("Quantization"):
      cb = self._server.gui.add_checkbox(
        "Use INT8 actor",
        initial_value=policy.use_int8,
      )

      @cb.on_update
      def _(_) -> None:
        policy.use_int8 = cb.value
        active = "INT8 (cpu)" if cb.value else "fp32 (cuda)"
        print(f"[jamebot quant] active actor switched to {active}")

  MjlabOnPolicyRunner.get_inference_policy = patched_get_inference_policy  # type: ignore[method-assign]
  ViserPlayViewer.setup = patched_setup  # type: ignore[method-assign]
  setattr(MjlabOnPolicyRunner, _PATCH_FLAG, True)


def install() -> None:
  """Entry point invoked by the play wrapper when --quantize-int8 is set."""
  _install_patch()
