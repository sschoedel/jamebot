"""Play wrapper: ensures jamebot tasks are registered before mjlab.play.

Jamebot-specific extra flags:
  --enable-perturbations    Keep the push_robot event active during play.
                            Default is no pushes (clean evaluation).
  --quantize-int8           Build an INT8 (CPU) copy of the loaded actor and
                            add a 'Use INT8 actor' checkbox to the viser GUI
                            for live switching between fp32 and INT8.
"""

import sys

TASK_ID = "Mjlab-Jamebot-Balance-v0"


def main() -> None:
  # Parse jamebot-only flags before mjlab's tyro CLI sees them.
  keep_pushes = "--enable-perturbations" in sys.argv
  if keep_pushes:
    sys.argv.remove("--enable-perturbations")
  enable_quant = "--quantize-int8" in sys.argv
  if enable_quant:
    sys.argv.remove("--quantize-int8")

  # Side effects: registers tasks (with push_robot included by default in the
  # play cfg) and monkey-patches the viser viewer.
  import jamebot_mjlab  # noqa: F401
  import jamebot_mjlab.viser_perturb  # noqa: F401

  if enable_quant:
    from jamebot_mjlab import quantize

    quantize.install()

  # Strip push_robot from the registered play cfg unless --enable-perturbations
  # was passed. We mutate after registration because the registry stores cfg
  # instances at import time -- before sys.argv has been parsed -- so we can't
  # gate the decision inside jamebot_balance_env_cfg().
  if not keep_pushes:
    from mjlab.tasks.registry import _REGISTRY

    play_cfg = _REGISTRY[TASK_ID].play_env_cfg
    play_cfg.events.pop("push_robot", None)

  from mjlab.scripts.play import main as _mjlab_play_main

  _mjlab_play_main()


if __name__ == "__main__":
  main()
