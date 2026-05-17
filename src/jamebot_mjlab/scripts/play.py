"""Play wrapper: ensures jamebot tasks are registered before mjlab.play.

Jamebot-specific extra flag:
  --enable-perturbations    Keep the push_robot event active during play.
                            Default is no pushes (clean evaluation).
"""

import os
import sys


def main() -> None:
  # Strip the jamebot-only flag from argv before mjlab's tyro CLI sees it,
  # and forward the decision to balance_env_cfg.py via an env var. This has
  # to happen BEFORE `import jamebot_mjlab` because task registration builds
  # the play env cfg at import time.
  if "--enable-perturbations" in sys.argv:
    sys.argv.remove("--enable-perturbations")
    os.environ["JAMEBOT_PLAY_PERTURBATIONS"] = "1"

  import jamebot_mjlab  # noqa: F401 -- side effect: task registration.
  from mjlab.scripts.play import main as _mjlab_play_main

  _mjlab_play_main()


if __name__ == "__main__":
  main()
