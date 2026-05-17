"""Train wrapper: ensures jamebot tasks are registered before mjlab.train."""

import jamebot_mjlab  # noqa: F401 -- side effect: register_mjlab_task() calls.
from mjlab.scripts.train import main as _mjlab_train_main


def main() -> None:
  _mjlab_train_main()


if __name__ == "__main__":
  main()
