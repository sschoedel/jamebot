from mjlab.tasks.registry import register_mjlab_task

from jamebot_mjlab.tasks.balance.balance_env_cfg import jamebot_balance_env_cfg
from jamebot_mjlab.tasks.balance.rl_cfg import jamebot_balance_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Jamebot-Balance-v0",
  env_cfg=jamebot_balance_env_cfg(),
  play_env_cfg=jamebot_balance_env_cfg(play=True),
  rl_cfg=jamebot_balance_ppo_runner_cfg(),
)
