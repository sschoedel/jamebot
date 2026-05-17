"""PPO runner config for the jamebot balance task.

Hyperparameters mirror the IsaacLab ``JamebotPPORunnerCfg`` baseline that
Sam tuned (init_noise_std=0.5, [128,128] MLP, entropy 0.02, lr 1e-3,
adaptive KL). Asymmetric actor-critic is achieved automatically because the
env config defines two observation groups ("actor" and "critic").

History / architecture decisions (see research notes):
- Actor obs uses history_length=5 (set in balance_env_cfg.py). Critic obs
  has no history; it sees privileged signals directly.
- MLP for both actor and critic. LSTM is overkill for balance-only and
  complicates rsl-rl deployment.
- No auxiliary state-estimator head yet. The Ji 2022 / DreamWaQ / HIMLoco
  recipe (auxiliary supervised prediction of base_lin_vel from actor obs
  history, concatenated back into the policy input) is the natural next step
  if balance trains but transfer to hardware is poor.

First ablation list once balance trains:
  1. history_length: 1 vs 5 vs 15 vs 30 (20 ms to 600 ms windows)
  2. with/without auxiliary lin-vel estimator head
  3. push event magnitude (currently 0.05 m/s, can scale up to 0.5 m/s)
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def jamebot_balance_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.02,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="jamebot_balance",
    save_interval=25,
    num_steps_per_env=32,
    max_iterations=1500,
    logger="tensorboard",
  )
