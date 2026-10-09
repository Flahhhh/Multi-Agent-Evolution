import gymnasium as gym
from gymnasium.wrappers import TimeLimit

gym.logger.min_level = 40
gym.register(
    "Flah/MABattle-v0",
    entry_point="MABattle.MABattleV0.env:MABattleEnv",
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=200,
    order_enforce=True,
    disable_env_checker=False,
)
