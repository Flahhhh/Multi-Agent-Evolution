from dataclasses import dataclass


@dataclass(frozen=True)
class DQNCfg:
    name: str = "DQN"
    target: str = "I"
    multi_agent_mode: str = "Independent"  # Independent | Join | Decomposition
    opponent_type: str = "Random"
    loss: str = "HuberLoss"

    device: str = "cpu"
    optimizer_name: str = "AdamW"
    lr: float = 0.0001

    buffer_size: int = 1_024_000
    batch_size: int = 256
    init_alpha: float = 0.4
    alpha_grow_ratio: float = 0.3
    max_alpha: float = 1
    grow_alpha: bool = True
    learning_starts: int = 512

    gamma: float = 0.99

    tau: float = 0.0003
    target_update_freq: int = 256

    num_epoch_steps: int = 1
    num_step_game: int = 8
    num_step_train: int = 2

    exploration_strategy: str = "Noisy Networks"
    sigma_init: float = 0.017
    # min_eps: float = 0.1
    # init_eps: float = 1
    # decay_ratio: float = 0.85

