from dataclasses import dataclass


@dataclass
class SACCfg:
    name: str = "SAC"
    summarization_mode: str = "DEC"
    loss: str = "HuberLoss"
    loss_pdv: str = "HuberLoss"

    optimizer: str = "AdamW"
    lr: float = 0.0001
    pdv_lr: float = 0.0002

    buffer_size: int = 1_024_000
    batch_size: int = 256
    init_alpha: float = 0.4
    alpha_grow_ratio: float = 0.3
    max_alpha: float = 1
    grow_alpha: bool = True

    gamma: float = 0.99

    tau: float = 0.0003
    target_update_freq: int = 256

    num_epoch_steps: int = 16
    num_step_game: int = 1
    num_step_train: int = 8

    exploration_strategy: str = "Noisy Networks"
    sigma_init: float = 0.017
    # min_eps: float = 0.1
    # init_eps: float = 1
    # decay_ratio: float = 0.85

