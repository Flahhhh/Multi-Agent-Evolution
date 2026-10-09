from dataclasses import dataclass

@dataclass
class MERLCfg:
    name: str = "MERL"

    num_processes: int = 8
    elite_amount: int = 8
    num_relatives: int = 3

    init_noise_scale: float = 0.1
    gradient_param_scale: float = 0.1

    min_noise_scale: float = 0.015