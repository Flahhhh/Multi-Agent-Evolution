from dataclasses import dataclass

@dataclass
class MERLCfg:
    name: str = "MERL"
    model_type: str = "MAFCQNoisyNet"
    gradient_alg: str = "DQN_PDV"
    fitness_type: str = "Random"

    num_processes: int = 8
    population_size: int = 128
    elite_amount: int = 8
    num_relatives: int = 3

    init_noise_scale: float = 0.1
    gradient_param_scale: float = 0.1

    min_noise_scale: float = 0.015