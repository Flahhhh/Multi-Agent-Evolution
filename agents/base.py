import datetime
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from gymnasium import Env

from MABattle.MABattleV0 import EnvCfg
from MABattle.utils import make_env
from utils import save_json


class BaseAgent(ABC):
    cfg: dataclass
    env_cfg: EnvCfg

    env: Env
    writer: SummaryWriter

    root_dir: str
    model_dir: str

    def __init__(self, cfg: dataclass, env_cfg: dataclass):
        self.load_cfg(cfg)
        self.get_env(env_cfg)

        self.root_dir = f"logs/{str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))}"

        if not os.path.isdir(self.root_dir):
            os.makedirs(self.root_dir)

        self.model_dir = os.path.join(self.root_dir, "Models")
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        self.writer = SummaryWriter(os.path.join(self.root_dir, f"learning-{self.cfg.name}"), comment="-" + self.cfg.name,
                                    flush_secs=120)

    def get_env(self, env_cfg: dataclass):
        self.env = make_env(env_cfg)
        self.env_cfg = env_cfg

    def load_cfg(self, cfg: dataclass):
        self.cfg = cfg

        for attr, val in vars(cfg).items():
            self.__setattr__(attr, val)

    def train(self, epochs: int):
        save_json(asdict(self.env_cfg), os.path.join(self.root_dir, "env_config.json"))
        print(self.root_dir)
        for epoch in tqdm(range(epochs)):
            metrics = self.train_epoch()
            self.callback(metrics, epoch)

    @abstractmethod
    def train_epoch(self) -> dict[str, int | float]:
        pass

    @abstractmethod
    def callback(self, metrics: dict[str, int | float], epoch: int):
        pass
