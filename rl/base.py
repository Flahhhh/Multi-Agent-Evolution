
class BaseAgent:
    def load_config(self, cfg):
        self.cfg = cfg

        for attr, value in vars(cfg).items():
            setattr(self, attr, value)

    def get_action(self) -> list[int]:
        raise NotImplementedError()