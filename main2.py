from agents import DQN_PDV, DQN_PDVCfg, DQN, DQNCfg
from MABattle.MABattleV0 import EnvCfg

agent_cfg = DQNCfg()
env_cfg = EnvCfg()

agent = DQN(agent_cfg, env_cfg)
agent.train(5000)
