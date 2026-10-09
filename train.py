from agents import DQN_PDV, DQN_PDVCfg, DQN, DQNCfg, MERL, MERLCfg, InGA, InGACfg
from MABattle.MABattleV0 import EnvCfg

turn_off = False
if __name__ == "__main__":
    try:
        for _ in range(10):
            agent_cfg = DQN_PDVCfg()
            env_cfg = EnvCfg()

            agent = DQN_PDV(agent_cfg, env_cfg)

            agent.train(100_000)
    except Exception as e:
        print(e)

    if turn_off:
        import os
        os.system("shutdown /s /t 0")
